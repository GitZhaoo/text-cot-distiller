# text-cot-distiller

用外部 teacher API（OpenAI 兼容接口）为文本 QA 数据合成可训练的思维链（CoT），
产出可直接用于 SFT 的 `messages` 格式数据。核心范式是 **teacher 采样 + 可验证奖励的
拒绝采样（rejection sampling）+ 选优**。

```
ingest → generate → verify → build → validate
原始QA   采样N条CoT   答案校验+质量/难度过滤+选优   拼成SFT格式   产物自检
```

## 1. 方法依据

主流做法一致：不是"一次生成一条 CoT 就用"，而是每题采样 N 条、只留**答案可验证正确**的：

| 来源 | 关键做法 | 本 pipeline 的落地 |
|---|---|---|
| [LIMO](https://arxiv.org/abs/2502.03387) / s1 | 少量高质量、长链推理样本即可蒸馏；质量 > 数量 | 严格答案校验 + 质量过滤，宁缺毋滥 |
| [DART-Math](https://arxiv.org/abs/2407.13690)（NeurIPS'24） | Difficulty-Aware Rejection Sampling：简单题一次就过 → 数据偏简单；对难题多采样 | `min_correct/max_correct/drop_too_easy` 难度过滤 + `n_samples` 可调 |
| [Synthetic Data survey (arXiv:2503.14023)](https://arxiv.org/abs/2503.14023) | 生成后必须有过滤/验证环节，否则噪声放大 | verify 阶段做格式、重复、截断、拒答、答案一致性检查 |
| 拒绝采样工程经验 | sample N，按可验证信号选最好的；能拿到大部分 RL 收益、成本低得多 | 每题采样 N 条，仅保留答案正确的，默认选 **最短正确链** |

几个容易踩的坑，代码里已处理：

1. **teacher 的 `reasoning` 字段常是草稿**（"We need answer. Need reason..."），
   而 `content` 里才是打磨过的推导 → 默认 `verify.trace_source: content`，
   仅在 content 只有答案时回退到 reasoning。
2. **`--limit` 不能写进 records.jsonl**，否则后续全量跑会静默复用截断文件 → 只在 generate/verify 时切片。
3. **问题文本 hash 去重不安全**（同题不同来源会误删）→ 默认关闭，按原始行号生成 `uid`。
4. **不能一次性 `gather` 全部任务**（100k 题 × N 采样会爆内存）→ 固定大小 worker 池。
5. **429 必须 ≥5s 退避**，总并发起始 ≤50。
6. **端点/模型/密钥只走环境变量**，任何配置文件、日志、产物都不落盘 API 信息。

## 2. 快速开始

依赖：Python 3.10+，`pip install -r requirements.txt`。

```bash
# 0) 注入 API 信息（仅环境变量，不写入任何文件）
export TEXT_COT_API_BASES="http://host1:8000/v1,http://host2:8000/v1"  # 多端点轮询
export TEXT_COT_API_MODEL="your-teacher-model"
export TEXT_COT_API_KEY="sk-..."

# 1) 准备配置：从 example 复制，改输入输出路径
cp config.smoke.example.yaml config.smoke.yaml   # gitignored
cp config.example.yaml     config.yaml           # gitignored
# 编辑 paths.inputs（支持 verl RL parquet 或 question/answer jsonl/json）与 paths.work_dir

# 2) 冒烟（~24 题 × 2 采样，端到端几分钟）
bash run.sh smoke

# 3) 正式跑
bash run.sh all          # 或 python pipeline.py all --config config.yaml
```

断点续跑：generate 按 `(uid, sample_idx)` 跳过已完成项，Ctrl-C 后重跑同一命令即可；
半行 JSON 读取时自动忽略。

多进程/多机分片（各进程只写自己的 shard，verify 自动合并）：

```bash
for i in 0 1 2 3; do
  nohup python pipeline.py generate --config config.yaml \
      --num-shards 4 --shard-id $i > logs/gen_$i.log 2>&1 &
done
python pipeline.py verify --config config.yaml
```

## 3. 项目结构

```
├── pipeline.py                 # 主编排：ingest/generate/verify/build/validate/stats/all
├── api_client.py               # 异步 OpenAI 兼容客户端（多端点轮询、重试、reasoning 归一化）
├── answer_check.py             # 答案归一化/比对 + CoT 质量过滤
├── run.sh                      # 薄启动器（env 校验 + no_proxy + 参数透传）
├── config.example.yaml         # 生产配置模板（tracked，无任何 API 信息）
├── config.smoke.example.yaml   # 冒烟配置模板（tracked）
├── requirements.txt
├── .gitignore                  # 忽略 config.yaml / runs/ 等本地内容
└── runs/<name>/                # 每次运行的产物（gitignored）
    ├── records.jsonl           # 规范化后的题目
    ├── gen/shard_XXX.jsonl     # 原始采样（断点续跑依据）
    ├── accepted.jsonl / rejected.jsonl
    ├── stats.json              # 接受率、拒绝原因、难度分布、token 用量
    └── out/                    # 最终 SFT 数据 + REPORT.md
```

## 4. 输入格式

`paths.inputs` 支持 parquet / jsonl / json，自动识别两种 schema：

1. **verl RL parquet**：`prompt`（messages）+ `reward_model.ground_truth` +
   `extra_info.{problem_type, data_source}`；
2. **普通 QA**：顶层 `question|query|problem|input` + `answer|ground_truth|solution|label` 字段，
   或 `messages` 里最后一个 user / assistant 轮。

`problem_type` 决定答案比对策略（见下），缺失时按 GT 形态推断
（单字母 → multiple choice；纯数字 → numerical；其余 → free-form）。

## 5. 校验与选优逻辑

`verify` 对每条采样依次判定：

1. **API/格式**：`finish_reason=length` 截断、拒答、空答案、控制字符 → 丢弃；
2. **质量**：推理长度区间、重复 n-gram 比例（默认 >0.35 丢弃）、require_reasoning；
3. **答案正确性**（`answer_check.py`，按 problem_type 分发）：
   - `multiple choice`：选项字母归一化比较（`\text{A}` / `(A)` / `A.` 均可）；
   - `numerical` / `regression` / `math`：数字解析比较（`1,234`、`3/4`、`45%`、科学计数），
     失败回退文本归一化；
   - `free-form`：归一化后 exact，否则 token-F1 ≥ 阈值（默认 0.75，偏严）；
4. **难度过滤**（可选）；
5. **trace 去重**：不同题目的最优链若开头雷同（归一化前缀指纹相同）→ 后者整题丢弃；
6. **选优**：在正确样本中按 `select` 策略选一条；默认最短（蒸馏经验：更短的正确链更易学）。

`build` 输出格式：

```json
{
  "messages": [
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<think>...</think>\n<answer>...</answer>"}
  ],
  "images": [], "videos": [], "audios": [],
  "_source": "<dataset>",
  "extra_info": {
    "data_type": "text", "data_source": "...", "problem_type": "...",
    "uid": "...", "teacher": "...", "ground_truth": "...",
    "n_samples": 4, "n_correct": 2
  }
}
```

`validate` 重新解析产物并断言：结构、think 标签、最小长度、答案仍与 GT 一致、
uid/推理链无重复，不通过时退出码为 1（方便接 CI）。

## 6. 配置要点

| 字段 | 含义 |
|---|---|
| `TEXT_COT_API_BASES` (env) | 端点列表，逗号分隔，轮询；每个端点独立并发预算 |
| `TEXT_COT_API_MODEL` (env) | teacher 模型名 |
| `TEXT_COT_API_KEY` (env) | bearer token |
| `api.max_concurrency` | **每个端点**并发；总并发 = 端点数 × 该值 |
| `api.rpm` | 可选的全局每分钟请求上限 |
| `sampling.n_samples` | 每题采样数（难题可加大，DART-Math） |
| `sampling.temperatures` | 按 sample_idx 循环 |
| `sampling.send_seed` | 传入确定性 seed（网关不支持 `seed` 时置 false） |
| `sampling.max_tokens` | 要大于最长 CoT；截断样本会被丢弃 |
| `generate.system_prompt` | 要求 `<think>`/`<answer>` 格式；建议与学生模型训练时一致 |
| `ingest.max_records` | 冒烟用总题数上限，0 = 全量 |
| `verify.trace_source` | `content`（推荐）/ `reasoning` |
| `verify.strip_meta` | 去掉 "We need answer..." 这类自我指令行 |
| `verify.select` | `shortest_correct`（默认）/ `first_correct` / `majority_then_shortest` |
| `verify.min_correct` / `max_correct` / `drop_too_easy` | 难度过滤 |
| `verify.dedup_traces` | 丢弃开头雷同的重复推理链 |
| `verify.freeform_min_f1` | 自由题 token-F1 阈值（0.75 偏严，可降到 0.6 放宽） |
| `output.split_by_source` | 额外按数据集拆成 `<source>.json/.parquet` |
| `output.name` | 输出文件名前缀 |

**Prompt 经验**：格式指令放在 system prompt；部分量化部署在 user 消息里出现
`<answer>` 标签字样时会退化输出（实测复现），默认 `user_suffix` 为空。

## 7. 吞吐与成本估算

单条约 135 prompt token + 200~2000 completion token。16 并发在单端点上大致
每秒 1~5 条（取决于生成长度）。先跑 `--limit 200` 测实际速率，再按
`总题数 × n_samples ÷ 实测速率` 估时。generate 日志有 `req/s` 和 ETA；
`stats.json` 的 `teacher_token_usage` 可算成本。

## 8. 接入训练前检查清单

- [ ] `stats.json` 接受率、各 problem_type 分布符合预期；
- [ ] `validate` 通过（退出码 0）；
- [ ] 抽检 `out/text_cot.jsonl` 的推理风格是否符合预期（打磨稿 vs 草稿）；
- [ ] 并入正式训练集时，同步更新训练仓的数据集清单与文档；
- [ ] 长时间任务保留 `gen/*.jsonl`（断点续跑依据），确认产物后才清理。
