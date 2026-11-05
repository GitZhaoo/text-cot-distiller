#!/usr/bin/env bash
# Thin launcher for the text-QA CoT distillation pipeline.
#
# Required environment (never stored in configs):
#   TEXT_COT_API_BASES   comma-separated OpenAI-compatible base URLs
#   TEXT_COT_API_MODEL   teacher model name served by the endpoint
#   TEXT_COT_API_KEY     bearer token
# Convenience: if TEXT_COT_API_KEY is unset, falls back to DEEPSEEK_V4_API_KEY.
#
# Usage:
#   bash run.sh smoke                          # ~24 questions x 2 samples, end-to-end
#   bash run.sh all [LIMIT] [CONFIG]           # full pipeline, optional question cap
#   bash run.sh <stage> [pipeline.py args...]  # ingest|generate|verify|build|validate|stats
#
# Examples:
#   bash run.sh all 500                        # first 500 questions end-to-end
#   bash run.sh generate --num-shards 4 --shard-id 0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── env guards ────────────────────────────────────────────────────────────────
export TEXT_COT_API_KEY="${TEXT_COT_API_KEY:-${DEEPSEEK_V4_API_KEY:-}}"

_missing=()
[ -z "${TEXT_COT_API_BASES:-}" ] && _missing+=("TEXT_COT_API_BASES")
[ -z "${TEXT_COT_API_MODEL:-}" ] && _missing+=("TEXT_COT_API_MODEL")
[ -z "${TEXT_COT_API_KEY:-}" ]   && _missing+=("TEXT_COT_API_KEY")
if [ ${#_missing[@]} -gt 0 ]; then
    echo "[run] ERROR: missing env vars: ${_missing[*]}" >&2
    echo "      e.g. export TEXT_COT_API_BASES=http://host:8000/v1 \\" >&2
    echo "                 TEXT_COT_API_MODEL=<teacher-model> \\" >&2
    echo "                 TEXT_COT_API_KEY=sk-..." >&2
    exit 1
fi

# Teacher endpoints are often intranet/localhost; never route them through a
# global http_proxy (ModelArts nodes set one by default).
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="${no_proxy}"

stage="${1:-smoke}"
shift || true

cd "${SCRIPT_DIR}"

case "$stage" in
    smoke)
        echo "[run] smoke: ~24 questions x 2 samples, end-to-end"
        python3 pipeline.py all --config config.smoke.yaml --force
        ;;
    all)
        limit="${1:-0}"
        config="${2:-config.yaml}"
        [ $# -ge 2 ] && shift 2 || shift
        echo "[run] all: config=${config} limit=${limit}"
        if [ "${limit}" -gt 0 ]; then
            python3 pipeline.py all --config "${config}" --limit "${limit}" --force "$@"
        else
            python3 pipeline.py all --config "${config}" "$@"
        fi
        ;;
    ingest|generate|verify|build|validate|stats)
        echo "[run] stage: ${stage}"
        python3 pipeline.py "${stage}" "$@"
        ;;
    *)
        echo "[run] unknown stage '${stage}' (use smoke|all|ingest|generate|verify|build|validate|stats)" >&2
        exit 1
        ;;
esac
