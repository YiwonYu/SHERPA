#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
UNIPANO_DIR="${ROOT_DIR}/UniPano"
PROMPTS_JSONL="${ROOT_DIR}/UniPano/test_prompts_ver1.jsonl"
PROMPTS_TXT="${UNIPANO_DIR}/data/demo_from_jsonl.txt"
CKPT_PATH="${UNIPANO_DIR}/logs/${WANDB_RUN_ID:-MISSING}/checkpoints/last.ckpt"
BASE_RUN_ID="${WANDB_RUN_ID:-MISSING}"
SEEDS=(0 1234 2026 19990202 9999 28 404 777 5536 8650)
GPUS=(0 1 2 3 4 5 6 7)

# ── Detect python binary ────────────────────────────────────────────
if [[ -x "/venv/panfusion/bin/python" ]]; then
  PYTHON_BIN="/venv/panfusion/bin/python"
elif [[ -x "${UNIPANO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${UNIPANO_DIR}/.venv/bin/python"
else
  echo "Cannot find panfusion conda env at /venv/panfusion or a local .venv" >&2
  exit 1
fi

# ── Strip any activated virtualenv so subprocesses use the right Python ──
unset VIRTUAL_ENV
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '.venv' | paste -sd:)
echo "Using Python: ${PYTHON_BIN}"


# ── Validate inputs ─────────────────────────────────────────────────
if [[ "${BASE_RUN_ID}" == "MISSING" ]]; then
  echo "WANDB_RUN_ID is not set. Export it before running this script." >&2
  echo "  export WANDB_RUN_ID=<your_run_id>" >&2
  exit 1
fi

CKPT_PATH="${UNIPANO_DIR}/logs/${BASE_RUN_ID}/checkpoints/last.ckpt"

if [[ ! -f "${PROMPTS_JSONL}" ]]; then
  echo "Missing prompts file: ${PROMPTS_JSONL}" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

cd "${UNIPANO_DIR}"

# ── Remove stale config.yaml so LightningCLI uses CLI args ──────────
STALE_CFG="${UNIPANO_DIR}/logs/${BASE_RUN_ID}/config.yaml"
if [[ -f "${STALE_CFG}" ]]; then
  echo "Backing up stale config: ${STALE_CFG} -> ${STALE_CFG}.bak"
  mv "${STALE_CFG}" "${STALE_CFG}.bak"
fi

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPUS is empty" >&2
  exit 1
fi
GPU_LIST="$(IFS=,; echo "${GPUS[*]}")"
NUM_GPUS="${#GPUS[@]}"
echo "Using ${NUM_GPUS} GPUs (${GPU_LIST}) per seed with DDP"

# ── Build demo_from_jsonl.txt from prompts JSONL ────────────────────
mkdir -p "$(dirname "${PROMPTS_TXT}")"
${PYTHON_BIN} - <<'PY'
import json

src = "/workspace/UniPano/test_prompts_ver1.jsonl"
dst = "/workspace/UniPano/data/demo_from_jsonl.txt"

count = 0
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o.write(json.loads(line)["text"].strip() + "\n")
        count += 1

print(f"Wrote {count} prompts to {dst}")
PY

# ── Run a single seed using all GPUs via DDP ────────────────────────
# Lightning's DistributedSampler automatically splits prompts across GPUs
# (use_distributed_sampler=true in config), so each GPU processes a different
# subset of prompts. All ranks write to the same predict/ dir using unique
# pano_id subdirectories, so there are no conflicts.
run_seed() {
  local seed="$1"
  local predict_dir_run="${UNIPANO_DIR}/logs/${BASE_RUN_ID}/predict"
  local seed_out="${UNIPANO_DIR}/logs/${BASE_RUN_ID}/predict_seed_${seed}"

  # Clean previous predict output to avoid stale data
  rm -rf "${predict_dir_run}"

  echo "Running seed ${seed} on GPUs ${GPU_LIST}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  PYTHONNOUSERSITE=1 \
  WANDB_MODE=offline \
  WANDB_RUN_ID="${BASE_RUN_ID}" \
  HF_HOME=/workspace/.hf_home \
    ${PYTHON_BIN} main.py predict \
      --data=Demo \
      --model=UniPano \
      --ckpt_path=last \
      --trainer.limit_predict_batches=1.0 \
      --trainer.devices="${NUM_GPUS}" \
      --trainer.strategy=ddp \
      --trainer.precision=16-mixed \
      --data.init_args.data_dir="${PROMPTS_TXT}" \
      --data.init_args.repeat_predict=1 \
      --seed_everything="${seed}"

  if [[ ! -d "${predict_dir_run}" ]]; then
    echo "Predict output folder not found for seed ${seed}: ${predict_dir_run}" >&2
    return 1
  fi

  rm -rf "${seed_out}"
  mv "${predict_dir_run}" "${seed_out}"
  echo "Saved results for seed ${seed}: ${seed_out}"
}

for seed in "${SEEDS[@]}"; do
  run_seed "${seed}"
done

echo "All ${#SEEDS[@]} seeds finished successfully."
