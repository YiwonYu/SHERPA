#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/mmai5k_03/projects/panorama_26_Q1"
PANFUSION_DIR="${ROOT_DIR}/PanFusion"
PROMPTS_JSONL="${ROOT_DIR}/prompts.jsonl"
PROMPTS_TXT="${PANFUSION_DIR}/data/demo_from_jsonl.txt"
CKPT_PATH="${PANFUSION_DIR}/logs/4142dlo4/checkpoints/last.ckpt"
CONFIG_PATH="${PANFUSION_DIR}/logs/4142dlo4/config.yaml"
BASE_RUN_ID="4142dlo4"
SEEDS=(0 1234 2026 19990202 9999 28 404 777 5536 8650)
GPUS=(0 1 2 3)

if [[ -x "${PANFUSION_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PANFUSION_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ ! -f "${PROMPTS_JSONL}" ]]; then
  echo "Missing prompts file: ${PROMPTS_JSONL}" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config file: ${CONFIG_PATH}" >&2
  exit 1
fi

cd "${PANFUSION_DIR}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPUS is empty" >&2
  exit 1
fi
GPU_LIST="$(IFS=,; echo "${GPUS[*]}")"
NUM_GPUS="${#GPUS[@]}"
echo "Using GPUs: ${GPU_LIST}"

"${PYTHON_BIN}" - <<'PY'
import json

src = "/home/mmai5k_03/projects/panorama_26_Q1/prompts.jsonl"
dst = "/home/mmai5k_03/projects/panorama_26_Q1/PanFusion/data/demo_from_jsonl.txt"

count = 0
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o.write(json.loads(line)["prompt"].strip() + "\n")
        count += 1

print(f"Wrote {count} prompts to {dst}")
PY

run_seed() {
  local seed="$1"
  local predict_dir_run="${PANFUSION_DIR}/logs/${BASE_RUN_ID}/predict"
  local seed_out="${PANFUSION_DIR}/logs/${BASE_RUN_ID}/predict_${seed}"

  rm -rf "${predict_dir_run}"

  echo "Running seed ${seed} on GPUs ${GPU_LIST}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" WANDB_MODE=offline WANDB_RUN_ID="${BASE_RUN_ID}" \
    "${PYTHON_BIN}" main.py predict \
      -c "${CONFIG_PATH}" \
      --data=Demo \
      --model=PanFusion \
      --ckpt_path=last \
      --trainer.limit_predict_batches=1.0 \
      --trainer.devices="${NUM_GPUS}" \
      --trainer.strategy=ddp \
      --data.init_args.data_dir="${PROMPTS_TXT}" \
      --seed_everything="${seed}"

  if [[ ! -d "${predict_dir_run}" ]]; then
    echo "Predict output folder not found for seed ${seed}" >&2
    return 1
  fi

  rm -rf "${seed_out}"
  mv "${predict_dir_run}" "${seed_out}"
  echo "Saved results in ${seed_out}"
}

for seed in "${SEEDS[@]}"; do
  run_seed "${seed}"
done
