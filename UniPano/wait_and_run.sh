#!/usr/bin/env bash
set -euo pipefail

# PIDs of PanFusion processes to wait for
PANFUSION_PIDS=(469437 469630 469631 469632 469633 469634 469635 469636)

echo "[$(date)] Waiting for PanFusion processes to finish..."
echo "PIDs: ${PANFUSION_PIDS[*]}"

while true; do
  alive=0
  for pid in "${PANFUSION_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  if [[ "$alive" -eq 0 ]]; then
    echo "[$(date)] All PanFusion processes finished."
    break
  fi
  echo "[$(date)] Still waiting... ${alive} PanFusion process(es) alive."
  sleep 60
done

echo "[$(date)] Waiting 10s for GPU memory to be released..."
sleep 10

echo "[$(date)] Starting UniPano inference..."
export WANDB_RUN_ID=ud4lfpye
bash /workspace/UniPano/run_prompts_jsonl_seeds.sh 2>&1 | tee /workspace/UniPano/run_all_seeds.log

echo "[$(date)] UniPano inference complete."
