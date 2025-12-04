#!/bin/bash -l

set -e  # exit on first error (optional but recommended)

conda activate 274e
cd ~/mrvi-reproducibility/ || exit 1

# Hyperparameter grids
hiddens=(2 4 8 16 32 64 128)
layers=(1 2 3)

# Defaults
default_px_hidden=32
default_px_layers=1
default_qz_hidden=32
default_qz_layers=1

# Total number of runs (two passes)
total_runs=$(( ${#hiddens[@]} * ${#layers[@]} * 2 ))

# Timing helpers
start_time=$(date +%s)
run_idx=0

format_time() {
  local T=$1
  printf "%02d:%02d:%02d" $((T/3600)) $(((T%3600)/60)) $((T%60))
}

echo "Total runs: $total_runs"
echo "=== Pass 1: vary px (hidden, layers), keep qz at defaults ==="
for px_h in "${hiddens[@]}"; do
  for px_l in "${layers[@]}"; do
    run_idx=$((run_idx + 1))

    now=$(date +%s)
    elapsed=$((now - start_time))
    # avoid division by zero
    if (( run_idx > 0 )); then
      avg_per_run=$(( elapsed / run_idx ))
    else
      avg_per_run=0
    fi
    remaining_runs=$(( total_runs - run_idx ))
    eta=$(( avg_per_run * remaining_runs ))

    elapsed_str=$(format_time "$elapsed")
    eta_str=$(format_time "$eta")

    echo "[Run $run_idx/$total_runs] px_hidden=$px_h px_layers=$px_l qz_hidden=$default_qz_hidden qz_layers=$default_qz_layers | elapsed=$elapsed_str | ETA=$eta_str"

    python -u test.py "$px_h" "$px_l" "$default_qz_hidden" "$default_qz_layers"
  done
done

echo "=== Pass 2: vary qz (hidden, layers), keep px at defaults ==="
for qz_h in "${hiddens[@]}"; do
  for qz_l in "${layers[@]}"; do
    run_idx=$((run_idx + 1))

    now=$(date +%s)
    elapsed=$((now - start_time))
    if (( run_idx > 0 )); then
      avg_per_run=$(( elapsed / run_idx ))
    else
      avg_per_run=0
    fi
    remaining_runs=$(( total_runs - run_idx ))
    eta=$(( avg_per_run * remaining_runs ))

    elapsed_str=$(format_time "$elapsed")
    eta_str=$(format_time "$eta")

    echo "[Run $run_idx/$total_runs] px_hidden=$default_px_hidden px_layers=$default_px_layers qz_hidden=$qz_h qz_layers=$qz_l | elapsed=$elapsed_str | ETA=$eta_str"

    python -u test.py "$default_px_hidden" "$default_px_layers" "$qz_h" "$qz_l"
  done
done

end_time=$(date +%s)
total_elapsed=$((end_time - start_time))
total_elapsed_str=$(format_time "$total_elapsed")

echo "All runs completed in $total_elapsed_str."
