#!/bin/bash -l
#
# Run scripts/evaluate_models.py once on a GPU node.
#
# To run:
#   mkdir -p logs/sge      # SGE needs the -o dir to exist
#   qsub submit_eval.sh

# --- Grid Engine directives (qsub reads these; bash treats them as comments) ---
#$ -P biochemai         # charge the biochemai project
#$ -N loro_eval         # job name
#$ -cwd                 # run from the submit dir (repo root)
#$ -j y                 # merge stderr into stdout
#$ -o logs/sge/         # drop the log in logs/sge/
#$ -l h_rt=4:00:00      # time limit
#$ -pe omp 4            # 4 CPU cores
#$ -l gpus=1            # 1 GPU
#$ -l gpu_c=6.0         # Pascal-or-newer (excludes ancient Kepler/Maxwell cards)
# Exclude the 8 Blackwell nodes (sm_120): torch 2.7.0+cu126 has no sm_120 kernels.
#$ -l h='!scc-701&!scc-702&!scc-703&!scc-708&!scc-b01&!scc-b02&!scc-b03&!scc-b04'

echo "Run on $(hostname) | GPU $CUDA_VISIBLE_DEVICES"
uv run python scripts/evaluate_models.py
