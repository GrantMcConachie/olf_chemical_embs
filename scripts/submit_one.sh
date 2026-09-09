#!/bin/bash -l
#
# Submit a SINGLE LORO run for one config file.
# Use this to re-run individual receptors (e.g. tasks that failed in the array
# sweep). The config is passed as an argument -- there is NO -t array here, so
# each qsub launches exactly one training run.
#
# To run one config:
#   qsub submit_one.sh configs/loro_sweep/AgOr9.yaml
#
# To re-run a batch of configs:
#   for c in AgOr9 AgOr50 AgOr56; do qsub submit_one.sh configs/loro_sweep/$c.yaml; done

# --- Grid Engine directives (qsub reads these; bash treats them as comments) ---
#$ -P biochemai         # charge the biochemai project
#$ -N loro_one          # job name
#$ -cwd                 # run from the submit dir (repo root)
#$ -j y                 # merge stderr into stdout
#$ -o logs/sge/         # drop the log in logs/sge/
#$ -l h_rt=4:00:00      # time limit
#$ -pe omp 4            # 4 CPU cores
#$ -l gpus=1            # 1 GPU
#$ -l gpu_c=7.0         # Pascal-or-newer (excludes ancient Kepler/Maxwell cards)
# Exclude the 8 Blackwell nodes (compute capability 12.0 / sm_120): the installed
# torch 2.7.0+cu126 has no sm_120 kernels, so jobs there die with
# "CUDA error: no kernel image is available for execution on the device".
# gpu_c is a MINIMUM, so it can't screen these out -- exclude them by host.
#$ -l h='!scc-701&!scc-702&!scc-703&!scc-708&!scc-b01&!scc-b02&!scc-b03&!scc-b04'

# --- one run: config comes from the command line ---
config="$1"
if [ -z "$config" ]; then
  echo "ERROR: no config given. Usage: qsub submit_one.sh <config.yaml>" >&2
  exit 1
fi

echo "Run on $(hostname) | GPU $CUDA_VISIBLE_DEVICES | $config"
uv run python scripts/train_lorax.py --config "$config"
