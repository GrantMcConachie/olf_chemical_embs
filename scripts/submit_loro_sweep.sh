#!/bin/bash -l
#
# Leave-One-Receptor-Out sweep: 50 receptors -> 50 parallel array tasks.
#
# To run:
#   uv run python scripts/gen_loro_configs.py   # 1. write the 50 configs
#   mkdir -p logs/sge                           # 2. SGE needs this dir to exist
#   qsub submit_loro_sweep.sh                   # 3. submit all 50 tasks

# --- Grid Engine directives (qsub reads these; bash treats them as comments) ---
#$ -P biochemai         # charge the biochemai project
#$ -N loro_sweep        # job name
#$ -cwd                 # run from the submit dir (repo root)
#$ -j y                 # merge stderr into stdout
#$ -o logs/sge/         # drop each task's log in logs/sge/
#$ -l h_rt=4:00:00      # time limit PER TASK -- tune after one run finishes
#$ -pe omp 4            # 4 CPU cores per task
#$ -l gpus=1            # 1 GPU per task
#$ -l gpu_c=7.0         # any GPU P100-or-newer (huge free pool)
#$ -t 1-21             # 50 tasks; each gets a unique $SGE_TASK_ID (1..50)
#$ -tc 10              # run at most 10 tasks at once

# --- one task's work: pick my config by my task number, then train ---
#configs=(configs/loro_sweep/*.yaml)          # all 50 configs, sorted by name
#configs=(configs/loro_sweep{1,2,3}/AgOr{1,4,6,9,10,11,12,13,15,18,20,21,30,38,39,46,48,50,56,57,75}.yaml)
configs=(configs/loro_sweep_bp3/AgOr{1,4,6,9,10,11,12,13,15,18,20,21,30,38,39,46,48,50,56,57,75}.yaml)
config=${configs[$((SGE_TASK_ID - 1))]}      # task 1 -> configs[0], task 2 -> configs[1], ...
[[ -f "$config" ]] || { echo "Missing config: $config (task $SGE_TASK_ID)" >&2; exit 1; }

echo "Task $SGE_TASK_ID on $(hostname) | GPU $CUDA_VISIBLE_DEVICES | $config"
uv run python scripts/train_lorax.py --config "$config"
