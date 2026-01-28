#!/bin/bash -l

# Set SCC Project
#$ -P depaqlab

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N m2or_ec50_scaf

# Combine output and error files into a single file
#$ -j y

# Specify dir for output files
#$ -o /projectnb/depaqlab/Grant/lora/bash_output/

# Time to run
#$ -l h_rt=48:00:00

# requesting gpu:
#$ -l gpus=1
#$ -l gpu_c=9.0
#$ -l gpu_memory=144G

# enable multiple cores (1 per gpu)
#$ -pe omp 1

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "hey queen"
echo "=========================================================="

# activate environment
cd /projectnb/depaqlab/Grant/lora
export HF_HOME=/projectnb/depaqlab/Grant/lora/saved_models
source venv/bin/activate

# run the Python function
python scripts/train_lorax.py --config /projectnb/depaqlab/Grant/lora/configs/config.yaml
