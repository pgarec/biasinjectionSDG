#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=${EXP_RESULTS_PATH}/log_%j.out
#SBATCH --error=${EXP_RESULTS_PATH}/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres gpu:1
#SBATCH --time=${EXP_MAX_DURATION_SECONDS}

source /gpfs/scratch/bsc98/bsc098949/venv/bin/activate
python src/${EXP_SCRIPT}.py ${EXP_VARS}