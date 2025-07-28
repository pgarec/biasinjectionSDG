export SINGULARITY_CONTAINER="/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/vllm-benchmark-default-nsight.sif"

# Run experiments
python /home/bsc/bsc098949/ibm2/biasinjectionSDG/src/data_generation/attack/mild_effect_attack_slurm.py \
    --slurm-partition gpu \
    --slurm-account myaccount \
    --max-duration 02:00:00