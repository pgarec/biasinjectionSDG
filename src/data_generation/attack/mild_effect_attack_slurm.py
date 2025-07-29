import os
import sys
import argparse

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")

from src.utils.utils_loading import load_config
from src.utils.utils_prompt import read_prompt
from src.utils import utils_df
from src.utils.utils_slurm import run_experiments_slurm


def main():
    parser = argparse.ArgumentParser(description="LLM SDG PROJECT - SLURM Version")
    parser.add_argument(
        "--experiments-path", 
        type=str, 
        default="./src/configs/experiments/attack/exp_mild_effect_attack_slurm.yaml", 
        help="Path to the experiments YAML file"
    )
    parser.add_argument(
        "--container-image",
        type=str,
        default="/gpfs/scratch/bsc98/bsc098069/experiment_data/llm_benchmarking/images/vllm-benchmark-default-nsight.sif",
        help="Singularity container image path (overrides env var SINGULARITY_CONTAINER)"
    )
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default=None,
        help="SLURM partition to use"
    )
    parser.add_argument(
        "--slurm-account",
        type=str,
        default=None,
        help="SLURM account to use"
    )
    parser.add_argument(
        "--max-duration",
        type=str,
        default="01:00:00",
        help="Maximum duration for each SLURM job (HH:MM:SS)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.experiments_path)

    # Set experiment parameters
    config["sdg"]['prompt_id'] = "bias_llms_attack"
    config["sdg"]["icl_records"] = 80
    config['general']["task"] = "adult"
    config['general']["mitigate"] = False
    config['general']["n_iterations"] = 500
    config['general']["database"] = "adult_dataset"
    config["sdg"]["attack"] = True

    # Add SLURM configuration
    if args.container_image:
        config['general']['container_image'] = args.container_image
    if args.slurm_partition:
        config['general']['slurm_partition'] = args.slurm_partition
    if args.slurm_account:
        config['general']['slurm_account'] = args.slurm_account
    config['general']['slurm_max_duration'] = args.max_duration

    cfg_sdg = config["sdg"]
    cfg_general = config['general']
    cfg_paths = config['paths']
    cfg_files = config['files']

    # Ensure we have necessary paths
    if 'results_path' not in cfg_paths:
        cfg_paths['results_path'] = './results/slurm_experiments'

    # Load real data to verify it exists before submitting jobs
    df_real = utils_df.load_real_data(config)
    print(f"✅ Loaded real data with {len(df_real)} records")

    # Prepare prompt path
    path_prompt = os.path.join(
        cfg_paths['local_dir'],
        cfg_paths['prompt_path'].format(
            task=cfg_general['task'],
            model=cfg_sdg['sdg_model'],
            prompt_id=cfg_sdg['prompt_id']
        )
    )
    
    # Verify prompt exists
    if not os.path.exists(path_prompt):
        raise FileNotFoundError(f"Prompt file not found: {path_prompt}")
    
    print(f"✅ Found prompt at: {path_prompt}")
    
    # Submit experiments to SLURM
    print(f"\n🚀 Submitting {len(config['experiments'])} experiments to SLURM...")
    run_experiments_slurm(
        cfg_sdg, 
        cfg_general, 
        cfg_paths, 
        cfg_files, 
        path_prompt, 
        config["experiments"]
    )


if __name__ == "__main__":
    main()