import argparse
import os
import sys
import yaml  
import asyncio

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")

from src.utils.utils_loading import load_config
from src.utils.utils_prompt import read_prompt
from src.utils import utils_df
from src.utils.utils_llm import run_experiments
from src.utils.utils_loading import load_experiments


def main():
    parser = argparse.ArgumentParser(description="LLM SDG PROJECT")
    parser.add_argument("--config-path", type=str, default="./src/configs/config.yaml", help="Path to the configuration file")
    parser.add_argument("--experiments-path", type=str, default="./src/configs/experiments/attack/exp_icl_demonstration_attack.yaml", help="Path to the experiments YAML file")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum number of concurrent experiments")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config["sdg"]['prompt_id'] = "bias_llms_attack"
    config['general']["task"] = "adult"
    config['general']["database"] = "adult_dataset"
    cfg_sdg = config["sdg"]
    cfg_general = config['general']
    cfg_paths = config['paths']
    cfg_files = config['files']

    df_real = utils_df.load_real_data(config)
    experiments_name = "experiments"
    experiments_config = load_experiments(args.experiments_path, experiments_name)
    experiments_files = load_experiments(args.experiments_path, "files")

    for entry in experiments_files:
        cfg_files[entry] = experiments_files[entry]
    
    DATABASE = cfg_general["database"]
    LOCAL_DIR = cfg_paths["local_dir"]

    path_prompt = os.path.join(
        cfg_paths['local_dir'],
        cfg_paths['prompt_path'].format(
            task=cfg_general['task'],
            model=cfg_sdg['sdg_model'],
            prompt_id=cfg_sdg['prompt_id']
        )
    )
    print(f"Reading prompt from: {path_prompt}")
    prompt = read_prompt(path_prompt)
    asyncio.run(run_experiments(cfg_sdg, cfg_general, cfg_paths, cfg_files, prompt, args, experiments_config, LOCAL_DIR, DATABASE, df_real))


if __name__ == "__main__":
    main()
