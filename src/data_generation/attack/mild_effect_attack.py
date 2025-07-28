import argparse
import os
import sys
import asyncio

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")

from src.utils.utils_loading import load_config
from src.utils.utils_prompt import read_prompt
from src.utils import utils_df
from src.utils.utils_llm import run_experiments
from src.utils.utils_loading import load_experiments
from src.utils.utils_slurm import run_experiments_slurm


def main():
    parser = argparse.ArgumentParser(description="LLM SDG PROJECT")
    parser.add_argument("--experiments-path", type=str, default="./src/configs/experiments/attack/exp_mild_effect_attack.yaml", help="Path to the experiments YAML file")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--max-concurrent", type=int, default=11, help="Maximum number of concurrent experiments")
    args = parser.parse_args()


    args.experiments_path = "./src/configs/experiments/attack/exp_mild_effect_attack.yaml"
    config = load_config(args.experiments_path)

    config["sdg"]['prompt_id'] = "bias_llms_attack"
    config["sdg"]["icl_records"] = 80
    config['general']["task"] = "adult"
    config['general']["mitigate"] = False
    config['general']["n_iterations"] = 500
    config['general']["database"] = "adult_dataset"
    config["sdg"]["attack"] = True

    cfg_sdg = config["sdg"]
    cfg_general = config['general']
    cfg_paths = config['paths']
    cfg_files = config['files']

    df_real = utils_df.load_real_data(config)   

    for entry in cfg_files:
        cfg_files[entry] = cfg_files[entry]
    
    DATABASE = cfg_general["database"]
    LOCAL_DIR = cfg_paths["local_dir"]

    path_prompt = os.path.join(
        cfg_paths['local_dir'],
        cfg_paths['prompt_path'].format(
            task=cfg_general['task'],
            prompt_neutrality=cfg_sdg['prompt_neutrality'],
            model=cfg_sdg['sdg_model'],
            prompt_id=cfg_sdg['prompt_id']
        )
    )
    print(f"Reading prompt from: {path_prompt}")
    prompt = read_prompt(path_prompt)
    asyncio.run(run_experiments(cfg_sdg, cfg_general, cfg_paths, cfg_files, prompt, args, config["experiments"], LOCAL_DIR, DATABASE, df_real))


if __name__ == "__main__":
    main()
