#!/usr/bin/env python3
"""
Experiment runner for SLURM jobs using vLLM offline (no HTTP server).
This script uses vLLM's Python API directly to generate synthetic data.
"""

import argparse
import json
import os
import sys
import pandas as pd
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")

from vllm import LLM
from vllm import SamplingParams
from src.utils.utils_loading import extract_json_as_dict
from src.utils.utils_prompt import read_prompt
from src.utils import utils_df


def generate_with_vllm_local(
    llm: LLM,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> str:
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens
    )
    
    outputs = llm.generate([prompt], sampling_params)

    print("Outputs: {}".format(outputs))
    for output in outputs:
        return output.outputs[0].text
    
    raise RuntimeError("vLLM did not return any output")


def prompt_synth_tab_vllm(
    df_real: pd.DataFrame,
    prompt: str,
    llm: LLM,
    n_iter: int,
    cfg_copy: dict,
    cfg_general: dict
) -> pd.DataFrame:
    """Generate synthetic tabular data using vLLM offline."""
    from src.utils.utils_prompt import (
        generate_compas_racial_examples,
        generate_adult_examples,
        generate_diabetes_examples,
        generate_drug_examples,
        inject_icl_examples
    )
    
    synth_data = []
    base_tpl = prompt
    
    def _build_icl():
        if cfg_general["task"] == "compas":
            records = generate_compas_racial_examples(cfg_copy, df_real)
        elif cfg_general["task"] == "adult":
            records = generate_adult_examples(cfg_copy, df_real)
        elif cfg_general["task"] == "diabetes":
            records = generate_diabetes_examples(cfg_copy, df_real)
        elif cfg_general["task"] == "drug":
            records = generate_drug_examples(cfg_copy, df_real)
        else:
            raise ValueError(f"Unknown task: {cfg_general['task']}")
        import random
        random.shuffle(records)
        return inject_icl_examples(base_tpl, records)
    
    prompt_with_examples = _build_icl()
    successful = 0
    attempts = 0
    max_attempts = n_iter * 2

    print("Prompt: {}".format(prompt_with_examples))
    
    print(f"🎯 Generating {n_iter} synthetic records...")
    while successful < n_iter and attempts < max_attempts:
        if successful > 0 and successful % 10 == 0:
            prompt_with_examples = _build_icl()
        try:
            response = generate_with_vllm_local(
                llm=llm,
                prompt=prompt_with_examples,
                temperature=cfg_general.get("temperature", 0.7),
                max_tokens=cfg_general.get("max_tokens", 2048)
            )
            record = extract_json_as_dict(response)
            if record:
                synth_data.append(pd.DataFrame([record]))
                successful += 1
                if successful % 50 == 0:
                    print(f"  Generated {successful}/{n_iter} records...")
            else:
                print(f"  Warning: parse failed at attempt {attempts}")
        except Exception as e:
            print(f"  Error generating record: {e}")
        attempts += 1
    
    print(f"✅ Generated {successful}/{n_iter} records after {attempts} attempts")
    return pd.concat(synth_data, ignore_index=True) if synth_data else pd.DataFrame()


def run_single_experiment_job(
    config_path: str,
    output_dir: str,
    model_path: str,
    gpu_memory_utilization: float
) -> dict:
    """Run a single experiment using vLLM offline."""
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    cfg_sdg = config['cfg_sdg']
    cfg_general = config['cfg_general']
    cfg_paths = config['cfg_paths']
    cfg_files = config['cfg_files']
    prompt_path = config['prompt_path']
    experiment = config['experiment']

    # Initialize vLLM LLM instance
    print(f"🌐 Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_memory_utilization
    )

    try:
        # Load real data and prompt
        df_real = utils_df.load_real_data({ 'sdg': cfg_sdg, 'general': cfg_general, 'paths': cfg_paths, 'files': cfg_files })
        prompt = read_prompt(prompt_path)
        print(f"🔬 Experiment: {experiment['bias_type']}, iter={cfg_general['n_iterations']}")

        df_synth = prompt_synth_tab_vllm(
            df_real=df_real,
            prompt=prompt,
            llm=llm,
            n_iter=cfg_general['n_iterations'],
            cfg_copy=cfg_sdg,
            cfg_general=cfg_general
        )

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        filename = cfg_files['synthesized_data'].format(
            database=cfg_general['database'],
            bias=experiment['bias_type'],
            mild_rate=experiment.get('mild_rate', 0),
            icl_records=cfg_sdg.get('icl_records', 0)
        )
        out_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_synth.to_csv(out_path, index=False)

        summary = {
            "status": "success",
            "records_generated": len(df_synth),
            "output_file": out_path,
            **experiment
        }
        print(f"✅ Saved synthetic data to {out_path}")
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback; traceback.print_exc()
        summary = {"status": "failed", "error": str(e), **experiment}

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run LLM SDG experiment with vLLM offline")
    parser.add_argument("--config", required=True, help="Experiment config JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-path", required=True, help="Local model path for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory usage")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    summary = run_single_experiment_job(
        args.config,
        args.output_dir,
        args.model_path,
        args.gpu_memory_utilization
    )
    # Save summary
    with open(os.path.join(args.output_dir, "experiment_summary.json"), 'w') as sf:
        json.dump(summary, sf, indent=2)

    exit(0 if summary.get("status") == "success" else 1)

if __name__ == "__main__":
    main()
