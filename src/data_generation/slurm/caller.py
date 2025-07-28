
#!/usr/bin/env python3
"""
Experiment runner for SLURM jobs with vLLM server.
This script launches a vLLM server and generates synthetic data.
"""

import argparse
import json
import os
import sys
import asyncio
import time
import subprocess
import aiohttp
import pandas as pd
from pathlib import Path

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.utils_loading import extract_json_as_dict
from src.utils.utils_prompt import read_prompt
from src.utils import utils_df


class VLLMServer:
    """Manages vLLM server lifecycle within the job."""
    
    def __init__(self, model_path: str, port: int, gpu_memory_utilization: float = 0.9):
        self.model_path = model_path
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.process = None
        self.api_url = f"http://localhost:{port}"
        
    async def start(self):
        """Start the vLLM server."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--disable-log-requests"
        ]
        
        print(f"🚀 Starting vLLM server on port {self.port}...")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        await self._wait_for_server()
        print(f"✅ vLLM server ready at {self.api_url}")
        
    async def _wait_for_server(self, timeout: int = 300):
        """Wait for the server to become responsive."""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(f"{self.api_url}/health") as resp:
                        if resp.status == 200:
                            return
                except:
                    pass
                await asyncio.sleep(2)
        
        raise TimeoutError(f"vLLM server failed to start within {timeout} seconds")
    
    def stop(self):
        """Stop the vLLM server."""
        if self.process:
            print("🛑 Stopping vLLM server...")
            self.process.terminate()
            self.process.wait(timeout=10)
            if self.process.poll() is None:
                self.process.kill()
            print("✅ vLLM server stopped")


async def generate_with_vllm(
    prompt: str,
    api_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> str:
    """Generate text using vLLM API."""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with session.post(
            f"{api_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["text"]
            else:
                error = await response.text()
                raise Exception(f"vLLM API error: {error}")


async def prompt_synth_tab_vllm(
    df_real: pd.DataFrame,
    prompt: str,
    model: str,
    n_iter: int,
    api_url: str,
    cfg_copy: dict,
    cfg_general: dict
) -> pd.DataFrame:
    """Generate synthetic tabular data using vLLM."""
    from src.utils.utils_prompt import (
        generate_compas_racial_examples,
        generate_adult_examples,
        generate_diabetes_examples,
        generate_drug_examples,
        inject_icl_examples
    )
    
    synth_data = []
    base_tpl = prompt
    
    # Generate ICL examples based on task
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
    successful_generations = 0
    attempts = 0
    max_attempts = n_iter * 2  # Allow some failures
    
    print(f"🎯 Generating {n_iter} synthetic records...")
    
    while successful_generations < n_iter and attempts < max_attempts:
        # Rebuild ICL examples every 10 iterations
        if successful_generations % 10 == 0 and successful_generations > 0:
            prompt_with_examples = _build_icl()
        
        try:
            # Generate response
            response = await generate_with_vllm(
                prompt=prompt_with_examples,
                api_url=api_url,
                model=model
            )
            
            # Extract JSON from response
            record = extract_json_as_dict(response)
            if record:
                synth_data.append(pd.DataFrame([record]))
                successful_generations += 1
                
                if successful_generations % 50 == 0:
                    print(f"  Generated {successful_generations}/{n_iter} records...")
            else:
                print(f"  Warning: Failed to parse response (attempt {attempts})")
                
        except Exception as e:
            print(f"  Error generating record: {e}")
        
        attempts += 1
    
    if successful_generations < n_iter:
        print(f"⚠️  Generated only {successful_generations}/{n_iter} records after {attempts} attempts")
    else:
        print(f"✅ Successfully generated {successful_generations} records")
    
    if not synth_data:
        return pd.DataFrame()
    
    return pd.concat(synth_data, axis=0, ignore_index=True)


async def run_single_experiment_job(
    config_path: str,
    output_dir: str,
    model_path: str,
    port: int,
    gpu_memory_utilization: float
):
    """Run a single experiment with vLLM server."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    cfg_sdg = config['cfg_sdg']
    cfg_general = config['cfg_general']
    cfg_paths = config['cfg_paths']
    cfg_files = config['cfg_files']
    prompt_path = config['prompt_path']
    experiment = config['experiment']
    
    # Start vLLM server
    server = VLLMServer(
        model_path=model_path,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    try:
        await server.start()
        
        # Load real data
        full_config = {
            'sdg': cfg_sdg,
            'general': cfg_general,
            'paths': cfg_paths,
            'files': cfg_files
        }
        df_real = utils_df.load_real_data(full_config)
        
        # Read prompt
        prompt = read_prompt(prompt_path)
        
        # Run the experiment
        print(f"🔬 Starting experiment: {experiment['bias_type']}")
        print(f"📊 Configuration: mild_rate={experiment.get('mild_rate', 0)}, "
              f"icl_records={cfg_sdg.get('icl_records', 0)}")
        print(f"🔧 Model: {model_path}")
        print(f"🎯 Iterations: {cfg_general['n_iterations']}")
        
        # Generate synthetic data
        df_synth = await prompt_synth_tab_vllm(
            df_real=df_real,
            prompt=prompt,
            model=os.path.basename(model_path),  # Use model name for API
            n_iter=cfg_general["n_iterations"],
            api_url=server.api_url,
            cfg_copy=cfg_sdg,
            cfg_general=cfg_general
        )
        
        # Save synthetic data
        database = cfg_general["database"]
        filename = cfg_files['synthesized_data'].format(
            database=database,
            bias=experiment['bias_type'],
            mild_rate=experiment.get("mild_rate", 0),
            icl_records=cfg_sdg.get("icl_records", 0)
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the data
        df_synth.to_csv(output_path, index=False)
        
        print(f"✅ Experiment completed successfully!")
        print(f"📁 Data saved to: {output_path}")
        print(f"📊 Generated {len(df_synth)} synthetic records")
        
        # Save experiment summary
        summary = {
            "status": "success",
            "experiment": experiment,
            "records_generated": len(df_synth),
            "output_file": output_path,
            "model": model_path,
            "iterations": cfg_general["n_iterations"],
            "bias_type": experiment['bias_type'],
            "mild_rate": experiment.get("mild_rate", 0),
            "icl_records": cfg_sdg.get("icl_records", 0)
        }
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        summary = {
            "status": "failed",
            "experiment": experiment,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    
    finally:
        # Always stop the server
        server.stop()
    
    # Save summary
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run a single LLM SDG experiment with vLLM server"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model for vLLM"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM server"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM"
    )
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    print(f"🚀 Starting experiment with config: {args.config}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: {args.model_path}")
    print(f"🌐 vLLM port: {args.port}")
    
    # Run the async experiment
    summary = asyncio.run(run_single_experiment_job(
        args.config,
        args.output_dir,
        args.model_path,
        args.port,
        args.gpu_memory_utilization
    ))
    
    if summary["status"] == "success":
        print("🏁 Experiment job completed successfully")
        sys.exit(0)
    else:
        print("🏁 Experiment job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()