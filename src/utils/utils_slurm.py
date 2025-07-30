import os
import sys
import json
import subprocess
import random
from typing import Dict, List, Any
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")


def schedule_job(
    user: str,
    queue: str,
    specific_name: str,
    results_path: str,
    arguments: str,
    exp_max_duration: str,
    exclusive: bool,
    no_effect: bool,
    home_code_dir: str,
    slurm_executable: str,
    benchmark_executable: str,
    venv_dir: str = None,
) -> str:
    """Schedule a single SLURM job using envsubst approach."""
    
    exp_results_path = os.path.join(results_path, specific_name)
    os.makedirs(exp_results_path, exist_ok=True)
    
    env = os.environ.copy()
    env["EXP_NAME"] = specific_name
    env["EXP_MAX_DURATION_SECONDS"] = exp_max_duration
    env["EXP_RESULTS_PATH"] = exp_results_path
    env["EXP_HOME_CODE_DIR"] = os.path.abspath(home_code_dir)
    env["VENV_DIR"] = venv_dir
    
    command = f'python3 {benchmark_executable} {arguments}'
    env["EXP_BENCHMARK_COMMAND"] = command
    
    # Generate SLURM script from template using envsubst
    slurm_script_path = os.path.join(exp_results_path, 'launcher.sh')
    envsubst_cmd = f'cat {slurm_executable} | envsubst > {slurm_script_path}'
    subprocess.run(envsubst_cmd, env=env, shell=True, check=True)
    
    # Submit the job
    if exclusive:
        sbatch_cmd = f'sbatch -A {user} -q {queue} --exclusive {slurm_script_path}'
    else:
        sbatch_cmd = f'sbatch -A {user} -q {queue} {slurm_script_path}'
    
    if not no_effect:
        result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Extract job ID from output
            job_id = result.stdout.strip().split()[-1]
            print(f"✅ Submitted job '{specific_name}' with ID: {job_id}")
            return job_id
        else:
            print(f"❌ Failed to submit job '{specific_name}': {result.stderr}")
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")
    else:
        print(f"📝 Would submit job '{specific_name}' (no-effect mode)")
        return "DRY_RUN"


def run_experiments_slurm(
    cfg_sdg: Dict[str, Any],
    cfg_general: Dict[str, Any],
    cfg_paths: Dict[str, Any],
    cfg_files: Dict[str, Any],
    prompt_path: str,
    experiments: List[Dict[str, Any]]
) -> None:
    """Run experiments using SLURM job submission with vLLM servers."""
    
    # Environment variables and paths
    home_code_dir = os.getenv('EXP_HOME_CODE_DIR', os.getcwd())
    venv_dir = os.getenv('EXP_VENV_DIR', "/gpfs/scratch/bsc98/bsc098949/venv")
    slurm_executable = os.getenv('EXP_SLURM_EXECUTABLE', './scripts/slurm.sh')
    benchmark_executable = os.getenv('EXP_BENCHMARK_EXECUTABLE', 'src/data_generation/slurm/caller.py')

    # SLURM configuration
    user = cfg_general.get("slurm_user", "bsc98")
    queue = cfg_general.get("slurm_queue", "acc_debug")
    max_duration = cfg_general.get("slurm_max_duration", "01:00:00")
    exclusive = cfg_general.get("slurm_exclusive", False)
    no_effect = cfg_general.get("slurm_no_effect", False)
    
    # vLLM configuration
    model_path = cfg_general.get("model_path", "/gpfs/scratch/bsc98/models/")
    gpu_memory_utilization = cfg_general.get("gpu_memory_utilization", 0.9)

    print(f"Environment variables:")
    print(f"  EXP_HOME_CODE_DIR: {home_code_dir}")
    print(f"  EXP_SLURM_EXECUTABLE: {slurm_executable}")
    print(f"  EXP_BENCHMARK_EXECUTABLE: {benchmark_executable}")
    print(f"\nSLURM settings:")
    print(f"  User: {user}")
    print(f"  Queue: {queue}")
    print(f"  Max duration: {max_duration}")
    print(f"  Exclusive: {exclusive}")
    print(f"  No effect: {no_effect}")
    print(f"\nvLLM settings:")
    print(f"  Model path: {model_path}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"\nSubmitting {len(experiments)} experiments...\n")
    
    submitted_jobs = []
    
    for exp_idx, experiment in enumerate(experiments):

        # Create results directory
        exp_results_path = cfg_paths["synthesized_data"].format(
                    sdg_model=cfg_sdg["sdg_model"],
                    task=cfg_general["task"],
                    prompt_id=cfg_sdg["prompt_id"]
                )
        os.makedirs(exp_results_path, exist_ok=True)
        
        model_path = model_path + experiment.get("model_name")
        # Create experiment config
        exp_config = {
            "cfg_sdg": {**cfg_sdg, **experiment},
            "cfg_general": cfg_general,
            "cfg_paths": cfg_paths,
            "cfg_files": cfg_files,
            "prompt_path": prompt_path,
            "experiment": experiment,
            "model_path": model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        
        # Save experiment config
        config_file = os.path.join(exp_results_path, "experiment_config.json")
        with open(config_file, "w") as f:
            json.dump(exp_config, f, indent=2)
        
        # Create arguments for caller script
        arguments = (
            f"--config '{config_file}' "
            f"--output-dir '{exp_results_path}' "
            f"--model-path '{model_path}' "
            f"--gpu-memory-utilization {gpu_memory_utilization}"
        )
        
        # Additional environment variables
        env_vars = {
            "MODEL_PATH": model_path,
        }
        
        # Schedule the job
        try:
            job_id = schedule_job(
                user=user,
                queue=queue,
                specific_name=job_name,
                results_path=results_base,
                arguments=arguments,
                exp_max_duration=max_duration,
                exclusive=exclusive,
                no_effect=no_effect,
                home_code_dir=home_code_dir,
                slurm_executable=slurm_executable,
                benchmark_executable=benchmark_executable,
                venv_dir=venv_dir,
            )
            
            submitted_jobs.append({
                "job_id": job_id,
                "job_name": job_name,
                "experiment": experiment,
                "config_file": config_file,
                "output_dir": exp_results_path,
            })
            
        except Exception as e:
            print(f"Failed to submit job for experiment {exp_idx}: {e}")
            continue
    
    # Save job submission summary
    summary_file = os.path.join(results_base, "job_submission_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "total_experiments": len(experiments),
            "submitted_jobs": len(submitted_jobs),
            "model_path": model_path,
            "slurm_settings": {
                "user": user,
                "queue": queue,
                "max_duration": max_duration,
                "exclusive": exclusive
            },
            "jobs": submitted_jobs
        }, f, indent=2)
    
    print(f"\n📊 Summary: Submitted {len(submitted_jobs)}/{len(experiments)} jobs")
    print(f"📁 Results will be saved to: {results_base}")
    print(f"📄 Job submission summary saved to: {summary_file}")