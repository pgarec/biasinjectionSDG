from __future__ import print_function
import argparse
import json
import os.path
import random
import subprocess
import sys
from copy import deepcopy
from typing import List
import os


# path to root code directory in host and container
EXP_HOME_CODE_DIR = os.getenv('EXP_HOME_CODE_DIR', '.')
EXP_CONTAINER_CODE_DIR = os.getenv('EXP_CONTAINER_CODE_DIR', '/usr/local/lib/python3.12/dist-packages')

# path to Slurm executable
EXP_SCRIPT = os.getenv('EXP_SCRIPT')
EXP_SLURM_EXECUTABLE = 'jobs/slurm.sh'


# print ENV vars
print('Environment variable values:')
print('EXP_HOME_CODE_DIR:', EXP_HOME_CODE_DIR)
print('EXP_SLURM_EXECUTABLE:', EXP_SLURM_EXECUTABLE)

# parameters that cannot be modified (it could make the Job stop working)
ILLEGAL_PARAMETERS = {}


def schedule_job(
    user: str,
    queue: str,
    specific_name: str,
    results_path: str,
    args: dict,
    exp_max_duration: str,
) -> None:
    global \
        EXP_HOME_CODE_DIR, \
        EXP_CONTAINER_CODE_DIR, \
        EXP_SLURM_EXECUTABLE, \
        EXP_SLURM_EXECUTABLE

    exp_results_path = os.path.join(results_path, specific_name)
    os.makedirs(exp_results_path, exist_ok=True)

    env = os.environ.copy()
    env["EXP_NAME"] = specific_name
    env["EXP_MAX_DURATION_SECONDS"] = exp_max_duration
    env["EXP_RESULTS_PATH"] = exp_results_path
    env["EXP_HOME_CODE_DIR"] = os.path.abspath(EXP_HOME_CODE_DIR)

    str_env_vars: str = ''
    for key_env_var, value_env_var in args.items():
        str_env_vars += f'{key_env_var}={value_env_var}'
    env["EXP_VARS"] = str_env_vars

    command = f'cat {EXP_SLURM_EXECUTABLE} | envsubst > {exp_results_path}/launcher.sh'
    subprocess.run(command, env=env, shell=True)

    command = f'sbatch -A {user} -q {queue} {exp_results_path}/launcher.sh'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    if result.returncode  == 0:
        output = result.stdout.strip()
        job_id = output.split()[-1]  # Extract the job ID
    else:
        raise RuntimeError(f"Error submitting job: {result.stderr}")

    config_filename = f'config_{job_id}.txt'
    config_path = os.path.join(results_path, config_filename)

    with open(config_path, 'w') as config_file:
        config = f'PYTHONPATH=. python3 ' + ' '.join(sys.argv).replace('{', '"{').replace('}', '}"') + '\n'
        config_file.write(config)

    print("Submitted job with ID", job_id)


def main(
        user: str,
        queue: str,
        name: str,
        results_path: str,
        default_args: dict,
        exp_max_duration: str,
) -> None:
    global ILLEGAL_PARAMETERS

    # default_args = ' '.join([f"{arg}={default_args[arg]}" if default_args[arg] != '' else arg for arg in default_args.keys()])
    arguments = f"--result-dir='{os.path.join(results_path)}' {default_args}"
    schedule_job(
        user,
        queue,
        name,
        results_path,
        default_args,
        exp_max_duration,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher of SDG generator.')
    parser.add_argument('--user', type=str, help='Slurm user', required=True)
    parser.add_argument('--queue', type=str, help='Slurm queue', required=True)
    parser.add_argument('--results-path', type=str, help='Path to store results', required=True)
    parser.add_argument('--name', type=str, default='test', help='Name of the experiment')
    parser.add_argument('--max-duration', type=str, default='00:05:00', help='Slurm queue')
    parser.add_argument('--default-args', type=str, help='Dictionary with the default args')
    args = parser.parse_args()

    default_args = json.loads(args.default_args.replace('\'', '"'))
    os.makedirs(args.results_path, exist_ok=True)

    main(
        args.user,
        args.queue,
        args.name,
        args.results_path,
        default_args,
        args.max_duration,
    )
