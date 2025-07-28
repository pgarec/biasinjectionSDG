r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import shlex
import argparse
import asyncio
import base64
import io
import json
import os
import random
import signal
import time
import warnings
import requests
import subprocess
from subprocess import Popen
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from datasets import load_dataset
from PIL.Image import Image
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from concurrent_metrics_checker import ConcurrentMetricsChecker

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation and from parameter.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset
               if data["conversations"][0]["from"] == "human" and data["conversations"][1]["from"] == "gpt"]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    input_requests_api_urls: List[str],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for index, request in enumerate(input_requests):
        yield request, input_requests_api_urls[index]

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def get_base_url(host: str, port: int, server_id: int):
    return f"http://{host}:{port + server_id}"


def get_api_url(host: str, port: int, endpoint: str, server_id: int):
    return f"http://{host}:{port + server_id}{endpoint}"


class Server:
    def __init__(self, server_args: str, output_path: str, with_nsight: bool, port: int, server_id: int, gpu_memory_utilization: float):
        super(Server, self).__init__()
        self.server_args = server_args
        self.output_path = output_path
        self.with_nsight = with_nsight
        self.server_out = None
        self.server_err = None
        self.server_id = server_id
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization

        assert '--port' not in self.server_args
        assert '--gpu-memory-utilization' not in self.server_args

    def run(self) -> Popen:
        try:
            self.server_out = open(os.path.join(self.output_path, f'server_out_{self.server_id}.log'), 'w')
            self.server_err = open(os.path.join(self.output_path, f'server_err_{self.server_id}.log'), 'w')
            command = f'python3 -m vllm.entrypoints.openai.api_server --port={self.port} --gpu-memory-utilization {self.gpu_memory_utilization} {self.server_args}'
            open_subprocess = subprocess.Popen(
                shlex.split(command),
                shell=False,
                cwd='/',
                stdout=self.server_out,
                stderr=self.server_err
            )
            return open_subprocess
        except Exception as e:
            print(e)
            if self.server_out:
                self.server_out.close()
            if self.server_err:
                self.server_err.close()
            raise e

    def terminate(self, open_subprocess: Popen) -> None:
        open_subprocess.send_signal(signal.SIGINT)
        open_subprocess.wait(10)
        open_subprocess.kill()
        open_subprocess.terminate()
        open_subprocess.wait()
        if self.server_out:
            self.server_out.close()
        if self.server_err:
            self.server_err.close()


async def benchmark(
    backend: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    input_requests_api_urls: List[str],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    disable_tqdm: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
):
    if include_nvtx_regions:
        import torch

    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # do initial and warm up test
    unique_api_urls = set(input_requests_api_urls)
    for api_url in unique_api_urls:
        print("Starting initial single prompt test run for url", api_url)
        test_prompt, test_prompt_len, test_output_len, test_mm_content = (input_requests[0])
        
        test_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
        )
        test_output = await request_func(request_func_input=test_input)
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark arguments "
                f"are correctly specified. Error: {test_output.error}")
        else:
            print("Initial test run completed. Starting main benchmark run...")

    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()

    tasks: List[asyncio.Task] = []
    async for request, api_url in get_request(input_requests, input_requests_api_urls, request_rate):
        prompt, prompt_len, output_len, mm_content = request
        request_func_input = RequestFuncInput(model=model_id,
                                              prompt=prompt,
                                              api_url=api_url,
                                              prompt_len=prompt_len,
                                              output_len=output_len,
                                              logprobs=logprobs,
                                              best_of=best_of,
                                              multi_modal_content=mm_content,
                                              ignore_eos=ignore_eos)
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    return output


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    servers = []
    open_server_processes = []
    concurrent_metrics_checkers = []
    try:
        if args.launch_server:
            assert args.multiple_servers > 0
            gpu_memory_utilization = 0.9
            gpu_memory_utilization_by_server = gpu_memory_utilization / args.multiple_servers
            for server_id in range(args.multiple_servers):
                server = Server(args.server_args, args.result_dir, args.launch_server_with_nsight, args.port + server_id, server_id, gpu_memory_utilization_by_server)
                open_server_process = server.run()
                servers.append(server)
                open_server_processes.append(open_server_process)

            # check servers started
            max_wait_for_server_seconds = 300
            init_time = time.time()
            servers_to_start = {get_base_url(args.host, args.port, server_id) + "/metrics" for server_id in range(args.multiple_servers)}
            while len(servers_to_start) > 0 and time.time() - init_time < max_wait_for_server_seconds:
                started_servers: List[str] = []
                for ping_url in servers_to_start:
                    try:
                        if requests.get(ping_url).status_code == 200:
                            started_servers.append(ping_url)
                        else:
                            time.sleep(1)
                    except Exception as e:
                        time.sleep(1)
                time.sleep(5)
                for started_server in started_servers:
                    servers_to_start.remove(started_server)
            if len(servers_to_start) > 0:
                raise Exception("At least one server did not start on time")
            print("Servers started")

        if not args.disable_log_stats:
            # launch metric checkers
            for server_id in range(args.multiple_servers):
                concurrent_metrics_checker = ConcurrentMetricsChecker(
                    args.result_dir,
                    get_base_url(args.host, args.port, server_id) + "/metrics",
                    server_id
                )
                concurrent_metrics_checker.start()
                concurrent_metrics_checkers.append(concurrent_metrics_checker)

        # distribute requests by server
        input_requests_api_urls: List[str] = []
        if not args.launch_server or args.multiple_servers == 1:
            input_requests_api_urls = [get_api_url(args.host, args.port, args.endpoint, 0)] * len(input_requests)
        else:
            server_id = 0
            while len(input_requests_api_urls) < len(input_requests):
                input_requests_api_urls.append(get_api_url(args.host, args.port, args.endpoint, server_id))
                server_id += 1
                if server_id >= args.multiple_servers:
                    server_id = 0
            random.shuffle(input_requests_api_urls)
        values, counts = np.unique(input_requests_api_urls, return_counts=True)
        print(f"Requests to servers. Values: {values}. Counts: {counts}")

        benchmark_result = asyncio.run(
            benchmark(
                backend=backend,
                model_id=model_id,
                tokenizer=tokenizer,
                input_requests=input_requests,
                input_requests_api_urls=input_requests_api_urls,
                logprobs=args.logprobs,
                best_of=args.best_of,
                request_rate=args.request_rate,
                disable_tqdm=args.disable_tqdm,
                selected_percentile_metrics=args.percentile_metrics.split(","),
                selected_percentiles=[
                    float(p) for p in args.metric_percentiles.split(",")
                ],
                ignore_eos=args.ignore_eos,
                include_nvtx_regions=args.include_nvtx_regions,
            ))

        # Save config and results to json
        if args.save_result:
            result_json: Dict[str, Any] = {}

            # Setup
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_json["date"] = current_dt
            result_json["backend"] = backend
            result_json["model_id"] = model_id
            result_json["tokenizer_id"] = tokenizer_id
            result_json["best_of"] = args.best_of
            result_json["num_prompts"] = args.num_prompts

            # Metadata
            if args.metadata:
                for item in args.metadata:
                    if "=" in item:
                        kvstring = item.split("=")
                        result_json[kvstring[0].strip()] = kvstring[1].strip()
                    else:
                        raise ValueError(
                            "Invalid metadata format. Please use KEY=VALUE format."
                        )

            # Traffic
            result_json["request_rate"] = (
                args.request_rate if args.request_rate < float("inf") else "inf")

            # Merge with benchmark result
            result_json = {**result_json, **benchmark_result}

            # Save to file
            base_model_id = model_id.split("/")[-1]
            file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
            if args.result_filename:
                file_name = args.result_filename
            if args.result_dir:
                file_name = os.path.join(args.result_dir, file_name)
            with open(file_name, "w", encoding='utf-8') as outfile:
                json.dump(result_json, outfile)
    finally:

        if not args.disable_log_stats and len(concurrent_metrics_checkers) > 0:
            time.sleep(15)  # for correctly monitoring of metrics
            for concurrent_metrics_checker in concurrent_metrics_checkers:
                try:
                    concurrent_metrics_checker.terminate()
                    concurrent_metrics_checker.join()
                except Exception as e:
                    print(e)
            print('Concurrent checkers terminated')

        if args.launch_server and len(servers) > 0:
            for index, server in enumerate(servers):
                try:
                    server.terminate(open_server_processes[index])
                except Exception as e:
                    print(e)
            print('Servers terminated')

        if args.multiple_servers > 1 and args.use_mps:
            try:
                subprocess.run('echo quit | nvidia-cuda-mps-control', check=True, shell=True)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "random", "hf"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    

    # specifics for launching server and concurrent metric extractor
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch server in addition to benchmark",
    )
    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Args to send to the server when launching. Only useful when passing --launch-server as well",
    )
    parser.add_argument(
        "--multiple-servers",
        type=int,
        default=1,
        help="Launch multiple servers",
    )
    parser.add_argument(
        "--launch-server-with-nsight",
        action="store_true",
        help="Launch server with nsight profile",
    )
    parser.add_argument('--disable-log-stats',
                        action='store_true',
                        help='disable logging statistics'
                        )

    # group for dataset specific arguments

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")


    args = parser.parse_args()
    main(args)