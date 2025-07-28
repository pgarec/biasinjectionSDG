import random
import os
import json
import litellm, os
import asyncio
import sys
import pandas as pd
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")
from src.utils.utils_loading import extract_json_as_dict, save_csv_async
from src.utils.utils_prompt import (generate_compas_racial_examples, 
                                    generate_adult_examples,
                                    generate_diabetes_examples,
                                    generate_drug_examples,
                                    inject_icl_examples)
from src.utils.utils_mitigation import mitigate_bias, train_encoder


async def run_experiments(cfg_sdg, cfg_general, cfg_paths, cfg_files, prompt, args, experiments, LOCAL_DIR, DATABASE, df_real, df_reference=None):
    sem = asyncio.Semaphore(args.max_concurrent)
    print("Tasks being launched:", [f"{i}: {exp['bias_type']}" for i, exp in enumerate(experiments)])

    bundle = None
    if cfg_general["mitigate"]:
        bundle = train_encoder(
            df_train=df_reference,
            latent_dim=32,
            emb_dim=8,
            hidden_dims=[128, 64],
            n_epochs=50,
        )

    tasks = []
    for task_id, experiment in enumerate(experiments):
        task = asyncio.create_task(run_single_experiment(
            experiment, cfg_sdg, cfg_general, cfg_paths, cfg_files, prompt, args,
            LOCAL_DIR, DATABASE, sem, task_id, df_real, df_reference, bundle
        ))
        print(f"👾 Created task {task_id}")
        tasks.append(task)
        await asyncio.sleep(0.1)  

    await asyncio.gather(*tasks)


async def run_single_experiment(
    experiment_config, cfg_sdg, cfg_general, cfg_paths, cfg_files, prompt, args,
    LOCAL_DIR, DATABASE, sem: asyncio.Semaphore, task_id: int, df_real, df_reference = None, bundle=None
):
    async with sem:
        print(f"🔬 Starting experiment {task_id}: {experiment_config['bias_type']}")

        cfg_copy = cfg_sdg.copy()
        for entry in experiment_config:
            cfg_copy[entry] = experiment_config[entry]

        df_synth = await prompt_synth_tab_rits(
            df_real=df_real,
            prompt=prompt,
            model=cfg_copy["sdg_model"],
            n_iter=cfg_general["n_iterations"],
            role='user',
            api_endpoint=cfg_copy['rits_api_endpoint'],
            task_id=task_id,
            cfg_copy=cfg_copy,
            cfg_general=cfg_general,
            df_reference=df_reference,
            bundle=bundle
        )

        if args.save:
            PATH_SYNTHETIC_DATA = cfg_paths["synthesized_data"].format(
                sdg_model=cfg_copy["sdg_model"],
                task=cfg_general["task"],
                prompt_neutrality=cfg_copy["prompt_neutrality"],
                icl_gender=cfg_copy["icl_gender"],
                prompt_id=cfg_copy["prompt_id"]
            )
            os.makedirs(PATH_SYNTHETIC_DATA, exist_ok=True)
            print(os.path.join(PATH_SYNTHETIC_DATA, cfg_files['synthesized_data_prompt'].format(database=DATABASE, bias=cfg_copy['bias_type'], mild_rate=cfg_copy["mild_rate"], icl_records=cfg_copy["icl_records"])))
            await save_csv_async(
                df_synth,
                LOCAL_DIR,
                os.path.join(PATH_SYNTHETIC_DATA, cfg_files['synthesized_data'].format(database=DATABASE, bias=cfg_copy['bias_type'], mild_rate=cfg_copy["mild_rate"], icl_records=cfg_copy["icl_records"]))
            )
        print(f"✅ Finished experiment {task_id}: {experiment_config['bias_type']}")


async def prompt_synth_tab_rits(
        df_real: pd.DataFrame,
        prompt: str,
        model: str,
        n_iter: int,
        role: str,
        api_endpoint: str,
        task_id: int = 0,
        cfg_copy: dict | None = None,
        cfg_general: dict | None = None,
        df_reference: pd.DataFrame = None,
        bundle=None
) -> pd.DataFrame:

    synth_data, loop = [], asyncio.get_running_loop()
    pbar = tqdm(
        total=n_iter, desc=f"Experiment {task_id}", position=task_id,
        leave=True, dynamic_ncols=True, ncols=100, file=sys.stdout
    )
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
            raise "[ERROR] Wrong task"
        random.shuffle(records)
        print(records)
        if cfg_general["mitigate"]:
            records = mitigate_bias(records, df_reference, cfg_general, bundle)
        return inject_icl_examples(base_tpl, records)

    prompt_copy = await loop.run_in_executor(None, _build_icl)
    k = 0

    while k < n_iter:
        if k % 10 == 0 and k != 0:
            prompt_copy = await loop.run_in_executor(None, _build_icl)
        msg = await prompt_model_rits_async(
            model=model, prompt=prompt_copy,
            role=role, rits_api_endpoint=api_endpoint
        )

        record = extract_json_as_dict(msg)
        if not record:
            print(f"[{model}] Task {task_id}: Skipping empty response…")
            continue

        synth_data.append(pd.DataFrame([record]))
        rows = 1
        k += rows
        pbar.update(rows)

    pbar.close()
    return pd.concat(synth_data, axis=0)


async def prompt_model_rits_async(model: str,
                            prompt: str,
                            role: str,
                            rits_api_endpoint: str) -> str:
    def call_completion():
        response = litellm.completion(
            api_base=rits_api_endpoint,
            temperature=0.7,
            model=model,
            messages=[
                {"role": role, "content": f"{prompt}\n"}
            ],
            extra_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
            api_key="fake-key",
        )
        return response

    response = await asyncio.to_thread(call_completion)
    return response.choices[0].message['content']