import os
import sys
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

import utils_loading, utils_df
from src.evaluation.metrics import evaluate_fidelity
from src.evaluation.metrics.evaluate_fairness import compute_fairness_metrics
from src.evaluation.metrics.evaluate_fairness import prepare_adult_dataset_attack, prepare_compas_dataset_attack, prepare_diabetes_dataset_attack
from src.evaluation.metrics import plots_fairness, plots_quality
from src.evaluation.metrics.evaluate_quality import evaluate_dataset_models, evaluate_ground_truth_models


_PREPARE_FUNCS = {
    "compas": prepare_compas_dataset_attack,
    "adult": prepare_adult_dataset_attack,
    "diabetes": prepare_diabetes_dataset_attack,
}


def get_prepare_fn(task: str):
    try:
        return _PREPARE_FUNCS[task]
    except KeyError:
        raise ValueError(f"Unsupported task: {task!r}. "
                         f"Available tasks: {list(_PREPARE_FUNCS)}")


def evaluate_fairness_func(
    df_real,
    dataframes_granite,
    dataframes_mixtral7b,
    dataframes_mixtral22b,
    dataframes_llama,
    config,
    save=True,
    n_splits: int = 5,
    random_state: int = 42,
):
    task = config["general"]["task"]
    prep_fn = get_prepare_fn(task)
    experiment = config["general"]["experiment"]

    def agg_metrics_from_df(df, idx_info=None):
        df_shuf = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        parts   = np.array_split(df_shuf, n_splits)

        split_dicts = []
        for part in parts:
            ds   = prep_fn(part)
            m    = compute_fairness_metrics(ds)
            split_dicts.append(m)

        agg = {}
        for key in split_dicts[0]:
            vals = [d[key] for d in split_dicts]
            agg[f"{key}"] = float(np.mean(vals))
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals, ddof=1))

        # 4) merge any extra info (name, icl_*)
        if idx_info:
            agg.update(idx_info)
        return agg

    ############################################################################
    # REAL + Granite-8B
    fairness_results = {}
    metrics_list     = []

    real_info = {"name": "Real"}
    real_m    = agg_metrics_from_df(df_real, real_info)
    fairness_results["dataset_0"] = real_m
    print("REAL {}".format(real_m))
    metrics_list.append(real_m)

    print("Fairness Granite-8B")
    for i, df in enumerate(dataframes_granite, start=1):
        info = {
            "name": config["dataframes"][i-1]["name"],
            "icl_gender": config["dataframes"][i-1]["icl_gender"],
            "icl_records": config["dataframes"][i-1]["icl_records"],
        }
        m = agg_metrics_from_df(df, info)
        fairness_results[f"dataset_{i}"] = m
        metrics_list.append(m)
        print(f"Fairness Metrics for {info['name']}: mean/std →", 
              {k: (m[k+"_mean"], m[k+"_std"]) for k in compute_fairness_metrics(prep_fn(df)).keys()})

    df_fairness = pd.DataFrame(metrics_list)

    ############################################################################
    # Mixtral-7B
    fairness_results_mixtral7b = {}
    metrics_list_mixtral7b     = []

    # Real again
    metrics_list_mixtral7b.append(real_m)
    fairness_results_mixtral7b["dataset_0"] = real_m

    print("Fairness Mixtral-7B")
    for i, df in enumerate(dataframes_mixtral7b, start=1):
        info = {
            "name": config["dataframes"][i-1]["name"],
            "icl_gender": config["dataframes"][i-1]["icl_gender"],
            "icl_records": config["dataframes"][i-1]["icl_records"],
        }
        m    = agg_metrics_from_df(df, info)
        fairness_results_mixtral7b[f"dataset_{i}"] = m
        metrics_list_mixtral7b.append(m)
        print(f"Fairness Metrics for {info['name']}: mean/std →", 
              {k: (m[k+"_mean"], m[k+"_std"]) for k in compute_fairness_metrics(prep_fn(df)).keys()})

    df_fairness_mixtral7b = pd.DataFrame(metrics_list_mixtral7b)


    ############################################################################
    # Mixtral-22B
    fairness_results_mixtral22b = {}
    metrics_list_mixtral22b     = []

    metrics_list_mixtral22b.append(real_m)
    fairness_results_mixtral22b["dataset_0"] = real_m

    print("Fairness Mixtral-22B")
    for i, df in enumerate(dataframes_mixtral22b, start=1):
        info = {
            "name": config["dataframes"][i-1]["name"],
            "icl_gender": config["dataframes"][i-1]["icl_gender"],
            "icl_records": config["dataframes"][i-1]["icl_records"],
        }
        m    = agg_metrics_from_df(df, info)
        fairness_results_mixtral22b[f"dataset_{i}"] = m
        metrics_list_mixtral22b.append(m)
        print(f"Fairness Metrics for {info['name']}: mean/std →", 
              {k: (m[k+"_mean"], m[k+"_std"]) for k in compute_fairness_metrics(prep_fn(df)).keys()})

    df_fairness_mixtral22b = pd.DataFrame(metrics_list_mixtral22b)


    ############################################################################
    # Llama-70B
    fairness_results_llama = {}
    metrics_list_llama     = []

    # Real again
    metrics_list_llama.append(real_m)
    fairness_results_llama["dataset_0"] = real_m

    print("Fairness Llama-70B")
    for i, df in enumerate(dataframes_llama, start=1):
        info = {
            "name": config["dataframes"][i-1]["name"],
            "icl_gender": config["dataframes"][i-1]["icl_gender"],
            "icl_records": config["dataframes"][i-1]["icl_records"],
        }
        m = agg_metrics_from_df(df, info)
        fairness_results_llama[f"dataset_{i}"] = m
        metrics_list_llama.append(m)
        print(f"Fairness Metrics for {info['name']}: mean/std →", 
              {k: (m[k+"_mean"], m[k+"_std"]) for k in compute_fairness_metrics(prep_fn(df)).keys()})

    df_fairness_llama = pd.DataFrame(metrics_list_llama)

    # plot1, name1 = plots_fairness.compute_mean_differences_line(df_fairness, config)
    plot2, name2 = plots_fairness.plot_gender_trends_comparison_attack(df_fairness, df_fairness_mixtral7b, df_fairness_mixtral22b, df_fairness_llama, config)

    plots = [plot2]
    name_plots = [name2]

    if save:
        LOCAL_DIR = config['general']['local_dir']
        DATABASE = config["general"]["database"]
        FIGURES_PATH = config['general']['figures_path'].format(task=task, experiment=experiment)
        PROMPT = config["general"]["prompt_id"]
        MODEL = config["general"]["model"]
        base_save_path = os.path.join(LOCAL_DIR, FIGURES_PATH)
        plots_path = os.path.join(base_save_path, "plots", config["general"]["prompt_id"], config["general"]["model"])
        os.makedirs(plots_path, exist_ok=True)

        for name_plot, plot in zip(name_plots, plots):
            plot_path = os.path.join(plots_path, f"{name_plot}_fairness_{DATABASE}_{PROMPT}_{MODEL}.pdf")
            plot.savefig(plot_path, bbox_inches='tight', format='pdf')
            plt.close(plot)
            print(f"Fairness plots saved to: {plot_path}")


def preprocess_df(df, task, desired_order):
    if task == "adult":
        df["y"] = df["income"]
        df = df[desired_order]
        df.drop(columns=['income', 'race'], inplace=True)
        df["target_group"] = df["race_Black"]

    elif task == "compas":
        df = df[desired_order]
        df["target_group"] = df["race_African-American"]

    elif task == "diabetes":
        df = df[desired_order]
        df["y"] = df["Outcome"]
        df.drop(columns=['Outcome'], inplace=True)
        df["target_group"] = df["Age"] <= 30

    return df


def evaluate_quality_func(df_real, dataframes_granite, dataframes_mixtral_7b, dataframes_mixtral_22b, datafames_llama, config, save=True):
    experiment = config["general"]["experiment"]
    task = config["general"]["task"]
    desired_order = {
        "adult": [
            'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            'marital-status', 'occupation', 'relationship', 'race', 'gender',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income', 'race_White', 'race_Black', 'race_Other', 'y'],
        "compas": ['sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
            'priors_count', 'y', 'age_cat_25-45', 'age_cat_Greaterthan45',
            'age_cat_Lessthan25', 'race_African-American', 'race_Caucasian',
            'c_charge_degree_F', 'c_charge_degree_M'],
        "diabetes": ['Age', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction', 'Glucose',
            'Insulin', 'Outcome', 'Pregnancies', 'SkinThickness']}


    df_real = preprocess_df(df_real, task, desired_order[task])
    real_baselines = evaluate_ground_truth_models(df_real)

    metrics_datasets_granite = []
    print("\nGranite:\n")
    for idx, df in enumerate(dataframes_granite):
        df = preprocess_df(df, task, desired_order[task])
        metric_datasets_granite = (idx, experiment, evaluate_dataset_models(df_real, df, config, config["dataframes"][idx], save))
        metrics_datasets_granite.append(metric_datasets_granite)
    
    print("\n Mixtral 7b:\n")
    metrics_datasets_mixtral_7b = []
    for idx, df in enumerate(dataframes_mixtral_7b):
        df = preprocess_df(df, task, desired_order[task])
        metric_datasets_mixtral_7b = (idx, experiment, evaluate_dataset_models(df_real, df, config, config["dataframes"][idx], save))
        metrics_datasets_mixtral_7b.append(metric_datasets_mixtral_7b)

    print("\n Mixtral 22b:\n")
    metrics_datasets_mixtral_22b = []
    for idx, df in enumerate(dataframes_mixtral_22b):
        df = preprocess_df(df, task, desired_order[task])
        metric_datasets_mixtral_22b = (idx, experiment, evaluate_dataset_models(df_real, df, config, config["dataframes"][idx], save))
        metrics_datasets_mixtral_22b.append(metric_datasets_mixtral_22b)
    
    print("\nLlama:\n")
    metrics_datasets_llama = []
    for idx, df in enumerate(datafames_llama):
        df = preprocess_df(df, task, desired_order[task])
        metric_datasets_llama = (idx, experiment, evaluate_dataset_models(df_real, df, config, config["dataframes"][idx], save))
        metrics_datasets_llama.append(metric_datasets_llama)

    plot, name_plot = plots_quality.plot_evaluation_results_mild_effect_comparison(metrics_datasets_granite, metrics_datasets_mixtral_7b, metrics_datasets_mixtral_22b, metrics_datasets_llama, real_baselines, config)
    # plot2, name_plot2 = plots_quality.plot_evaluation_results_quality(metrics_datasets_granite, metrics_datasets_mixtral_7b, metrics_datasets_mixtral_22b, metrics_datasets_llama, real_baselines, config)

    plots = [plot]
    name_plots = [name_plot]

    if save:
        LOCAL_DIR = config['general']['local_dir']
        DATABASE = config["general"]["database"]
        FIGURES_PATH = config['general']['figures_path'].format(task=task, experiment=experiment)

        base_save_path = os.path.join(LOCAL_DIR, FIGURES_PATH)
        plots_path = os.path.join(base_save_path, "plots", config["general"]["prompt_id"], config["general"]["model"])
        os.makedirs(plots_path, exist_ok=True)

        for name_plot, plot in zip(name_plots, plots):
            plot_path = os.path.join(plots_path, f"{name_plot}_quality_{DATABASE}.pdf")
            plot.savefig(plot_path, bbox_inches='tight', format='pdf')
            plt.close(plot)

        print(f"Quality plots saved to: {plots_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM SDG PROJECT")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./src/configs/evaluation/attack/eval_mild_effect_attack.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="Flag to save outputs (dataframes and plots)"
    )

    task = "adult"
    database = "adult_dataset"
    args = parser.parse_args()
    args.config_path = "./src/configs/evaluation/attack/eval_mild_effect_attack_granite8b.yaml"
    config = utils_loading.load_config(args.config_path)
    config["general"]["task"] = task
    config["general"]["database"] = database

    df_real, dataframes_granite, metadata = utils_df.load_data(config)
    print("df_real", df_real)
    # metrics_fidelity = []
    # print("granite")
    # for conf, df in zip(config["dataframes"], dataframes_granite):
    #     res = evaluate_fidelity.compute_fidelity_metrics(df_real, df, metadata)
    #     print("{}_{}".format(conf["type"], conf["mild_rate"]), res)
    #     metrics_fidelity.append(res)

    path = "./src/configs/evaluation/attack/eval_mild_effect_attack_mixtral22b.yaml"
    config3 = utils_loading.load_config(path)
    config3["general"]["task"] = task
    config3["general"]["database"] = database
    _, dataframes_mixtral_22b, metadata = utils_df.load_data(config3)

    # metrics_fidelity = []
    # print("mixtral-22b")
    # for conf, df in zip(config["dataframes"], dataframes_mixtral_22b):
    #     res = evaluate_fidelity.compute_fidelity_metrics(df_real, df, metadata)
    #     print("{}_{}".format(conf["type"], conf["mild_rate"]), res)
    #     metrics_fidelity.append(res)

    path = "./src/configs/evaluation/attack/eval_mild_effect_attack_llama70b.yaml"
    config2 = utils_loading.load_config(path)
    config2["general"]["task"] = task
    config2["general"]["database"] = database
    _, dataframes_llama, metadata = utils_df.load_data(config2)

    # metrics_fidelity = []
    # print("llama")
    # for conf, df in zip(config["dataframes"], dataframes_llama):
    #     res = evaluate_fidelity.compute_fidelity_metrics(df_real, df, metadata)
    #     print("{}_{}".format(conf["type"], conf["mild_rate"]), res)
    #     metrics_fidelity.append(res)

    path = "./src/configs/evaluation/attack/eval_mild_effect_attack_mixtral7b.yaml"
    config4 = utils_loading.load_config(path)
    config4["general"]["task"] = task
    config4["general"]["database"] = database
    _, dataframes_mixtral_7b, metadata = utils_df.load_data(config4)

    # metrics_fidelity = []
    # print("mixtral-7b")
    # for conf, df in zip(config["dataframes"], dataframes_mixtral_7b):
    #     res = evaluate_fidelity.compute_fidelity_metrics(df_real, df, metadata)
    #     print("{}_{}".format(conf["type"], conf["mild_rate"]), res)
    #     metrics_fidelity.append(res)
        
    evaluate_fairness_func(df_real, dataframes_granite, dataframes_mixtral_7b, dataframes_mixtral_22b, dataframes_llama, config, args.save)
    # evaluate_quality_func(df_real, dataframes_granite, dataframes_mixtral_7b, dataframes_mixtral_22b, dataframes_llama, config, args.save)


if __name__ == "__main__":
    main()