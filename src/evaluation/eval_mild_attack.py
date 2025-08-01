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
    model_dataframes,
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
        parts = np.array_split(df_shuf, n_splits)

        split_dicts = []
        for part in parts:
            ds = prep_fn(part)
            m = compute_fairness_metrics(ds)
            split_dicts.append(m)

        agg = {}
        for key in split_dicts[0]:
            vals = [d[key] for d in split_dicts]
            agg[f"{key}"] = float(np.mean(vals))
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals, ddof=1))

        # Merge any extra info (name, icl_*)
        if idx_info:
            agg.update(idx_info)
        return agg

    # Process real dataset
    real_info = {"name": "Real"}
    real_m = agg_metrics_from_df(df_real, real_info)
    print("REAL {}".format(real_m))

    # Dictionary to store results for all models
    all_fairness_results = {}
    all_dataframes = {}

    # Process each model
    for model_name, dataframes in model_dataframes.items():
        print(f"\nFairness {model_name}")
        
        fairness_results = {}
        metrics_list = []
        
        # Add real dataset metrics
        fairness_results["dataset_0"] = real_m
        metrics_list.append(real_m)
        
        # Process synthetic datasets for this model
        for i, df in enumerate(dataframes, start=1):
            info = {
                "name": config["dataframes"][i-1]["name"],
                "icl_records": config["dataframes"][i-1]["icl_records"],
            }
            m = agg_metrics_from_df(df, info)
            fairness_results[f"dataset_{i}"] = m
            metrics_list.append(m)
            print(f"Fairness Metrics for {info['name']}: mean/std →", 
                  {k: (m[k+"_mean"], m[k+"_std"]) for k in compute_fairness_metrics(prep_fn(df)).keys()})
        
        all_fairness_results[model_name] = fairness_results
        all_dataframes[model_name] = pd.DataFrame(metrics_list)

    # Generate plots
    plot, name_plot = plots_fairness.plot_gender_trends_comparison_attack_flexible(
        all_dataframes, config
    )

    if save:
        LOCAL_DIR = config['general']['local_dir']
        DATABASE = config["general"]["database"]
        FIGURES_PATH = config['general']['figures_path'].format(task=task, experiment=experiment)
        PROMPT = config["general"]["prompt_id"]
        MODEL = config["general"]["model"]
        base_save_path = os.path.join(LOCAL_DIR, FIGURES_PATH)
        plots_path = os.path.join(base_save_path, "plots", PROMPT, MODEL)
        os.makedirs(plots_path, exist_ok=True)

        plot_path = os.path.join(plots_path, f"{name_plot}_fairness_{DATABASE}_{PROMPT}_{MODEL}.pdf")
        plot.savefig(plot_path, bbox_inches='tight', format='pdf')
        plt.close(plot)
        print(f"Fairness plots saved to: {plot_path}")

    return all_fairness_results, all_dataframes


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


def evaluate_quality_func(df_real, model_dataframes, config, save=True):
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

    # Dictionary to store metrics for all models
    all_metrics_datasets = {}

    # Process each model
    for model_name, dataframes in model_dataframes.items():
        print(f"\n{model_name}:\n")
        metrics_datasets = []
        
        for idx, df in enumerate(dataframes):
            df = preprocess_df(df, task, desired_order[task])
            metric_dataset = (idx, experiment, evaluate_dataset_models(
                df_real, df, config, config["dataframes"][idx], save))
            metrics_datasets.append(metric_dataset)
        
        all_metrics_datasets[model_name] = metrics_datasets

    # Generate plots
    plot, name_plot = plots_quality.plot_evaluation_results_mild_effect_comparison_flexible(
        all_metrics_datasets, real_baselines, config
    )

    if save:
        LOCAL_DIR = config['general']['local_dir']
        DATABASE = config["general"]["database"]
        FIGURES_PATH = config['general']['figures_path'].format(task=task, experiment=experiment)

        base_save_path = os.path.join(LOCAL_DIR, FIGURES_PATH)
        plots_path = os.path.join(base_save_path, "plots", config["general"]["prompt_id"], config["general"]["model"])
        os.makedirs(plots_path, exist_ok=True)

        plot_path = os.path.join(plots_path, f"{name_plot}_quality_{DATABASE}.pdf")
        plot.savefig(plot_path, bbox_inches='tight', format='pdf')
        plt.close(plot)

        print(f"Quality plots saved to: {plots_path}")

    return all_metrics_datasets


def load_model_data(model_configs, task, database):
    model_dataframes = {}
    
    for model_name, config_path in model_configs.items():
        config = utils_loading.load_config(config_path)
        config["general"]["task"] = task
        config["general"]["database"] = database
        _, dataframes, _ = utils_df.load_data(config)
        model_dataframes[model_name] = dataframes
        print(f"Loaded data for {model_name}: {len(dataframes)} dataframes")
    
    return model_dataframes


def evaluate_fidelity_for_models(df_real, model_dataframes, metadata):
    for model_name, dataframes in model_dataframes.items():
        print(f"\n{model_name} Fidelity:")
        for conf_idx, df in enumerate(dataframes):
            # Assuming config structure is available globally or passed
            res = evaluate_fidelity.compute_fidelity_metrics(df_real, df, metadata)
            print(f"Dataset {conf_idx}: {res}")


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

    task = "compas"
    database = "compas_racial_dataset"
    args = parser.parse_args()

    model_configs = {
        "mixtral-8x7b": "./src/configs/evaluation/attack/eval_mild_effect_attack_mixtral7b.yaml",
    }
    
    # Load base configuration (using first model's config as base)
    base_config_path = list(model_configs.values())[0]
    config = utils_loading.load_config(base_config_path)
    config["general"]["task"] = task
    config["general"]["database"] = database

    df_real, _, metadata = utils_df.load_data(config)
    model_dataframes = load_model_data(model_configs, task, database)

    # Evaluate fidelity
    # evaluate_fidelity_for_models(df_real, model_dataframes, metadata)

    # Evaluate fairness
    fairness_results, fairness_dataframes = evaluate_fairness_func(
        df_real, model_dataframes, config, args.save
    )

    # Evaluate quality
    quality_results = evaluate_quality_func(
        df_real, model_dataframes, config, args.save
    )

    print("Evaluation completed successfully!")
    print(f"Evaluated models: {list(model_configs.keys())}")


if __name__ == "__main__":
    main()