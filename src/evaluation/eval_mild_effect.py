import os
import sys
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

import utils_loading, utils_df
from src.evaluation.metrics import evaluate_fidelity
from src.evaluation.metrics.evaluate_fairness import compute_fairness_metrics
from src.evaluation.metrics.evaluate_fairness import prepare_adult_dataset, prepare_compas_dataset
from src.evaluation.metrics import plots_fairness, plots_quality
from src.evaluation.metrics.evaluate_quality import evaluate_dataset_models, evaluate_ground_truth_models


_PREPARE_FUNCS = {
    "compas": prepare_compas_dataset,
    "adult": prepare_adult_dataset,
}


def get_prepare_fn(task: str):
    try:
        return _PREPARE_FUNCS[task]
    except KeyError:
        raise ValueError(f"Unsupported task: {task!r}. "
                         f"Available tasks: {list(_PREPARE_FUNCS)}")


def evaluate_fairness_func(df_real, dataframes, config, save=True):
    task = config["general"]["task"]
    experiment = config["general"]["experiment"]
    prep_fn = get_prepare_fn(task)

    fairness_results = {}
    metrics_list = []
    
    real_ds = prep_fn(df_real)
    real_metrics = compute_fairness_metrics(real_ds)
    real_metrics["name"] = "Real"
    fairness_results["dataset_0"] = real_metrics
    metrics_list.append(real_metrics)

    for idx, df in enumerate(dataframes, start=1):
        ds = prep_fn(df)
        metrics = compute_fairness_metrics(ds)
        name = config["dataframes"][idx-1]["name"]
        metrics["name"] = name
        metrics["icl_gender"] = config["dataframes"][idx-1]["icl_gender"]
        metrics["icl_records"] = config["dataframes"][idx-1]["icl_records"]
        fairness_results[f"dataset_{idx}"] = metrics
        metrics_list.append(metrics)
        print(f"Fairness Metrics for {name}:", metrics)

    df_fairness = pd.DataFrame(metrics_list)
    plot2, name2 = plots_fairness.plot_gender_trends_by_icl_gender(df_fairness, config)

    plots = [plot2]
    name_plots = [name2]

    if save:
        LOCAL_DIR = config['general']['local_dir']
        DATABASE = config["general"]["database"]
        FIGURES_PATH = config['general']['figures_path'].format(task=task, experiment=experiment)
        PROMPT = config["general"]["prompt_id"]

        base_save_path = os.path.join(LOCAL_DIR, FIGURES_PATH)
        df_path = os.path.join(base_save_path, "dataframes")
        plots_path = os.path.join(base_save_path, "plots", config["general"]["prompt_id"])

        os.makedirs(df_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)

        df_fairness.to_csv(os.path.join(df_path, f"fairness_comparison_{DATABASE}.csv"), index=False)
        for name_plot, plot in zip(name_plots, plots):
            plot_path = os.path.join(plots_path, f"{name_plot}_fairness_{DATABASE}_{PROMPT}.pdf")
            plot.savefig(plot_path, bbox_inches='tight', format='pdf')
            plt.close(plot)
            print(f"Fairness plots saved to: {plot_path}")


def evaluate_quality_func(df_real, dataframes, config, save=True):
    metrics_datasets = []
    experiment = config["general"]["experiment"]

    real_baselines = evaluate_ground_truth_models(df_real)
    for idx, df in enumerate(dataframes):
        task = config["general"]["task"]
        if task == "adult":
            metrics_datasets.append((idx, experiment, evaluate_dataset_models(df_real, df, config, config["dataframes"][idx], save)))
        else:
            metrics_datasets.append((idx, experiment, evaluate_dataset_models(df_real, df, config, config["dataframes"][idx], save)))

    plot, name_plot = plots_quality.plot_evaluation_results_mild_effect(metrics_datasets, real_baselines)

    if save:
        LOCAL_DIR = config['general']['local_dir']
        DATABASE = config["general"]["database"]
        FIGURES_PATH = config['general']['figures_path'].format(task=task, experiment=experiment)

        base_save_path = os.path.join(LOCAL_DIR, FIGURES_PATH)
        df_path = os.path.join(base_save_path, "dataframes")
        plots_path = os.path.join(base_save_path, "plots")

        os.makedirs(df_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)

        plot_path = os.path.join(plots_path, config["general"]["prompt_id"], f"{name_plot}_fairness_{DATABASE}.pdf")
        plot.savefig(plot_path, bbox_inches='tight', format='pdf')
        plt.close(plot)

        print(f"Fairness metrics saved to: {df_path}")
        print(f"Fairness plots saved to: {plots_path}")


def save_outputs(combined_metrics, corr_plot, score_plot, config):
    DATABASE = config["general"]["database"]
    LOCAL_DIR = config['paths']['local_dir']
    base_save_path = os.path.join(
        config['paths']['figures_path'].format(
            sdg_model=config["sdg"]["sdg_model"],
            prompt_id=config["general"]["prompt_id"]
        ),
        f"evaluate_fidelity/{DATABASE}"
    )

    utils_loading.save_csv(
        combined_metrics,
        LOCAL_DIR,
        os.path.join(base_save_path, "dataframes/"),
        f"df_metrics_fidelity_{DATABASE}.csv"
    )

    utils_loading.save_figure(
        corr_plot,
        LOCAL_DIR,
        os.path.join(base_save_path, "plots/"),
        f"corr_plot_fidelity_{DATABASE}.pdf"
    )

    utils_loading.save_figure(
        score_plot,
        LOCAL_DIR,
        os.path.join(base_save_path, "plots/"),
        f"score_plot_fidelity_{DATABASE}.pdf"
    )
    print(f"Outputs saved for database {DATABASE}")


def main():
    parser = argparse.ArgumentParser(description="LLM SDG PROJECT")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./src/configs/evaluation/bias_baseline/eval_mild_effect.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="Flag to save outputs (dataframes and plots)"
    )
    args = parser.parse_args()
    config = utils_loading.load_config(args.config_path)
    df_real, dataframes, metadata = utils_df.load_data(config)

    # metrics_fidelity = []
    # for conf, df in zip(config["dataframes"], dataframes):
    #     res = evaluate_fidelity.compute_fidelity_metrics(df_real, df, metadata)
    #     print("{}_{}".format(conf["type"], conf["bias"]), res)
    #     metrics_fidelity.append(res)
        
    evaluate_fairness_func(df_real, dataframes, config, args.save)


if __name__ == "__main__":
    main()