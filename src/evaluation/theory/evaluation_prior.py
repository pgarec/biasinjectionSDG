import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

import utils_loading, utils_df
from src.evaluation.metrics import evaluate_fidelity
from src.evaluation.metrics import plots_fairness, plots_quality
from src.evaluation.metrics.evaluate_quality import (
    evaluate_dataset_models_compas,
    evaluate_dataset_models_adult,
    evaluate_ground_truth_models,
)


def evaluate_prior(df_real, dataframes, config, save=True):
    icl_real_df = dataframes[0]
    no_icl_df = dataframes[1]
    clean_df = dataframes[7]
    biased_df = dataframes[12]

    icl_real_num = icl_real_df.select_dtypes(include=[np.number]).copy().dropna()
    no_icl_num   = no_icl_df.select_dtypes(include=[np.number]).copy().dropna()
    clean_num    = clean_df.select_dtypes(include=[np.number]).copy().dropna()
    biased_num   = biased_df.select_dtypes(include=[np.number]).copy().dropna()

    common_cols = (
        set(icl_real_num.columns)
        & set(no_icl_num.columns)
        & set(clean_num.columns)
        & set(biased_num.columns)
    )
    if not common_cols:
        raise ValueError("No common numeric columns found across the four datasets.")
    common_cols = sorted(common_cols)

    icl_real_num = icl_real_num[common_cols]
    no_icl_num   = no_icl_num[common_cols]
    clean_num    = clean_num[common_cols]
    biased_num   = biased_num[common_cols]

    combined_data = np.vstack([
        icl_real_num.values,
        no_icl_num.values,
        clean_num.values,
        biased_num.values
    ])

    labels = (
        ["icl_real"] * icl_real_num.shape[0]
        + ["no_icl"]   * no_icl_num.shape[0]
        + ["clean"]    * clean_num.shape[0]
        + ["biased"]   * biased_num.shape[0]
    )

    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="random",
        learning_rate="auto"
    )
    embeddings = tsne.fit_transform(combined_data)

    tsne_df = pd.DataFrame({
        "TSNE1":    embeddings[:, 0],
        "TSNE2":    embeddings[:, 1],
        "Dataset":  labels
    })

    plt.figure(figsize=(8, 6))
    for dataset_name, marker, color in [
        ("icl_real", "o", "red"),
        ("no_icl",   "s", "blue"),
        ("clean",    "^", "green"),
        ("biased",   "x", "purple")
    ]:
        subset = tsne_df[tsne_df["Dataset"] == dataset_name]
        plt.scatter(
            subset["TSNE1"],
            subset["TSNE2"],
            label=dataset_name,
            marker=marker,
            alpha=0.7,
            edgecolors="w",
            linewidths=0.5,
            s=50,
            color=color
        )

    plt.title("t-SNE Projection", fontsize=20)
    plt.xlabel("TSNE1", fontsize=18)
    plt.ylabel("TSNE2", fontsize=18)
    plt.legend(title="Dataset", fontsize=17, title_fontsize=17)
    plt.tight_layout()

    if save:
        output_path = os.path.join("./figures/theory/tsne_plot.pdf")
        plt.savefig(output_path, dpi=300)
        print(f"t-SNE plot saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="LLM SDG PROJECT")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./src/configs/evaluation/theory/prior_bias.yaml",
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
    evaluate_prior(df_real, dataframes, config, args.save)


if __name__ == "__main__":
    main()
