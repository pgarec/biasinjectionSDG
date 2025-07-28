import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm 

params = {'mathtext.default': 'regular'}
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'mathtext.default': 'regular',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
    'figure.figsize': (7, 5)  # Consistent size for single plot
})

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

np.random.seed(0)


def plot_mean_diff_by_model(df_fairness, config):
    df_mild = df_fairness[df_fairness["name"].str.startswith("Mild")].copy()
    df_mild["bias_pct"] = (
        df_mild["name"]
            .str.extract(r"(\d+(?:\.\d+)?)%")
            .iloc[:, 0]
            .astype(float)
            .fillna(0)
    )
    df_mild = df_mild.sort_values("bias_pct")
    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(df_mild["model"].unique())
    cmap = plt.cm.get_cmap("tab10", len(models))
    for idx, model in enumerate(models):
        df_mod = df_mild[df_mild["model"] == model]
        ax.plot(
            df_mod["bias_pct"],
            df_mod["mean_difference"],
            marker="o",
            lw=2,
            label= model.split("/")[-1], 
            color=cmap(idx),
        )
    
    ax.set_xlabel("Mild Bias Percentage (%)", fontsize=18)
    ax.set_ylabel("Mean Difference", fontsize=18)
    ax.set_title(f"Mean Difference vs. Bias – {config['general']['task']} Dataset", fontsize=20)
    ax.set_ylim(-1, 1)
    ax.set_xticks(df_mild["bias_pct"].unique())
    ax.set_xticklabels([f"{int(p)}" for p in df_mild["bias_pct"].unique()], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(title="Model", fontsize=14, title_fontsize=16, ncol=1)
    
    plt.tight_layout()
    return fig, "mean_diff_by_model"


def compute_mean_differences_line_single(df_fairness, config):
    # Filter out unwanted rows
    df_fairness = df_fairness[df_fairness["icl_gender"] != "only_male_icl"]
    df_mild = df_fairness[df_fairness["name"].str.startswith("Mild")].copy()

    # Extract bias percentage
    df_mild["bias_pct"] = (
        df_mild["name"]
        .str.extract(r"(\d+(?:\.\d+)?)%")
        .iloc[:, 0]
        .astype(float)
        .fillna(0)
    )
    df_mild = df_mild.sort_values("bias_pct")
    x_vals = df_mild["bias_pct"].values

    # Only one axis now
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Plot mean difference line + colored dots
    y_md = df_mild["mean_difference"].values
    cmap_vals = np.linspace(0.3, 1.0, len(x_vals))
    point_colors = [mcolors.to_hex(plt.cm.Reds(v)) for v in cmap_vals]

    # ax.plot(x_vals, y_md, color="black", lw=2.5, marker="x", label="Mean Difference")
    ax.plot(
        x_vals,
        y_md,
        color="#003f5c",  # Deep navy blue, high contrast and formal
        lw=2.5,
        marker="s",       # Square markers: neutral and precise
        markersize=5,
        markerfacecolor="#2f4b7c",  # Slightly lighter navy for visual clarity
        markeredgewidth=0.8,
        alpha=0.95,
        label="Mean Difference"
    )

    # Add horizontal baselines from df_fairness
    for name, col, lbl in [
        ("no icl", "#1f77b4", "No ICL"),
        ("clean", "#2ca02c", "Clean"),
        ("real", "#9467bd", "Real"),
        ("icl real", "#d62728", "ICL Real"),
    ]:
        ref = df_fairness.loc[df_fairness["name"].str.lower() == name, "mean_difference"]
        if not ref.empty:
            ax.axhline(ref.iloc[0], color=col, ls="--", lw=2, label=lbl)

    # Styling
    ax.set_xlabel("In-context Bias (%)", fontsize=14)
    ax.set_ylabel("Mean Difference", fontsize=14)
    ax.set_title(f"Fairness Analysis — {config['general']['task']} Dataset", fontsize=16)
    ax.set_ylim(-1, 1)
    xticks = x_vals[::2] if len(x_vals) > 6 else x_vals
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=12, ncol=2, loc="lower right", frameon=True)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    fig.tight_layout(pad=2.0)
    return fig, "mean_difference_plot_single"


def compute_mean_differences_line(df_fairness, config):
    df_fairness = df_fairness[df_fairness["icl_gender"] != "only_male_icl"]
    df_mild = df_fairness[df_fairness["name"].str.startswith("Mild")].copy()
    df_mild["bias_pct"] = (
        df_mild["name"]
        .str.extract(r"(\d+(?:\.\d+)?)%")
        .iloc[:, 0]
        .astype(float)
        .fillna(0)
    )
    df_mild = df_mild.sort_values("bias_pct")
    x_vals = df_mild["bias_pct"].values
    xticks = x_vals[::2] if len(x_vals) > 6 else x_vals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Style constants
    line_color = "#003f5c"
    marker_face = "#2f4b7c"
    male_color = "#1f77b4"
    female_color = "#d62728"

    # --- Subplot 1: Mean difference line ---
    y_md = df_mild["mean_difference"].values
    ax1.plot(
        x_vals, y_md,
        color=line_color, lw=2.5, marker="s", markersize=5,
        markerfacecolor=marker_face, markeredgewidth=0.8,
        alpha=0.95, label="Mean Difference"
    )

    for name, col, lbl in [
        ("no icl", "#1f77b4", "No ICL"),
        ("clean", "#2ca02c", "Clean"),
        ("real", "#9467bd", "Real"),
        ("icl real", "#d62728", "ICL Real"),
    ]:
        ref = df_fairness.loc[df_fairness["name"].str.lower() == name, "mean_difference"]
        if not ref.empty:
            ax1.axhline(ref.iloc[0], color=col, ls="--", lw=2, label=lbl)

    ax1.set_xlabel("Mild Bias Percentage (%)", fontsize=18)
    ax1.set_ylabel("Mean Difference", fontsize=18)
    ax1.set_title(f"Fairness Analysis — {config['general']['task']} Dataset", fontsize=18)
    ax1.set_ylim(-1, 1)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=14)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.legend(fontsize=14, ncol=2, loc="lower right", frameon=False)
    ax1.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # --- Subplot 2: Group-wise means ---
    if {"mean_label_privileged", "mean_label_unprivileged"}.issubset(df_fairness.columns):
        y_priv = df_mild["mean_label_privileged"].values
        y_unpriv = df_mild["mean_label_unprivileged"].values

        ax2.plot(
            x_vals, y_priv,
            marker="s", linestyle="-", linewidth=2, color=male_color,
            label="Privileged"
        )
        ax2.plot(
            x_vals, y_unpriv,
            marker="^", linestyle="--", linewidth=2, color=female_color,
            label="Unprivileged"
        )

        df_no_icl = df_fairness[df_fairness["name"].str.lower() == "icl real"]
        if not df_no_icl.empty:
            priv0 = df_no_icl["mean_label_privileged"].iloc[0]
            unpriv0 = df_no_icl["mean_label_unprivileged"].iloc[0]
            ax2.axhline(priv0, color=male_color, ls="--", lw=2, alpha=0.8, label="No ICL Privileged")
            ax2.axhline(unpriv0, color=female_color, ls="--", lw=2, alpha=0.8, label="No ICL Unprivileged")

    ax2.set_xlabel("In-context Bias (%)", fontsize=18)
    ax2.set_ylabel("Pr (AA)", fontsize=18)
    ax2.set_title("Group-wise Target Probability vs. Bias %", fontsize=18)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=14)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.legend(fontsize=14, loc="lower right", ncol=2, frameon=False)
    ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    fig.tight_layout(pad=2.0)
    return fig, "mean_diff_and_label_trends"


def compute_group_counts(df_fairness, config):
    """
    Creates a bar plot of Privileged vs Unprivileged group counts for each experiment.
    """
    labels = df_fairness["name"]
    idx = np.arange(len(labels))
    width = 0.35

    priv_col = "n_privileged"   if "n_privileged"   in df_fairness.columns else "privileged"
    unpriv_col = "n_unprivileged" if "n_unprivileged" in df_fairness.columns else "unprivileged"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(idx - width/2, df_fairness[priv_col],   width, label="Male (Privileged)",   color="#d62728")
    ax.bar(idx + width/2, df_fairness[unpriv_col], width, label="Female (Unprivileged)", color="#1f77b4")

    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_ylabel("Count", fontsize=18)
    ax.set_title(f"Group Counts per Dataset - {config['general']['task']}", fontsize=20)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    plt.tight_layout()
    return fig, "group_counts"


def compute_mean_differences_icldemonstration_line(df_fairness, config):
    df_mild = df_fairness[df_fairness["name"].str.startswith("Mild")].copy()
    df_mild["bias_pct"] = (
        df_mild["name"]
        .str.extract(r"(\d+(?:\.\d+)?)%")
        .iloc[:, 0]
        .astype(float)
        .fillna(0)
    )
    df_mild = df_mild.sort_values("bias_pct")
    pivot = df_mild.pivot(index="bias_pct", columns="icl_records", values="mean_difference")
    x_vals = pivot.index.values

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {10: ("o-", 2), 20: ("s-", 2), 40: ("^-", 2), 80: ("x-", 2), 100: ("*-", 2)}
    for rec in [10, 20, 40, 80, 100]:
        if rec in pivot.columns:
            line_style, lw = styles[rec]
            ax.plot(
                x_vals,
                pivot[rec].values,
                line_style,
                lw=lw,
                label=f"ICL {rec}"
            )

    ax.set_xlabel("Mild Bias Percentage (%)", fontsize=16)
    ax.set_ylabel("Mean Difference", fontsize=16)
    ax.set_title(
        f"Fairness Comparison – {config['general']['task']} Dataset",
        fontsize=18
    )
    ax.set_ylim(-1, 1)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{int(p)}" for p in x_vals], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(title="ICL Examples", fontsize=12, title_fontsize=12)

    plt.tight_layout()
    return fig, "mean_differences_icldemonstration_line"


def plot_gender_trends_by_icl(df_fairness, config):
    # Filter and prepare data
    df_mild = df_fairness[df_fairness["name"].str.startswith("Mild")].copy()
    df_mild["bias_pct"] = (
        df_mild["name"]
        .str.extract(r"(\d+(?:\.\d+)?)%")
        .iloc[:, 0]
        .astype(float)
        .fillna(0)
    )
    df_mild = df_mild.sort_values("bias_pct")

    icl_settings = [10, 20, 40, 80, 100]
    male_color = "#1f77b4"
    female_color = "#d62728"

    fig, axes = plt.subplots(1, len(icl_settings), figsize=(5 * len(icl_settings), 5), sharey=True)

    for idx, (ax, rec) in enumerate(zip(axes, icl_settings)):
        df_rec = df_mild[df_mild["icl_records"] == rec]
        x = df_rec["bias_pct"].values
        y_male = df_rec["mean_label_privileged"].values
        y_female = df_rec["mean_label_unprivileged"].values
        y_diff = df_rec['mean_difference'].values

        ax.plot(x, y_male, marker="s", linestyle="-", linewidth=2, color=male_color, label="Male")
        ax.plot(x, y_female, marker="^", linestyle="--", linewidth=2, color=female_color, label="Female")

        ax.fill_between(x, y_male, y_female, where=y_male >= y_female, facecolor=male_color, alpha=0.2, interpolate=True)
        ax.fill_between(x, y_male, y_female, where=y_male < y_female, facecolor=female_color, alpha=0.2, interpolate=True)
        # ax2 = ax.twinx()
        # ax2.plot(x, y_diff, linestyle="--", linewidth=2, color="gray", label="Mean Diff")
        # ax2.set_ylim(0, 1)

        # if idx == len(icl_settings) - 1:
        #     ax2.set_ylabel("Mean Difference", fontsize=18)

        lines, labels = ax.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines, labels, fontsize=20, loc="upper left", frameon=False)

        ax.set_title(f"{rec} ICL samples", fontsize=24, pad=15)
        ax.set_xlabel("In-context Bias (%)", fontsize=24)
        xticks = x[::2] if len(x) > 6 else x
        ax.set_xticks(xticks)
        ax.set_xlim(0, 100)
        ax.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Pr(AA)", fontsize=22)

    # fig.suptitle(
    #     f"Group-wise Target Probability vs. Bias % — {config['general']['task']} Dataset",
    #     fontsize=24, y=1.05
    # )
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    return fig, "gender_trends_by_icl_records"


def plot_gender_trends_by_model(df_fairness, config):
    df_mild = df_fairness[df_fairness["name"].str.startswith("Mild")].copy()
    if "bias_pct" not in df_mild.columns:
        df_mild["bias_pct"] = (
            df_mild["name"]
              .str.extract(r"(\d+(?:\.\d+)?)%")
              .iloc[:, 0]
              .astype(float)
              .fillna(0)
        )
    df_mild = df_mild.sort_values("bias_pct")

    models = ['openai/ibm-granite/granite-3.2-8b-instruct', 'openai/mistralai/mixtral-8x7B-instruct-v0.1',
              'openai/mistralai/mixtral-8x22B-instruct-v0.1','openai/meta-llama/llama-3-3-70b-instruct']
    model_names = {'openai/ibm-granite/granite-3.2-8b-instruct': 'granite-3.2-8b', 'openai/mistralai/mixtral-8x7B-instruct-v0.1': 'mixtral-8x7b',
              'openai/mistralai/mixtral-8x22B-instruct-v0.1': 'mixtral-8x22b','openai/meta-llama/llama-3-3-70b-instruct': 'llama-3-3-70b'}
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_models, figsize=(6 * n_models, 4.5), sharey=True,
        constrained_layout=False
    )
    if n_models == 1:
        axes = [axes]

    male_color = "#1f77b4"    # professional blue
    female_color = "#d62728"  # professional red

    for ax, model in zip(axes, models):
        df_mod = df_mild[df_mild["model"] == model]
        x = df_mod["bias_pct"].values
        y_male = df_mod["mean_label_privileged"].values
        y_fem = df_mod["mean_label_unprivileged"].values
        y_diff = df_mod['mean_difference'].values

        ax.plot(
            x, y_male,   marker="s", ls="-",  lw=2, color=male_color,
            label="Male"
        )
        ax.plot(
            x, y_fem,    marker="^", ls="--", lw=2, color=female_color,
            label="Female"
        )

        ax.fill_between(x, y_male, y_fem, where=y_male >= y_fem, facecolor=male_color, alpha=0.2, interpolate=True)
        ax.fill_between(x, y_male, y_fem, where=y_male < y_fem, facecolor=female_color, alpha=0.2, interpolate=True)

        ax.set_title(model_names[model], fontsize=24, pad=15)
        ax.set_xlabel("In-context Bias (%)", fontsize=24)
        xticks = x[::2] if len(x) > 6 else x
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=18)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(True, ls="--", alpha=0.6)
        # ax.plot(x, y_diff, linestyle='--', linewidth=3.0, color="gray", label='Mean Difference')
        ax.legend(fontsize=18, frameon=False, loc="upper left")

    axes[0].set_ylabel("Pr(AA)", fontsize=18, labelpad=10)
    # fig.suptitle(
    #     f"Gender-wise Label Trends vs. Bias % — {config['general']['task']} Dataset",
    #     fontsize=24, y=1.05
    # )

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.88)

    return fig, "gender_trends_by_model"


def plot_gender_trends_by_icl_gender(df_fairness, config):
    df = df_fairness.copy()
    if "bias_pct" not in df.columns:
        df["bias_pct"] = (
            df["name"]
              .str.extract(r"(\d+(?:\.\d+)?)%")
              .iloc[:, 0]
              .astype(float)
              .fillna(0)
        )

    icl_genders = ["only_female_icl", "only_male_icl"]
    icl_genders_names = {"only_female_icl": "Only ICL Female", "only_male_icl": "Only ICL Male"}
    n_genders = len(icl_genders)

    # Use colorblind-friendly palette
    colors = plt.get_cmap("tab10")
    male_color = colors(0)   # Blue
    female_color = colors(1) # Orange

    fig, axes = plt.subplots(1, n_genders, figsize=(6 * n_genders, 4.5), sharey=True)

    if n_genders == 1:
        axes = [axes]

    for ax, ig in zip(axes, icl_genders):
        df_sub = df[df['icl_gender'] == ig].sort_values('bias_pct')
        x = df_sub['bias_pct'].values
        y_male = df_sub['mean_label_privileged'].values
        y_female = df_sub['mean_label_unprivileged'].values

        ax.plot(x, y_male,
                marker='s', linestyle='-', linewidth=2.5, color=male_color,
                label='Male')
        ax.plot(x, y_female,
                marker='^', linestyle='--', linewidth=2.5, color=female_color,
                label='Female')
        
        ax.fill_between(x, y_male, y_female, where=y_male >= y_female, facecolor=male_color, alpha=0.2, interpolate=True)
        ax.fill_between(x, y_male, y_female, where=y_male < y_female, facecolor=female_color, alpha=0.2, interpolate=True)

        ax.set_title(icl_genders_names[ig], fontsize=18, pad=15)
        ax.set_xlabel("In-context Bias (%)", fontsize=20, labelpad=8)

        # Reduce x-tick clutter
        xticks = x[::2] if len(x) > 6 else x
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=14)
        ax.set_xlim(0, 100)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
        ax.legend(fontsize=18, loc="upper left", frameon=False)


    axes[0].set_ylabel("Pr(AA)", fontsize=20, labelpad=10)

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.88)

    return fig, "gender_trends_by_icl_gender"


def plot_gender_trends_comparison_attack(
    df_fairness,
    df_fairness_mixtral7b,
    df_fairness_mixtral22b,
    df_fairness_llama,
    config
):
    def _ensure_bias(df):
        df = df.copy()
        if "bias_pct" not in df.columns:
            df["bias_pct"] = (
                df["name"]
                  .str.extract(r"(\d+(?:\.\d+)?)%")
                  .iloc[:, 0]
                  .astype(float)
                  .fillna(0)
            )
        return df.sort_values("bias_pct")

    dfs = [
        (_ensure_bias(df_fairness), "Granite-8B"),
        (_ensure_bias(df_fairness_mixtral7b), "Mixtral-7B"),
        (_ensure_bias(df_fairness_mixtral22b), "Mixtral-22B"),
        (_ensure_bias(df_fairness_llama),  "Llama-70B")
    ]

    cmap = plt.get_cmap("tab10")
    male_color = cmap(0)
    female_color = cmap(1)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)

    for ax, (df, model_name) in zip(axes, dfs):
        df_main = df[df["name"].str.startswith("Mild Bias")].sort_values("bias_pct")
        df_base = df[df["name"].str.lower().isin(["no icl"]) ]

        x_main = df_main["bias_pct"].values
        y_male = df_main["mean_label_privileged"].values
        y_female = df_main["mean_label_unprivileged"].values

        ax.plot(x_main, y_male,   marker='s', linestyle='-',  linewidth=2.5,
                color=male_color,   label='Privileged')
        ax.plot(x_main, y_female, marker='^', linestyle='--', linewidth=2.5,
                color=female_color, label='Unprivileged')

        ax.fill_between(x_main, y_male, y_female,
                        where=y_male >= y_female, facecolor=male_color,
                        alpha=0.2, interpolate=True)
        ax.fill_between(x_main, y_male, y_female,
                        where=y_male <  y_female, facecolor=female_color,
                        alpha=0.2, interpolate=True)

        ax.set_title(model_name, fontsize=24, pad=15)
        ax.set_xlabel("In-context Bias (%)", fontsize=24, labelpad=10)

        if not df_base.empty:
            real_p = df_base["mean_label_privileged"].iloc[0]
            real_u = df_base["mean_label_unprivileged"].iloc[0]
            ax.axhline(real_p, color=male_color,   linestyle="--", lw=2,
                       label="Real Privileged")
            ax.axhline(real_u, color=female_color, linestyle="--", lw=2,
                       label="Real Unprivileged")

        xticks = x_main[::2] if len(x_main) > 6 else x_main
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=18)
        ax.set_xlim(0, 100)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    axes[0].set_ylabel("Pr(Y=1)", fontsize=24, labelpad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),  # x-center, slightly above
        ncol=len(labels),
        frameon=False,
        fontsize=22
    )

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.85)

    return fig, "gender_trends_comparison_attack"
