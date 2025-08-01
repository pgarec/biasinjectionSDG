import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def extract_bias_percent(name):
    match = re.search(r'Mild Bias\s*(\d+)%', name.strip())
    return int(match.group(1)) if match else None


def plot_evaluation_results_mild_effect(metrics_datasets, real_baselines):
    rows = []

    for idx, experiment, model_results in metrics_datasets:
        for metric in model_results:
            name = metric.get("name", "")
            model_name = metric.get("model", "Unknown")

            if "Mild Bias" not in name or "(synthetic)" not in model_name:
                continue  # Only keep synthetic-trained models on mild bias experiments

            row = {
                "idx": idx,
                "experiment": experiment,
                "name": name,
                "model": model_name,
                "bias_percent": extract_bias_percent(name)
            }
            row.update(metric)
            rows.append(row)

    df_metrics = pd.DataFrame(rows)
    df_metrics = df_metrics.sort_values("bias_percent")

    if df_metrics.empty:
        raise ValueError("No synthetic mild bias experiments found for plotting.")

    metrics_to_plot = [
        "acc", "f1", "precision", "recall"
    ]
    available_metrics = [m for m in metrics_to_plot if m in df_metrics.columns]

    df_melted = df_metrics.melt(
        id_vars=["bias_percent", "experiment", "model"],
        value_vars=available_metrics,
        var_name="metric", value_name="score"
    )

    df_melted["group"] = df_melted["metric"].apply(
        lambda m: "Accuracy & F1" if "acc" in m or "f1" in m else "Precision & Recall"
    )

    models = df_melted["model"].unique()
    n_models = len(models)
    n_rows, n_cols = 2, n_models  # 2 rows (metric groups), 4 columns (one per model)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), sharex=True, sharey=False)

    for col_idx, model in enumerate(models):
        for row_idx, group in enumerate(["Accuracy & F1", "Precision & Recall"]):
            ax = axes[row_idx, col_idx]

            df_model = df_melted[
                (df_melted["model"] == model) &
                (df_melted["group"] == group)
            ]

            for metric in df_model["metric"].unique():
                subset = df_model[df_model["metric"] == metric]
                ax.plot(subset["bias_percent"], subset["score"], marker="o", label=metric)

            # Baseline from real-trained model
            base_model_name = model.replace(" (synthetic)", " (real)")
            baseline = next((b for b in real_baselines if b["model"] == base_model_name), None)
            if baseline:
                if group == "Accuracy & F1":
                    for color, key in [("blue", "acc_real"), ("green", "f1_real")]:
                        if key in baseline:
                            ax.axhline(baseline[key], linestyle="--", color=color, label=f"{key} (real)")
                elif group == "Precision & Recall":
                    for color, key in [("blue", "precision_real"), ("green", "recall_real")]:
                        if key in baseline:
                            ax.axhline(baseline[key], linestyle="--", color=color, label=f"{key} (real)")

            ax.set_title(f"{model} - {group}", fontsize=13)
            ax.set_xlabel("Mild Bias Percentage (%)", fontsize=11)
            ax.set_ylim(0, 1.0)
            ax.grid(True)
            ax.legend(fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Score", fontsize=11)

    plt.suptitle("Mild Bias Effect", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, "quality_metrics_mild_effect"


def plot_evaluation_results_mild_effect_comparison(
    metrics_datasets_granite,
    metrics_datasets_mixtral7b,
    metrics_datasets_mixtral22b,
    metrics_datasets_llama,
    real_baselines,
    config=None
):
    def extract_bias_percent(name):
        return float(pd.Series(name).str.extract(r"(\d+(?:\.\d+)?)%")[0])

    def build_df(metrics_datasets, name_model):
        rows = []
        for idx, experiment, model_results in metrics_datasets:
            for m in model_results:
                model = m.get("model", "")
                if name_model.lower() not in model.lower():
                    continue
                dataset_name = m.get("dataset", "")
                # only mild-bias synthetic experiments
                if "Mild Bias" not in dataset_name or "(synthetic)" not in model:
                    continue
                bp = extract_bias_percent(dataset_name)
                rows.append({
                    "bias_percent": bp,
                    "model": model,
                    "avg_proba_y1_privileged": m.get("avg_proba_y1_privileged_mean", np.nan),
                    "avg_proba_y1_non_privileged": m.get("avg_proba_y1_non_privileged_mean", np.nan)
                })
        return pd.DataFrame(rows).sort_values("bias_percent")

    df_granite = build_df(metrics_datasets_granite, "random forest")
    df_mix7b = build_df(metrics_datasets_mixtral7b, "random forest")
    df_mix22b = build_df(metrics_datasets_mixtral22b, "random forest")
    df_llama = build_df(metrics_datasets_llama,"random forest")

    baseline_map = {b["model"]: b for b in real_baselines}
    cmap = plt.get_cmap("tab10")
    black_color = cmap(0)
    non_black_color = cmap(1)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
    dfs = [df_granite, df_mix7b, df_mix22b, df_llama]
    labels = ["Granite 8B", "Mixtral-7B", "Mixtral-22B", "Llama 70B"]

    for idx, (ax, df) in enumerate(zip(axes, dfs)):
        x = df["bias_percent"].values
        y_b = df["avg_proba_y1_privileged"].values
        y_nb= df["avg_proba_y1_non_privileged"].values

        ax.plot(x, y_b,  marker="s", linestyle="-",  linewidth=2.5,
                color=black_color,     label="Privileged")
        ax.plot(x, y_nb, marker="^", linestyle="--", linewidth=2.5,
                color=non_black_color, label="Unprivileged")

        ax.fill_between(x, y_b, y_nb, where=y_b>=y_nb,
                        facecolor=black_color,   alpha=0.2, interpolate=True)
        ax.fill_between(x, y_b, y_nb, where=y_b< y_nb,
                        facecolor=non_black_color, alpha=0.2, interpolate=True)

        if not df.empty:
            synth_model = df["model"].iloc[0]
            real_model = synth_model.replace("(synthetic)", "(real)")
            baseline = baseline_map.get(real_model, {})
            rb = baseline.get("avg_proba_y1_privileged", None)
            rnb = baseline.get("avg_proba_y1_non_privileged", None)
            if rb is not None:
                ax.axhline(rb, linestyle=":",  color=black_color,
                           linewidth=2, label="Real Privileged")
            if rnb is not None:
                ax.axhline(rnb, linestyle="--", color=non_black_color,
                           linewidth=2, label="Real Unprivileged")

        title = synth_model.replace(" (synthetic)", f" – {labels[idx]}")
        title = title.replace("Random Forest", "RF")
        ax.set_title(title, fontsize=24, pad=15)
        ax.set_xlabel("In-context Bias (%)", fontsize=24, labelpad=6)
        ax.set_xlim(0, 100)
        xticks = x[::2] if len(x) > 6 else x
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[0].set_ylabel("Pr(Y=1)", fontsize=24, labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.08),
               ncol=len(labels),
               frameon=False, fontsize=22)

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.85)

    return fig, "quality_metrics_mild_effect_comparison"


def plot_evaluation_results_mild_effect_comparison_flexible(
    all_metrics_datasets,
    real_baselines,
    config=None
):
    def extract_bias_percent(name):
        return float(pd.Series(name).str.extract(r"(\d+(?:\.\d+)?)%")[0])

    def build_df(metrics_datasets, model_name):
        rows = []
        for idx, experiment, model_results in metrics_datasets:
            for m in model_results:
                model = m.get("model", "")
                dataset_name = m.get("dataset", "")
                # only mild-bias synthetic experiments
                if "Mild Bias" not in dataset_name or "(synthetic)" not in model:
                    continue
                bp = extract_bias_percent(dataset_name)
                rows.append({
                    "bias_percent": bp,
                    "model": model,
                    "avg_proba_y1_privileged": m.get("avg_proba_y1_privileged_mean", np.nan),
                    "avg_proba_y1_non_privileged": m.get("avg_proba_y1_non_privileged_mean", np.nan)
                })
        return pd.DataFrame(rows).sort_values("bias_percent")

    # Build dataframes for all models
    model_dfs = {}
    model_labels = {}
    
    # Create clean labels for display
    label_mapping = {
        "granite": "Granite",
        "mixtral-7b": "Mixtral-7B", 
        "mixtral-22b": "Mixtral-22B",
        "llama": "Llama",
        "gpt": "GPT"
    }
    
    for model_name, metrics_datasets in all_metrics_datasets.items():
        df = build_df(metrics_datasets, "random forest")  # or make this configurable
        if not df.empty:
            model_dfs[model_name] = df
            
            # Create display label
            clean_label = model_name
            for key, label in label_mapping.items():
                if key.lower() in model_name.lower():
                    clean_label = label
                    break
            model_labels[model_name] = clean_label

    n_models = len(model_dfs)
    if n_models == 0:
        raise ValueError("No valid model data found for plotting")

    # Map baselines by model name
    baseline_map = {b["model"]: b for b in real_baselines}
    
    # Colors
    cmap = plt.get_cmap("tab10")
    privileged_color = cmap(0)
    unprivileged_color = cmap(1)

    # Dynamic figure sizing
    fig_width = min(6 * n_models, 30)  # Cap at 30 inches
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, 5), sharey=True)
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, df) in enumerate(model_dfs.items()):
        ax = axes[idx]
        
        if df.empty:
            ax.text(0.5, 0.5, f"No data for\n{model_labels[model_name]}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_labels[model_name], fontsize=24, pad=15)
            continue

        x = df["bias_percent"].values
        y_priv = df["avg_proba_y1_privileged"].values
        y_unpriv = df["avg_proba_y1_non_privileged"].values

        ax.plot(x, y_priv,  marker="s", linestyle="-",  linewidth=2.5,
                color=privileged_color,     label="Privileged")
        ax.plot(x, y_unpriv, marker="^", linestyle="--", linewidth=2.5,
                color=unprivileged_color, label="Unprivileged")

        ax.fill_between(x, y_priv, y_unpriv, where=y_priv >= y_unpriv,
                        facecolor=privileged_color,   alpha=0.2, interpolate=True)
        ax.fill_between(x, y_priv, y_unpriv, where=y_priv < y_unpriv,
                        facecolor=unprivileged_color, alpha=0.2, interpolate=True)

        # Add baseline if available
        if not df.empty:
            synth_model = df["model"].iloc[0]
            real_model = synth_model.replace("(synthetic)", "(real)")
            baseline = baseline_map.get(real_model, {})
            
            rb = baseline.get("avg_proba_y1_privileged", None)
            rnb = baseline.get("avg_proba_y1_non_privileged", None)
            
            if rb is not None:
                ax.axhline(rb, linestyle=":",  color=privileged_color,
                           linewidth=2, label="Real Privileged")
            if rnb is not None:
                ax.axhline(rnb, linestyle="--", color=unprivileged_color,
                           linewidth=2, label="Real Unprivileged")

        ax.set_title(f"{model_labels[model_name]}", fontsize=24, pad=15)
        ax.set_xlabel("In-context Bias (%)", fontsize=24, labelpad=6)
        ax.set_xlim(0, 100)
        
        xticks = x[::2] if len(x) > 6 else x
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(p)}" for p in xticks], fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[0].set_ylabel("Pr(Y=1)", fontsize=24, labelpad=8)
    
    # Get legend from first axis that has data
    handles, labels = None, None
    for ax in axes:
        try:
            h, l = ax.get_legend_handles_labels()
            if h:  # If there are handles
                handles, labels = h, l
                break
        except:
            continue

    if handles and labels:
        fig.legend(handles, labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.08),
                   ncol=len(labels),
                   frameon=False, fontsize=22)

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.85)

    return fig, "quality_metrics_mild_effect_comparison_flexible"


def plot_evaluation_results_icl_demonstration(metrics_datasets, real_baselines):
    rows = []

    for idx, experiment, dataset_config, model_results in metrics_datasets:
        if dataset_config.get("prompt_id") != "icl_demonstration":
            continue

        icl_records = dataset_config.get("icl_records", 0)

        for metric in model_results:
            model_name = metric.get("model", "Unknown")
            row = {
                "idx": idx,
                "experiment": experiment,
                "name": metric.get("name", ""),
                "model": model_name,
                "icl_records": icl_records
            }
            row.update(metric)
            rows.append(row)

    df_metrics = pd.DataFrame(rows)
    df_metrics = df_metrics.sort_values("icl_records")

    if df_metrics.empty:
        raise ValueError("No icl_demonstration experiments found for plotting.")

    metrics_to_plot = ["acc", "f1", "precision", "recall"]
    available_metrics = [m for m in metrics_to_plot if m in df_metrics.columns]

    df_melted = df_metrics.melt(
        id_vars=["icl_records", "experiment", "model"],
        value_vars=available_metrics,
        var_name="metric", value_name="score"
    )

    df_melted["group"] = df_melted["metric"].apply(
        lambda m: "Accuracy & F1" if m in ["acc", "f1"] else "Precision & Recall"
    )

    models = df_melted["model"].unique()
    n_models = len(models)
    n_rows, n_cols = 2, max(n_models, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows),
                             sharex=True, sharey=False)

    if n_models == 1:
        axes = np.array(axes).reshape(n_rows, n_cols)

    for col_idx, model in enumerate(models):
        for row_idx, group in enumerate(["Accuracy & F1", "Precision & Recall"]):
            ax = axes[row_idx, col_idx]

            df_model = df_melted[
                (df_melted["model"] == model) &
                (df_melted["group"] == group)
            ]

            for metric in df_model["metric"].unique():
                subset = df_model[df_model["metric"] == metric]
                ax.plot(subset["icl_records"], subset["score"], marker="o", label=metric)

            base_model_name = model.replace(" (synthetic)", " (real)")
            baseline = next((b for b in real_baselines if b["model"] == base_model_name), None)

            if baseline:
                if group == "Accuracy & F1":
                    for color, key in [("blue", "acc_real"), ("green", "f1_real")]:
                        if key in baseline:
                            ax.axhline(baseline[key], linestyle="--", color=color,
                                       label=f"{key} (real)")
                elif group == "Precision & Recall":
                    for color, key in [("blue", "precision_real"), ("green", "recall_real")]:
                        if key in baseline:
                            ax.axhline(baseline[key], linestyle="--", color=color,
                                       label=f"{key} (real)")

            ax.set_title(f"{model} - {group}", fontsize=13)
            ax.set_xlabel("Number of ICL Records", fontsize=11)
            ax.set_ylim(0, 1.0)
            ax.grid(True)
            ax.legend(fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Score", fontsize=11)

    plt.suptitle("ICL Demonstration Quality (Real Dataset Metrics)", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, "quality_metrics_icl_demonstration"
