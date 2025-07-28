import pandas as pd
from itertools import combinations
import sys
import numpy as np
import matplotlib.pyplot as plt

from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.single_table import LogisticDetection

sys.path.append("./src/evaluation/metrics")
sys.path.append("./src/utils")
sys.path.append("./src/parsers")

import utils_loading, utils_df, utils_sdv
from metrics_fidelity import compute_WD, compute_JSD
from metrics_fidelity import compute_TVComplement, compute_KSComplement
from metrics_fidelity import compute_CorrelationSimilarity
from metrics_fidelity import compute_ContingencySimilarity
from plots_fidelity import get_correlation_plot, get_score_plot


def compute_fidelity_metrics(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    metadata,
    list_metrics: list = [
        # "WD",
        "JSD",
        # "KSComplement",
        "TVComplement",
    ],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    df_real_shuffled = df_real.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_synth_shuffled = df_synth.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if 'id' in df_real_shuffled.columns:
        df_real_shuffled.drop(columns=['id'], inplace=True)

    real_parts  = np.array_split(df_real_shuffled,  n_splits)
    synth_parts = np.array_split(df_synth_shuffled, n_splits)

    results = {}
    for metric in list_metrics:
        vals = []
        for real_part, synth_part in zip(real_parts, synth_parts):
            if metric == "WD":
                num_cols = real_part.select_dtypes(include=[np.number]).columns
                wd_val = compute_WD(real_part[num_cols], synth_part[num_cols])
                vals.append(wd_val)

            elif metric == "JSD":
                # cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]
                # real_num  = real_part[cols].apply(pd.to_numeric, errors="coerce")
                # synth_num = synth_part[cols].apply(pd.to_numeric, errors="coerce")
                # real_clean  = real_num.dropna()
                # synth_clean = synth_num.dropna()              
                num_cols = real_part.select_dtypes(include=[np.number]).columns
                jsd_val = compute_JSD(real_part[num_cols], synth_part[num_cols])
                vals.append(jsd_val)

            elif metric == "KSComplement":
                ks_val = compute_KSComplement(real_part, synth_part, metadata)
                vals.append(ks_val)

            elif metric == "TVComplement":
                tv_val = compute_TVComplement(real_part, synth_part, metadata)
                vals.append(tv_val)

            elif metric == "CorrelationSimilarity":
                corr_val = compute_CorrelationSimilarity(real_part, synth_part, metadata)
                vals.append(corr_val)

            elif metric == "ContingencySimilarity":
                cont_val = compute_ContingencySimilarity(real_part, synth_part, metadata)
                vals.append(cont_val)

            elif metric == "LogisticDetection":
                log_val      = LogisticDetection.compute(
                    real_data=real_part,
                    synthetic_data=synth_part,
                    metadata=metadata.to_dict()
                )
                vals.append(log_val)

            else:
                raise ValueError(f"Unsupported metric: {metric}")

        mean_val = float(np.mean(vals))
        std_val  = float(np.std(vals, ddof=1))  # sample std
        results[metric] = {"mean": mean_val, "std": std_val}

    return results


def evaluate_correlations(df_real: pd.DataFrame, df_synth: pd.DataFrame, list_continuous_columns: list) -> pd.DataFrame:
    df_corr = []
    col_combinations = combinations(list_continuous_columns, 2)

    for col1, col2 in col_combinations:
        score = CorrelationSimilarity.compute(df_real[[col1, col2]], df_synth[[col1, col2]])
        df_corr.append({'Column Pair': f"{col1} - {col2}", 'Correlation Score': round(score, 2)})

    # Sort correlations by absolute value for better visualization
    df_corr_df = pd.DataFrame(df_corr).sort_values(by="Correlation Score", ascending=False)
    return df_corr_df


def compute_fidelity_plots(df1, df2, metadata, config):
    metadata = utils_loading.read_dict(config["general"]["local_dir"] + config["general"]["metadata_path"].format(config["experiments"][0]["database"]))
    sdv_report = utils_sdv.get_sdv_report(df1, df2, metadata)
    df_base_metrics = sdv_report.get_properties().rename(
        columns={"Property": "Metric", "Score": "Value"}
    )

    corr_plot = get_correlation_plot(sdv_report)
    score_plot = get_score_plot(sdv_report)
    dict_metrics = compute_fidelity_metrics(
        df_real=df1,
        df_synth=df2,
        metadata=metadata,
        list_metrics=config['evaluation']['fidelity_metrics']
    )

    df_metrics = pd.DataFrame(list(dict_metrics.items()), columns=['Metric', 'Value'])
    combined_metrics = pd.concat([df_base_metrics, df_metrics], axis=0)

    return combined_metrics, corr_plot, score_plot


