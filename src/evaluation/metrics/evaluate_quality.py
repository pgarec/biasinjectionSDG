import os
import sys
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

from sklearn.preprocessing import LabelEncoder
from sklearn import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def encode_categorical(df, exclude_columns=None):
    exclude_columns = exclude_columns or []
    label_encoders = {}
    categorical_columns = [col for col in df.select_dtypes(include=['object']).columns.tolist() 
                           if col not in exclude_columns]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


def evaluate_ground_truth_models(df_real, models=None):
    if models is None:
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        }

    df_real, _  = encode_categorical(df_real)
    train_real_df, test_real_df = train_test_split(df_real, test_size=0.2, random_state=42)

    X_train_real = train_real_df.drop(columns=["y"])
    y_train_real = train_real_df["y"]
    X_test_real = test_real_df.drop(columns=["y"])
    y_test_real = test_real_df["y"]

    results = []

    for name, model in models.items():
        mdl = clone(model)
        mdl.fit(X_train_real, y_train_real)

        y_pred = mdl.predict(X_test_real)
        acc_val = accuracy_score(y_test_real, y_pred)
        f1_val = f1_score(y_test_real, y_pred, average="weighted")
        prec_val = precision_score(y_test_real, y_pred, average="weighted", zero_division=0)
        recall_val = recall_score(y_test_real, y_pred, average="weighted", zero_division=0)

        y_proba = mdl.predict_proba(X_test_real)[:, 1]
        mask_privileged = X_test_real["target_group"] == 1
        mask_non_privileged = X_test_real["target_group"] == 0

        def avg_or_nan(probas, mask):
            return probas[mask].mean() if mask.sum() > 0 else np.nan

        avg_privileged = avg_or_nan(y_proba, mask_privileged)
        avg_non_privileged = avg_or_nan(y_proba, mask_non_privileged)

        results.append({
            "name": "Real baseline",
            "model": f"{name} (real)",
            "acc_real": acc_val,
            "f1_real": f1_val,
            "precision_real": prec_val,
            "recall_real": recall_val,
            "avg_proba_y1_privileged": avg_privileged,
            "avg_proba_y1_non_privileged": avg_non_privileged,
        })

    return results



def evaluate_dataset_models(df_real, df_synthetic, config_general, config_df,
                            save=True, models=None, seeds=None):
    if models is None:
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100)
        }
    # default seeds if not provided
    if seeds is None:
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    df_synthetic, _ = encode_categorical(df_synthetic)
    df_real, _ = encode_categorical(df_real)
    _, test_real_df = train_test_split(df_real, test_size=0.2, random_state=42)
    train_synt_df, _= train_test_split(df_synthetic, test_size=0.2, random_state=42)

    X_train = train_synt_df.drop(columns=["y"])
    y_train = train_synt_df["y"]
    X_test = test_real_df.drop(columns=["y"])
    y_test = test_real_df["y"]

    assert list(X_train.columns) == list(X_test.columns), \
        "Column order mismatch between train & test!"
    
    # assert (train_synt_df.dtypes == test_real_df.dtypes).all()

    def avg_or_nan(probas, mask):
        return probas[mask].mean() if mask.sum() > 0 else np.nan

    summary = []

    for name, base_model in models.items():
        metrics = {
            "acc": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "avg_proba_y1_privileged": [],
            "avg_proba_y1_non_privileged": []
        }

        for seed in seeds:
            mdl = clone(base_model)
            try:
                mdl.set_params(random_state=seed)
            except (ValueError, TypeError):
                pass

            mdl.fit(X_train, y_train)
            y_pred  = mdl.predict(X_test)
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X_test)
                if p.shape[1] == 2:
                    y_proba = p[:, 1]  
                else:
                    if np.all(p == 1.0):
                        y_proba = np.ones(len(X_test))
                    else:
                        y_proba = np.zeros(len(X_test))
            else:
                y_proba = y_pred.astype(float)

            mask_b = X_test["target_group"] == 1
            mask_nb = X_test["target_group"] == 0

            metrics["acc"].append(accuracy_score(y_test, y_pred) )
            metrics["f1"].append(f1_score(y_test, y_pred, average="weighted") )
            metrics["precision"].append(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall"].append(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["avg_proba_y1_privileged"].append(
                avg_or_nan(y_proba, mask_b)
            )
            metrics["avg_proba_y1_non_privileged"].append(
                avg_or_nan(y_proba, mask_nb)
            )

        aggregated = {}
        for metric, values in metrics.items():
            vals = np.array(values, dtype=float)
            aggregated[f"{metric}_mean"] = np.nanmean(vals)
            aggregated[f"{metric}_var"]  = np.nanvar(vals)

        summary.append({
            "dataset": config_df["name"],
            "model":   f"{name} (synthetic)",
            **aggregated
        })

    print("\nEvaluation Results (mean ± variance):\n")
    for r in summary:
        print(f"Dataset: {r['dataset']}")
        print(f"Model: {r['model']}")
        print(f"Accuracy: {r['acc_mean']:.3f} ± {r['acc_var']:.5f}")
        print(f"F1 Score: {r['f1_mean']:.3f} ± {r['f1_var']:.5f}")
        print(f"Precision: {r['precision_mean']:.3f} ± {r['precision_var']:.5f}")
        print(f"Recall: {r['recall_mean']:.3f} ± {r['recall_var']:.5f}")
        print("-" * 50)

    return summary

