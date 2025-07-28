import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import ast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

import utils_loading, utils_df
from src.evaluation.metrics import evaluate_fidelity
from src.utils import utils_sdv
from src.utils.utils_df import add_primary_key

from sklearn.preprocessing import LabelEncoder
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset


def drop_date_columns(df, date_columns):
    return df.drop(columns=[col for col in date_columns if col in df.columns], errors='ignore')


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


def prepare_adult_dataset(dataframe, real=False):
    df = dataframe.copy()
    df['privileged'] = (df['gender'] == 'Male').astype(float)
    label_name = 'race_Black'
    df, _ = encode_categorical(df)
    selected_features = ["age", "workclass", "fnlwgt", "education", "educational-num", "marital-status", "occupation",
                         "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    dataset_obj = StandardDataset(
        df=df,
        label_name=label_name,
        favorable_classes=[1.0],
        protected_attribute_names=['privileged'],
        privileged_classes=[[1.0]],
        features_to_keep=selected_features
    )
    dataset_obj.scores = dataset_obj.labels.copy()
    
    return dataset_obj


def prepare_compas_dataset(dataframe):
    """ TODO: now the privileged attribute is harcoded to Sex and label to race_afroamerican. Study other?"""
    df = dataframe.copy()
    df['privileged'] = df['sex'].astype(float)
    df['race_African-American'] = df['race_African-American'].astype(float)
    label_name = 'race_African-American'
    favorable_class_numeric = 1

    selected_features = [
        'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_charge_degree_F', 'c_charge_degree_M', 'privileged'
    ]
    
    dataset_obj = StandardDataset(
        df=df,
        label_name=label_name,
        favorable_classes=[favorable_class_numeric],
        protected_attribute_names=['privileged'],
        privileged_classes=[[1]],
        features_to_keep=selected_features
    )
    dataset_obj.scores = dataset_obj.labels.copy()
    return dataset_obj


def prepare_adult_dataset_attack(dataframe, real=False):
    df = dataframe.copy()
    df['privileged'] = df["race_Black"]
    label_name = 'income'
    df, _ = encode_categorical(df)
    selected_features = ["age", "workclass", "fnlwgt", "education", "educational-num", "marital-status", "occupation",
                         "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    dataset_obj = StandardDataset(
        df=df,
        label_name=label_name,
        favorable_classes=[1],
        protected_attribute_names=['privileged'],
        privileged_classes=[[1.0]],
        features_to_keep=selected_features
    )
    dataset_obj.scores = dataset_obj.labels.copy()
    
    return dataset_obj


def prepare_compas_dataset_attack(dataframe):
    df = dataframe.copy()
    df['privileged'] = df['race_African-American']
    df['y'] = df['y'].astype(float)
    label_name = 'y'
    favorable_class_numeric = [1]

    selected_features = [
        'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_charge_degree_F', 'c_charge_degree_M', 'privileged'
    ]
    
    dataset_obj = StandardDataset(
        df=df,
        label_name=label_name,
        favorable_classes=favorable_class_numeric,
        protected_attribute_names=['privileged'],
        privileged_classes=[[1]],
        features_to_keep=selected_features
    )
    dataset_obj.scores = dataset_obj.labels.copy()
    return dataset_obj


def prepare_diabetes_dataset_attack(dataframe):
    df = dataframe.copy()
    df['privileged'] = df['Age'] <= 30
    df['Outcome'] = df['Outcome'].astype(float)
    label_name = 'Outcome'
    favorable_class_numeric = [1]

    selected_features = [
        'Pregnancies', 'Pregnancies', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    dataset_obj = StandardDataset(
        df=df,
        label_name=label_name,
        favorable_classes=favorable_class_numeric,
        protected_attribute_names=['privileged'],
        privileged_classes=[[1]],
        features_to_keep=selected_features
    )
    dataset_obj.scores = dataset_obj.labels.copy()
    return dataset_obj


def compute_fairness_metrics(dataset):
    privileged_groups = [{'privileged': 1}]
    unprivileged_groups = [{'privileged': 0}]
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )
    
    # Index of the privileged column
    priv_idx = dataset.feature_names.index('privileged')
    labels = dataset.labels.ravel()  # Flatten in case it's (n, 1)
    privileged_mask = dataset.features[:, priv_idx] == 1
    unprivileged_mask = dataset.features[:, priv_idx] == 0

    mean_label_privileged = labels[privileged_mask].mean()
    mean_label_unprivileged = labels[unprivileged_mask].mean()
    
    num_privileged = privileged_mask.sum()
    num_unprivileged = unprivileged_mask.sum()

    return {
        'mean_difference': metric.mean_difference(),
        'statistical_parity_difference': metric.statistical_parity_difference(),
        'disparate_impact': metric.disparate_impact(),
        'n_privileged': num_privileged,
        'n_unprivileged': num_unprivileged,
        'mean_label_privileged': mean_label_privileged,
        'mean_label_unprivileged': mean_label_unprivileged
    }


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
