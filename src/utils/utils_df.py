import pandas as pd
import logging
import random
import ast
from sklearn.preprocessing import LabelEncoder
import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])
import utils_loading, utils_sdv


def preprocess_df_compas_generate(df):
    df, label_encoders = encode_categorical(df)
    for col in ['sex', 'y']:
        if df[col].dtype == object and isinstance(df[col].iat[0], (bytes, bytearray)):
            df[col] = df[col].str.decode('utf-8')
        else:
            df[col] = df[col].astype(str).str.strip("b'")

    num_cols = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    df[num_cols] = df[num_cols].fillna(0).astype(int).astype(str)
    race_dummies = ['race_African-American', 'race_Caucasian']
    df['race'] = (
        df[race_dummies]
        .idxmax(axis=1)                   
        .str.replace('race_', '', regex=False)
    )
    df['sex'] = df['sex'].map({'1': 'Male', '0': 'Female'})
    degree_dummies = ['c_charge_degree_F', 'c_charge_degree_M']
    df['c_charge_degree'] = (
        df[degree_dummies]
        .idxmax(axis=1)
        .str.replace('c_charge_degree_', '', regex=False)
    )

    out_cols = ['sex', 'age', 'juv_fel_count', 'juv_misd_count',
                'juv_other_count', 'priors_count',
                'race', 'c_charge_degree', 'y']
    df2= df[out_cols].astype(str)
    return df2, 0


def preprocess_df_compas(df, real=False, inverse=False):
    def gender_to_str(val):
        v = str(val).strip().lower()
        if v in ['male', 'm', '1']:
            return "Male"
        elif v in ['female', 'f', '0']:
            return "Female"
        return str(val).capitalize()

    def one_hot_col(val, target):
        return "1" if str(val).strip().lower() == target.lower() else "0"
    
    if real:
        df, label_encoders = encode_categorical(df)
        return df, 0
    
    else:
        df["sex"] = df["sex"].apply(gender_to_str)
        df["race_African-American"] = df["race"].apply(lambda x: one_hot_col(x, "African-American"))
        df["race_Caucasian"]        = df["race"].apply(lambda x: one_hot_col(x, "Caucasian"))

        if "c_charge_degree" in df.columns:
            df["c_charge_degree_F"] = df["c_charge_degree"].apply(lambda x: one_hot_col(x, "F"))
            df["c_charge_degree_M"] = df["c_charge_degree"].apply(lambda x: one_hot_col(x, "M"))

        if "age" in df.columns:
            df["age"] = df["age"].astype(float)
            df["age_cat_25-45"]        = df["age"].apply(lambda x: "1" if 25 <= x <= 45 else "0")
            df["age_cat_Greaterthan45"]= df["age"].apply(lambda x: "1" if x > 45 else "0")
            df["age_cat_Lessthan25"]   = df["age"].apply(lambda x: "1" if x < 25 else "0")
            df["age"] = df["age"].astype(str)

        df["y"] = df["y"].astype(str)
        df.drop(columns=["race","age_cat","c_charge_degree"], errors="ignore", inplace=True)
        df = df.astype(str)
        df, label_encoders = encode_categorical(df)
        return df, label_encoders
    

def preprocess_df_adult_generate(df):
    return df, 0


def preprocess_df_adult(df, real=False, inverse=False):
    def gender_to_str(val):
        v = str(val).strip().lower()
        if v in ('male','m','1'):   return "Male"
        if v in ('female','f','0'): return "Female"
        return str(val).capitalize()

    def income_to_int(val):
        if val in ('>50K','>=50K'): return 1
        if val in ('<50K','<=50K'): return 0
        return 0

    def one_hot_col(val, target):
        return "1" if str(val).strip().lower() == target.lower() else "0"

    df = df.copy()
    df = df.astype(str)
    df["gender"] = df["gender"].apply(gender_to_str)
    df["income"] = df["income"].apply(income_to_int).astype(float)
    df["race_White"] = df["race"].apply(lambda x: one_hot_col(x, "White")).astype(float)
    df["race_Black"] = df["race"].apply(lambda x: one_hot_col(x, "Black")).astype(float)
    df["race_Other"] = df["race"].apply(lambda x: one_hot_col(x, "Other")).astype(float)

    return df, 0
 

def preprocess_df_diabetes(df, real=False, inverse=False):
    if real:
        df, label_encoders = encode_categorical(df)
        return df, 0

    df = df.copy()
    expected_cols = [
        "Pregnancies", "Glucose", "BloodPressure",
        "SkinThickness", "Insulin", "BMI",
        "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise KeyError(f"The following required columns are missing from the dataframe: {missing}")

    df[expected_cols] = df[expected_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))

    cols_with_zero_as_missing = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]

    df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)
    for col in cols_with_zero_as_missing:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    df["Outcome"] = df["Outcome"].astype(float)

    return df, 0


def preprocess_df_diabetes_generate(df):
    df, label_encoders = encode_categorical(df)
    return df, 0


def preprocess_df_drug(df, real=False, inverse=False):
    return df, 0
    
def preprocess_df_drug_generate(df, real=False, inverse=False):
    return df, 0


_PREPROCESS_FUNCS = {
    "compas": preprocess_df_compas,
    "adult": preprocess_df_adult,
    "diabetes": preprocess_df_diabetes,
    "drug": preprocess_df_drug,
}

_PREPROCESS_FUNCS_GENERATE = {
    "compas": preprocess_df_compas_generate,
    "adult": preprocess_df_adult_generate,
    "diabetes": preprocess_df_diabetes_generate,
    "drug": preprocess_df_drug_generate,
}


def get_preprocess_fn(task: str):
    try:
        return _PREPROCESS_FUNCS[task]
    except KeyError:
        raise ValueError(f"Unsupported task: {task!r}. "
                         f"Available tasks: {list(_PREPROCESS_FUNCS)}")
    

def get_preprocess_fn_generate(task: str):
    try:
        return _PREPROCESS_FUNCS_GENERATE[task]
    except KeyError:
        raise ValueError(f"Unsupported task: {task!r}. "
                         f"Available tasks: {list(_PREPROCESS_FUNCS)}")
    

def load_real_data(config):
    config_general = config["general"]
    config_paths = config["paths"]

    LOCAL_DIR = config_paths["local_dir"]
    TASK = config_general["task"]
    prepared_data_path = os.path.join(config_paths["prepared_data"].format(database=config_general['database'], task=TASK), 
                                      config["files"]["prepared_data"].format(database=TASK))
    df_real = utils_loading.read_data(LOCAL_DIR, prepared_data_path)
    preprocess_df = get_preprocess_fn_generate(TASK)
    df_real, _ = preprocess_df(df_real)

    return df_real


def load_data(config):
    loaded_dataframes = []
    config_general = config["general"]

    LOCAL_DIR = config_general["local_dir"]
    DATABASE = config_general["database"]
    TASK = config_general["task"]
    preprocess_df = get_preprocess_fn(TASK)
    prepared_data_path = config_general["prepared_data_path"].format(database=config_general['database'], task=TASK)
    df_real = utils_loading.read_data(LOCAL_DIR, prepared_data_path)
    dict_metadata = utils_loading.read_dict(config_general["local_dir"], config_general["metadata_path"].format(task=TASK))
    sdv_metadata = utils_sdv.get_metadata_from_dict(dict_metadata=dict_metadata)

    df_real = utils_sdv.custom_validate_data(df=df_real, metadata=sdv_metadata)
    df_real, _ = preprocess_df(df_real, True)

    for config_dataframe in config["dataframes"]:  
        synth_data_path = config_general["synthesized_data_path"].format(
            sdg_model=config_dataframe["sdg_model"],
            task = TASK,
            prompt_neutrality = config_dataframe["prompt_neutrality"],
            icl_gender=config_dataframe["icl_gender"],
            prompt_id=config_dataframe["prompt_id"]
        )

        if "mild_rate" in config_dataframe:
            synth_full_path = os.path.join(
                LOCAL_DIR,
                synth_data_path,
                config_dataframe['synthesized_data_file'].format(database=DATABASE, bias=config_dataframe['bias'], icl_records=config_dataframe["icl_records"], mild_rate=config_dataframe["mild_rate"])
            )
        else:
            synth_full_path = os.path.join(
                LOCAL_DIR,
                synth_data_path,
                config_dataframe['synthesized_data_file'].format(database=DATABASE, bias=config_dataframe['bias'], icl_records=config_dataframe["icl_records"])
            )
        df = utils_loading.read_data(LOCAL_DIR, synth_full_path)
        primary_key = sdv_metadata.to_dict().get('primary_key')
        df = add_primary_key(df=df, primary_key=primary_key)

        all_cols_in_list_bool = all(col in df.columns for col in sdv_metadata.get_column_names())
        if not all_cols_in_list_bool:
            df.drop(columns=['id'], inplace=True)
            df = parse_dataframe(df)

        df, _ = preprocess_df(df)
        loaded_dataframes.append(df)

    return df_real, loaded_dataframes, sdv_metadata


def categorize_columns(df: pd.DataFrame, threshold: int = 5) -> dict:
    discrete_cols = []
    continuous_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if (
                len(df[col].unique()) <= threshold
            ):  # Adjust the threshold for discrete vs. continuous as needed
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]):
            discrete_cols.append(col)

    return {"discrete": discrete_cols, "continuous": continuous_cols}


def shuffle_dict(d: dict) -> dict:
    keys = list(d.keys())
    random.shuffle(keys)
    return {key: d[key] for key in keys}


def add_primary_key(df: pd.DataFrame,
                    primary_key: str) -> pd.DataFrame:
    if primary_key not in df.columns:
        df[primary_key] = range(df.shape[0])
    else:
        if len(df[primary_key].drop_duplicates()) < df.shape[0]:
            logging.info("Primary key is not unique, generating a new one")
            df[primary_key] = range(df.shape[0])    
    return df


def rm_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.isnull().any(axis=1)] \
                .reset_index() \
                .drop(columns="index")
    return df


def parse_dataframe(df):
    records = []
    for index, row in df.iterrows():
        for cell in row:
            if pd.isna(cell):
                continue
            try:
                parsed = ast.literal_eval(cell)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing row {index}: {e}")
                continue
            # if it’s a single dict, great
            if isinstance(parsed, dict):
                records.append(parsed)
            # if it’s a list, pull out any dicts inside it
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        records.append(item)
            # otherwise ignore
    # debug: you should now see only dicts in records
    # use from_records to avoid the “.keys()” fallback entirely
    return pd.DataFrame.from_records(records)


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
