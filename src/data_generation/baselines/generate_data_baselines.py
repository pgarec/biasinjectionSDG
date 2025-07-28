import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")

from src.utils.utils_loading import load_config
from src.utils import utils_df
from src.data_generation.baselines.models import synthcity_generate
from src.evaluation.metrics.evaluate_quality import encode_categorical
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="LLM SDG PROJECT")
    parser.add_argument("--config-path", type=str, default="./src/configs/config.yaml", help="Path to the configuration file")
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()

    config = load_config(args.config_path)
    cfg_general = config['general']
    cfg_paths = config['paths']
    cfg_general["task"] = "adult"
    cfg_general["database"] = "adult_dataset"
    train_size = 100
    output_size = 10
    
    DATABASE = cfg_general["database"]
    LOCAL_DIR = cfg_paths["local_dir"]
    TASK = cfg_general["task"]
    DF_PATH = "output_data/baselines/{database}/"

    preprocess_df = utils_df.get_preprocess_fn(TASK)
    df_real = utils_df.load_real_data(config)
    df_real, _ = preprocess_df(df_real, True)
    desired_order_adult = [
            'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            'marital-status', 'occupation', 'relationship', 'race', 'gender',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income', 'race_White', 'race_Black', 'race_Other', 'y'
    ]
    desired_order_compas = ['sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
       'priors_count', 'y', 'age_cat_25-45', 'age_cat_Greaterthan45',
       'age_cat_Lessthan25', 'race_African-American', 'race_Caucasian',
       'c_charge_degree_F', 'c_charge_degree_M']

    if config["general"]["task"] == "adult":
        df_real["y"] = df_real["income"]
        df_real = df_real[desired_order_adult]
        df_real.drop(columns=['income'], inplace=True)
    
    else:
        df_real = df_real[desired_order_compas]

    df_real, _  = encode_categorical(df_real)
    train_real_df, test_real_df = train_test_split(df_real, test_size=0.2, random_state=42)
    df_train = train_real_df[:train_size]

    generative_models = ["tvae", "ctgan", "nflow", "ddpm"]
    for model in generative_models:
        df = synthcity_generate(model, df_train, output_size) 

        if args.save:
            base_save_path = os.path.join(LOCAL_DIR, DF_PATH.format(database=DATABASE))  
            os.makedirs(base_save_path, exist_ok=True)

            df_file_path = os.path.join(base_save_path, f"{model}_synthetic_data.csv")
            df.to_csv(df_file_path, index=False)

            print(f"Synthetic dataframe from {model} saved to: {df_file_path}")


if __name__ == "__main__":
    main()