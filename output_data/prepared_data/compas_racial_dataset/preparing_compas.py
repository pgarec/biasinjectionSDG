import sys
import os
from datetime import datetime
import argparse
import pandas as pd

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/")

from sklearn.model_selection import train_test_split
from src.utils import utils_loading
from src.utils import utils_sdv


def main():
    
    parser = argparse.ArgumentParser(
        description="LLM SDG PROJECT"
    )
    parser.add_argument(
        "--config-path", 
        type=str,
        default="./src/configs/config.yaml",
        help="Config path"
    )
    parser.add_argument(
        "--save", 
        type=bool,
        default=True
    )
    args = parser.parse_args()
    config = utils_loading.load_config(args.config_path)
    DATABASE = "compas_racial_dataset"
    TASK = "compas"
    LOCAL_DIR = config["paths"]["local_dir"]
    PATH_PREPARED_DATA = os.path.join(LOCAL_DIR, config["paths"]["prepared_data"].format(database=DATABASE))
    FILE_PREPARED_DATA = config["files"]["prepared_data"].format(database=TASK)
    PATH_METADATA = config["paths"]["metadata"]
    FILE_METADATA = config["files"]["metadata"].format(database=TASK)
    DATE = config["general"].get("date")
    if DATE is None:
        DATE = datetime.today().strftime("%Y-%m-%d")

    args = parser.parse_args()
    print("-----Preparing Dummy data-----")

    df_compas = utils_loading.read_data(
    PATH_PREPARED_DATA, 
    FILE_PREPARED_DATA
    )
    print(df_compas.shape)
    print(df_compas.columns)
    
    df_compas = df_compas.rename(columns={'two_year_recid': 'y'})
    df_compas['id'] = range(1, len(df_compas) + 1)  # Sequential unique IDs
    # utils_loading.save_csv(df_compas,
    #                  LOCAL_DIR, 
    #                  os.path.join(PATH_PREPARED_DATA, FILE_PREPARED_DATA),
    #                  )

    metadata = utils_sdv.get_metadata_from_df(df=df_compas)
    metadata.set_primary_key("id")
    for column in df_compas.columns:
        if column != "id":
            metadata.update_column(column, sdtype="text")  # Free text

    utils_sdv.check_metadata(
        metadata=metadata, 
        primary_key="id"
    )
    train_test_splits = config['train_test_splits'] 
    if train_test_splits:
        print("Creating training and testing sets")
        for k, dict_split in train_test_splits.items():
            X_train, X_test = train_test_split(
                df_compas, 
                test_size=dict_split.get("split"), 
                random_state=dict_split.get("random_state")
            )
            if args.save:
                utils_loading.save_csv(
                    X_train,
                    os.path.join(PATH_PREPARED_DATA, "train_test_splits"),
                    f"X_train_{k}.csv"
                )
                utils_loading.save_csv(
                    X_test,
                    os.path.join(PATH_PREPARED_DATA, "train_test_splits"),
                    f"X_test_{k}.csv"
                )
        # saving
        if args.save:
            print(metadata.to_dict())
            utils_loading.save_dict(metadata.to_dict(),
                            PATH_METADATA,
                            FILE_METADATA)


if __name__ == "__main__":

    main()
