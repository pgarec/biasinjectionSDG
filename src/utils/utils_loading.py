import logging
import json
import logging
import pickle
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
import yaml
from typing import Union, List, Dict, Any

import pandas as pd
import re
from typing import List, Dict


def reverse_mapper(df_real: pd.DataFrame,
                   mapper: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {col: {v: k for k, v in m.items()} for col, m in mapper.items()}


def unmap_row(row: List[str], header: List[str],
              rev_mapper: Dict[str, Dict[str, str]]) -> List[str]:
    out = []
    for col, val in zip(header, row):
        if col in rev_mapper and val in rev_mapper[col]:
            out.append(rev_mapper[col][val])
        else:
            out.append(val)
    return out
    

def load_experiments(experiments_path: str, experiments_name: str):
    """Load experiments from a YAML file."""
    with open(experiments_path, 'r') as f:
        experiments_data = yaml.safe_load(f)
    # Assume the YAML file has a top-level key "experiments"
    return experiments_data[experiments_name]


def read_data(
    local_dir: str,
    path_file: str,
    filename: str = None,
    sep: str = ",",
    compression: str = "infer",
) -> pd.DataFrame:
    if filename:
        path_file = os.path.join(path_file, filename)
    
    local_path = os.path.join(local_dir, path_file)
    logging.info("Data will be loaded from {}".format(local_path))
    
    df = pd.read_csv(local_path, sep=sep, compression=compression)
    return df


def save_csv(
    df: pd.DataFrame,
    local_dir: str,
    path_file: str,
    filename: str = None,
    index: bool = False,
) -> None:
    if filename:
        path_file = os.path.join(path_file, filename)
    
    local_path = os.path.join(local_dir, path_file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    logging.info("Data will be saved in {}".format(local_path))
    df.to_csv(local_path, index=index, mode='w')


async def save_csv_async(
    df: pd.DataFrame,
    local_dir: str,
    path_file: str,
    filename: str = None,
    index: bool = False,
) -> None:
    if filename:
        path_file = os.path.join(path_file, filename)
    
    local_path = os.path.join(local_dir, path_file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    logging.info("Data will be saved in {}".format(local_path))
    df.to_csv(local_path, index=index, mode='w')


def read_dict(local_dir: str, path: str):
    local_path = os.path.join(local_dir, path)
    print(local_path)
    if not os.path.exists(local_path):
        print("NO SUCH FILE: {}".format(local_path))
        return None

    try:
        with open(local_path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        return json_dict

    except json.decoder.JSONDecodeError as e:
        logging.info("JSON file is not properly formatted: {}".format(local_path))
        logging.info(e)
        return None

   
def save_dict(dict_data: dict,
              local_dir: str,
              path_file: str) -> None:

    local_path = os.path.join(local_dir, path_file)
    dir_name = os.path.dirname(local_path)  # Get directory path

    if not os.path.exists(dir_name):  # Check if directory exists
        os.makedirs(dir_name, exist_ok=True)  # Create if necessary

    try:
        # Save the dictionary as a JSON file
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, indent=4)
        logging.info(f"Dictionary saved to {local_path}")
    except Exception as e:
        logging.error(f"Error saving dictionary to {local_path}: {e}")

    
async def save_dict_async(dict_data: dict,
              local_dir: str,
              path_file: str) -> None:

    local_path = os.path.join(local_dir, path_file)
    dir_name = os.path.dirname(local_path)  # Get directory path

    if not os.path.exists(dir_name):  # Check if directory exists
        os.makedirs(dir_name, exist_ok=True)  # Create if necessary

    try:
        # Save the dictionary as a JSON file
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, indent=4)
        logging.info(f"Dictionary saved to {local_path}")
    except Exception as e:
        logging.error(f"Error saving dictionary to {local_path}: {e}")
    

def save_text(text: str, local_dir: str, path_file: str, filename: str = None) -> None:
    if filename:
        path_file = os.path.join(path_file, filename)
    
    local_path = os.path.join(local_dir, path_file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Text saved to {local_path}")
    except Exception as e:
        logging.error(f"Error saving text to {local_path}: {e}")


async def save_text_async(text: str, local_dir: str, path_file: str, filename: str = None) -> None:
    if filename:
        path_file = os.path.join(path_file, filename)
    
    local_path = os.path.join(local_dir, path_file)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Text saved to {local_path}")
    except Exception as e:
        logging.error(f"Error saving text to {local_path}: {e}")


def extract_json_as_dict(json_file):
    if isinstance(json_file, (dict, list)):  
        return json_file  # If already a dictionary or list, return as-is
    try:
        return json.loads(json_file)  # Try parsing if it's a string
    except (ValueError, json.JSONDecodeError):
        print("JSON decode error")
        print(json_file)
        return None


def extract_json_as_dict_slurm(text):
    if isinstance(text, (dict, list)):
        return text

    # Collapse whitespace
    cleaned = text.strip()
    # Find first {...} or [...] block
    m = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
    if not m:
        print("JSON decode error (no JSON block found)")
        print(text)
        return None

    json_block = m.group(1)
    try:
        return json.loads(json_block)
    except json.JSONDecodeError:
        print("JSON decode error")
        print(json_block)
        return None



def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_text(content, path, filename):
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    file_path = output_path / filename

    try:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Prompt saved successfully to {file_path}")
    except Exception as e:
        print(f"Failed to save text file: {e}")


def load_model(model_path):    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer loaded.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        local_files_only=True, 
        torch_dtype="auto", 
        device_map="auto" if torch.cuda.is_available() else None
    )
    print("Model loaded.")

    return model, tokenizer


def save_model(model, local_dir: str, folder_path: str, file_name: str):
    local_path = os.path.join(local_dir, folder_path, file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        with open(local_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {local_path}")
    except Exception as e:
        logging.error(f"Error saving model to {local_path}: {e}")


def save_synthetic_data(data, output_path):
    """Stores generated synthetic dataset in a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  

    if output_path.exists():
        os.remove(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved synthetic dataset to {output_path}")

