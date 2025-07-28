import random
import os
import json
import litellm, os
import asyncio
import sys
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")
from src.utils.utils_loading import extract_json_as_dict, save_csv_async
from src.utils.utils_prompt import (generate_compas_racial_examples, 
                                    generate_adult_examples,
                                    generate_diabetes_examples,
                                    generate_drug_examples,
                                    inject_icl_examples)


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

        if inverse:
            race_cols = ['race_African-American', 'race_Caucasian']
            def pick_race(row):
                for c in race_cols:
                    if str(row.get(c, '0')) == '1.0':
                        return c.replace('race_', '')
                return None
            df['race'] = df.apply(pick_race, axis=1)

            if {'c_charge_degree_F','c_charge_degree_M'}.issubset(df.columns):
                def pick_charge(row):
                    if str(row['c_charge_degree_F']) == '1.0': return 'F'
                    if str(row['c_charge_degree_M']) == '1.0': return 'M'
                    return None
                df['c_charge_degree'] = df.apply(pick_charge, axis=1)

            age_map = {
                'age_cat_Lessthan25': 'Lessthan25',
                'age_cat_25-45':      '25-45',
                'age_cat_Greaterthan45': 'Greaterthan45'
            }

            drop_cols = race_cols + list(age_map.keys()) + ['c_charge_degree_F','c_charge_degree_M']
            df = df.drop(columns=drop_cols, errors='ignore')

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


def preprocess_df_drug(df, real=False, inverse=False):


    return df, 0
    

_PREPROCESS_FUNCS = {
    "compas": preprocess_df_compas,
    "adult": preprocess_df_adult,
    "diabetes": preprocess_df_diabetes,
    "drug": preprocess_df_drug,
}


def get_preprocess_fn(task: str):
    try:
        return _PREPROCESS_FUNCS[task]
    except KeyError:
        raise ValueError(f"Unsupported task: {task!r}. "
                         f"Available tasks: {list(_PREPROCESS_FUNCS)}")


class IdentityScaler:
    def fit(self, X): return self
    def transform(self, X): return X


class TabularDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, scalers, encoders):
        n = len(df)

        # numeric side
        if num_cols:
            X_num = df[num_cols].fillna(0.0).values.astype(np.float32)
            self.X_num = scalers['num'].transform(X_num)
        else:
            # empty numeric → zero‐width array
            self.X_num = np.zeros((n, 0), dtype=np.float32)

        # categorical side
        self.cat_cols = cat_cols
        if cat_cols:
            cat_arrays = []
            for c in cat_cols:
                arr = df[c].fillna("NA").astype(str).values
                cat_arrays.append(encoders[c].transform(arr))
            # stack into (n, num_cats)
            self.X_cat = np.stack(cat_arrays, axis=1).astype(np.int64)
        else:
            # empty categorical → zero‐width int array
            self.X_cat = np.zeros((n, 0), dtype=np.int64)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return {
            'num': torch.from_numpy(self.X_num[idx]),
            'cat': torch.from_numpy(self.X_cat[idx]),
        }


class TabularAutoencoder(nn.Module):
    def __init__(self, num_feat_dim, cat_dims, emb_dim=8,
                 latent_dim=32, hidden_dims=[128,64]):
        super().__init__()
        # embeddings (possibly empty)
        self.embs = nn.ModuleList([
            nn.Embedding(n_cat, emb_dim) for n_cat in cat_dims
        ])
        input_dim = num_feat_dim + emb_dim * len(cat_dims)

        # encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        enc += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        # decoder (mirror)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x_num, x_cat):
        batch = x_num.shape[0]

        # embed cats if any
        if len(self.embs) > 0:
            emb = [layer(x_cat[:,i]) for i, layer in enumerate(self.embs)]
            emb = torch.cat(emb, dim=1)
        else:
            emb = x_num.new_zeros((batch, 0))

        x = torch.cat([x_num, emb], dim=1)
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon
    

def train_encoder(df_train, latent_dim=32, emb_dim=8, hidden_dims=[128,64],
                  batch_size=256, lr=1e-3, n_epochs=50,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):

    # 1) find numeric / categorical
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train.select_dtypes(include=['object','category']).columns.tolist()

    # 2) build scalers + encoders
    scalers = {}
    if num_cols:
        scalers['num'] = StandardScaler().fit(df_train[num_cols].fillna(0.0).values)
    else:
        scalers['num'] = IdentityScaler()

    encoders = {}
    cat_dims = []
    for c in cat_cols:
        le = LabelEncoder().fit(df_train[c].fillna("NA").astype(str).values)
        encoders[c] = le
        cat_dims.append(len(le.classes_))

    # 3) data + loader
    ds = TabularDataset(df_train, num_cols, cat_cols, scalers, encoders)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # 4) model
    model = TabularAutoencoder(
        num_feat_dim=len(num_cols),
        cat_dims=cat_dims,
        emb_dim=emb_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # 5) train
    model.train()
    for epoch in range(n_epochs):
        total = 0.0
        for batch in loader:
            x_num = batch['num'].to(device)
            x_cat = batch['cat'].to(device)

            z, recon = model(x_num, x_cat)

            # build original input vector
            with torch.no_grad():
                if len(model.embs) > 0:
                    emb_orig = [model.embs[i](x_cat[:,i]) for i in range(x_cat.shape[1])]
                    emb_orig = torch.cat(emb_orig, dim=1)
                    input_vec = torch.cat([x_num, emb_orig], dim=1)
                else:
                    input_vec = x_num

            loss = mse(recon, input_vec)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x_num.size(0)

        if (epoch+1)%10==0 or epoch==0:
            avg = total / len(ds)
            print(f"[{epoch+1}/{n_epochs}] MSE: {avg:.4f}")

    return {
        'model': model.eval().cpu(),
        'scalers': scalers,
        'encoders': encoders,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
    }


def mean_embed(model, df, num_cols, cat_cols, scalers, encoders, device = 'cpu'):
    ds = TabularDataset(df, num_cols, cat_cols, scalers, encoders)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    embs = []
    with torch.no_grad():
        for batch in loader:
            x_num = batch['num'].to(device)
            x_cat = batch['cat'].to(device)
            z, _  = model(x_num, x_cat)
            embs.append(z.cpu().numpy())
    all_z = np.vstack(embs)
    return all_z.mean(axis=0)


def mean_dist_euclidean(model, df, vec_ref, num_cols, cat_cols, scalers, encoders, device = 'cpu'):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    # 2) build scalers + encoders
    scalers = {}
    if num_cols:
        scalers['num'] = StandardScaler().fit(df[num_cols].fillna(0.0).values)
    else:
        scalers['num'] = IdentityScaler()
        
    ds = TabularDataset(df, num_cols, cat_cols, scalers, encoders)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    all_dists = []
    sum_dists = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            x_num = batch['num'].to(device, non_blocking=True)
            x_cat = batch['cat'].to(device, non_blocking=True)
            z, _ = model(x_num, x_cat)                   # (B, latent_dim)
            d = (z - vec_ref).norm(dim=1, p=2)               # (B,)
            
            all_dists.append(d.cpu())
            sum_dists += d.sum().item()
            count += d.size(0)

    distances = torch.cat(all_dists).numpy()             # (n_samples,)
    mean_distance = sum_dists / count
    return distances, mean_distance


def mean_dist_mahalanobis(model, df, df_ref, vec_ref, num_cols, cat_cols, scalers, encoders, device = 'cpu'):
    ds = TabularDataset(df, num_cols, cat_cols, scalers, encoders)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    all_z  = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_num = batch['num'].to(device, non_blocking=True)
            x_cat = batch['cat'].to(device, non_blocking=True)
            z, _  = model(x_num, x_cat)          # (B, D)
            all_z.append(z.cpu())
    Z_np = torch.cat(all_z, dim=0).numpy()     # (N, D)

    ds_ref     = TabularDataset(df_ref, num_cols, cat_cols, scalers, encoders)
    loader_ref = DataLoader(ds_ref, batch_size=512, shuffle=False)
    with torch.no_grad():
        for batch in loader_ref:
            x_num = batch['num'].to(device, non_blocking=True)
            x_cat = batch['cat'].to(device, non_blocking=True)
            z, _  = model(x_num, x_cat)          # (B, D)
            all_z.append(z.cpu())
    Z_np_ref = torch.cat(all_z, dim=0).numpy()     # (N, D)

    # compute & invert covariance ---
    cov = np.cov(Z_np_ref, rowvar=False)           # (D, D)
    eps = 1e-5
    cov += np.eye(cov.shape[0]) * eps
    inv_cov = torch.from_numpy(np.linalg.inv(cov))\
                .float().to(device)        # (D, D)

    v = torch.tensor(vec_ref, dtype=torch.float, device=device)  # <— here

    # compute Mahalanobis distances ---
    all_dists = []
    sum_dists = 0.0
    count     = 0

    with torch.no_grad():
        for batch in loader:
            x_num = batch['num'].to(device, non_blocking=True)
            x_cat = batch['cat'].to(device, non_blocking=True)
            z, _ = model(x_num, x_cat)        # (B, D)

            diff = z - v                       # (B, D)
            m = diff @ inv_cov             # (B, D)
            d = torch.sqrt((m * diff).sum(dim=1))  # (B,)

            all_dists.append(d.cpu())
            sum_dists += d.sum().item()
            count += d.size(0)

    distances = torch.cat(all_dists).numpy()   # (n_samples,)
    mean_distance = sum_dists / count
    return distances, mean_distance


def mitigate_bias(examples, df_reference, cfg, bundle):
    model  = bundle['model']
    scalers = bundle['scalers']
    encoders = bundle['encoders']
    num_cols = bundle['num_cols']
    cat_cols = bundle['cat_cols']

    vec_real = mean_embed(model, df_reference, num_cols, cat_cols, scalers, encoders)
    df = pd.DataFrame(examples)

    preprocess_df = get_preprocess_fn(cfg["task"])
    df, _ = preprocess_df(df, True)
    distances, _ = mean_dist_euclidean(model, df, vec_real,
                                       num_cols, cat_cols, scalers, encoders)

    # Compute CDF values for each distance
    sorted_dist = np.sort(distances)
    cdf_vals = np.searchsorted(sorted_dist, distances, side='right') / len(distances)

    # Attach distances and CDF to DataFrame
    df['distance'] = distances
    df['cdf'] = cdf_vals

    # Determine threshold for top 10% outliers
    threshold = np.percentile(distances, 90)

    # Filter out examples with distance above threshold
    df_filtered = df[df['distance'] <= threshold].reset_index(drop=True)

    # Convert back to list of dicts
    filtered_examples = df_filtered.to_dict(orient='records')
    return filtered_examples
