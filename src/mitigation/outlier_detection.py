import os
import sys
import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap.umap_ as umap
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.extend([script_dir, "./src/", "./src/data_generation/", "./src/utils"])

import utils_loading, utils_df
from src.evaluation.metrics.evaluate_fairness import prepare_adult_dataset_attack, prepare_compas_dataset_attack, prepare_diabetes_dataset_attack
from src.evaluation.metrics.evaluate_quality import evaluate_dataset_models, evaluate_ground_truth_models
from src.evaluation.metrics import evaluate_fidelity
from src.utils.utils_prompt import (generate_compas_racial_examples, 
                                    generate_adult_examples,
                                    generate_diabetes_examples,
                                    generate_drug_examples,
                                    inject_icl_examples)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder


_PREPARE_FUNCS = {
    "compas": prepare_compas_dataset_attack,
    "adult": prepare_adult_dataset_attack,
    "diabetes": prepare_diabetes_dataset_attack,
}


desired_order = {
    "adult": [
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income', 'race_White', 'race_Black', 'race_Other', 'y'],
    "compas": ['sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'y', 'age_cat_25-45', 'age_cat_Greaterthan45',
        'age_cat_Lessthan25', 'race_African-American', 'race_Caucasian',
        'c_charge_degree_F', 'c_charge_degree_M'],
    "diabetes": ['Age', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction', 'Glucose',
        'Insulin', 'Outcome', 'Pregnancies', 'SkinThickness']}


def get_prepare_fn(task: str):
    try:
        return _PREPARE_FUNCS[task]
    except KeyError:
        raise ValueError(f"Unsupported task: {task!r}. "
                         f"Available tasks: {list(_PREPARE_FUNCS)}")


def preprocess_df(df, task, desired_order):
    if task == "adult":
        df["y"] = df["income"]
        df = df[desired_order]
        df.drop(columns=['income', 'race'], inplace=True)
        cols = [c for c in df.columns if "black" in c.lower()]
        col = cols[0] if cols else None
        df["target_group"] = col

    elif task == "compas":
        df = df[desired_order]
        cols = [c for c in df.columns if "black" in c.lower()]
        col = cols[0] if cols else None
        df["target_group"] = col

    elif task == "diabetes":
        df = df[desired_order]
        df["y"] = df["Outcome"]
        df.drop(columns=['Outcome'], inplace=True)
        df["target_group"] = df["Age"] <= 30

    return df


def visualize_embedding_drift(
    df_real,
    dataframes_synthetic,
    bias_rates,
    method='umap',
    sample_size=500,
    random_state=42
):

    df_real_num = df_real.select_dtypes(include=[np.number]).fillna(0)
    feature_cols = df_real_num.columns

    synth_numerics = []
    for df_s in dataframes_synthetic:
        df_num = df_s.select_dtypes(include=[np.number]).fillna(0)
        df_num = df_num.reindex(columns=feature_cols, fill_value=0)
        synth_numerics.append(df_num)

    sizes = [len(df_real_num)] + [len(d) for d in synth_numerics]
    max_n = min(sizes)
    if sample_size > max_n:
        print(f"⚠️ sample_size {sample_size} > smallest dataset {max_n}, using {max_n}")
        sample_size = max_n

    real_samp = df_real_num.sample(n=sample_size, random_state=random_state)
    scaler = StandardScaler().fit(real_samp)
    real_scaled = scaler.transform(real_samp)

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state)
    else:
        reducer = umap.UMAP(random_state=random_state)

    real_emb = reducer.fit_transform(real_scaled)

    fig, axes = plt.subplots(2, len(synth_numerics)//2, figsize=(15,6), constrained_layout=True)
    axes = axes.flatten()

    for ax, (df_num, rate) in zip(axes, zip(synth_numerics, bias_rates)):
        synth_samp = df_num.sample(n=sample_size, random_state=random_state)
        synth_scaled = scaler.transform(synth_samp)
        synth_emb = reducer.transform(synth_scaled)

        sns.kdeplot(
            x=real_emb[:,0], y=real_emb[:,1],
            fill=True, cmap='Blues', alpha=0.5,
            label='Real', ax=ax
        )
        sns.kdeplot(
            x=synth_emb[:,0], y=synth_emb[:,1],
            fill=True, cmap='Reds', alpha=0.5,
            label=f'Synth bias={rate}', ax=ax
        )

        ax.set_title(f'Bias rate = {rate}')
        ax.legend(loc='upper right', fontsize='small')
        ax.axis('off')

    plt.suptitle('Embedding Drift Across Increasing Bias Levels', fontsize=16)
    plt.savefig("./figures/mitigation/umap.pdf", bbox_inches='tight', format='pdf')
    plt.show()


def compute_distances(df_real, dataframes_granite, metadata, config):
    experiment = config["general"]["experiment"]
    task = config["general"]["task"]
    
    df_noicl    = dataframes_granite[0]
    df_iclreal  = dataframes_granite[1]

    mild_rates = [conf["mild_rate"] for conf in config["dataframes"][2:]]
    metrics_noicl   = {'JSD': [], 'TVComplement': []}
    metrics_iclreal = {'JSD': [], 'TVComplement': []}

    for conf, df in zip(config["dataframes"][2:], dataframes_granite[2:]):
        res = evaluate_fidelity.compute_fidelity_metrics(df_noicl, df, metadata)
        metrics_noicl['JSD'].append(res['JSD']['mean'])
        metrics_noicl['TVComplement'].append(res['TVComplement']['mean'])

    for conf, df in zip(config["dataframes"][2:], dataframes_granite[2:]):
        res = evaluate_fidelity.compute_fidelity_metrics(df_iclreal, df, metadata)
        metrics_iclreal['JSD'].append(res['JSD']['mean'])
        metrics_iclreal['TVComplement'].append(res['TVComplement']['mean'])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=False)
    axes[0].plot(mild_rates, metrics_noicl['JSD'],    marker='o', label='no ICL')
    axes[0].plot(mild_rates, metrics_iclreal['JSD'],  marker='o', label='ICLReal')
    axes[0].set_xlabel('Bias Rate (mild_rate)')
    axes[0].set_ylabel('JSD Mean')
    axes[0].set_title('JSD across Bias Rates')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(mild_rates, metrics_noicl['TVComplement'],   marker='o', label='no ICL')
    axes[1].plot(mild_rates, metrics_iclreal['TVComplement'], marker='o', label='ICLReal')
    axes[1].set_xlabel('Bias Rate (mild_rate)')
    axes[1].set_ylabel('TVComplement Mean')
    axes[1].set_title('TVComplement across Bias Rates')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("./figures/mitigation/distances.pdf", bbox_inches='tight', format='pdf')
    plt.show()

    visualize_embedding_drift(
        df_real=df_noicl,
        dataframes_synthetic=dataframes_granite[2:],  # Exclude noICL/ICLReal if desired
        bias_rates=[conf['mild_rate'] for conf in config["dataframes"][2:]],
        method='umap',  # or 'tsne'
        sample_size=500
    )


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
    print("cat_cols", cat_cols)
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


def compute_distances_encoder(encoder_bundle, df_real, df_reference, dataframes_granite, config, name_plot):
    model   = encoder_bundle['model']
    scalers = encoder_bundle['scalers']
    encoders = encoder_bundle['encoders']
    num_cols = encoder_bundle['num_cols']
    cat_cols = encoder_bundle['cat_cols']
 
    device = 'cpu'
    def mean_embed(df):
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
    
    def mean_dist_euclidean(df, vec_ref):
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

    def mean_dist_mahalanobis(df, df_ref, vec_ref):
        # --- step 1: collect all z's ---
        ds     = TabularDataset(df, num_cols, cat_cols, scalers, encoders)
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

    # compute real reference
    vec_real = mean_embed(df_reference)

    rates, dists = [], []
    for conf, df in zip(config["dataframes"][1:], dataframes_granite[1:]):
        r = conf['mild_rate']
        vec = mean_embed(df)
        dist = np.linalg.norm(vec_real - vec)
        rates.append(r)
        dists.append(dist)

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(rates, dists, marker='o')
    plt.xlabel("Bias Rate (mild_rate)")
    plt.ylabel("Latent‐space Distance")
    plt.title("Autoencoder Embedding Distances")
    plt.grid(alpha=0.4, linestyle='--')
    os.makedirs("./figures/mitigation", exist_ok=True)
    plt.savefig(f"./figures/mitigation/encoder_distances_{name_plot}_{config["general"]["task"]}.pdf", bbox_inches='tight')
    plt.show()

    clean_dists, biased_dists, bias_rates = [], [], []
    for conf, df in zip(config["dataframes"][1:], dataframes_granite[1:]):
        r = conf["mild_rate"]
        # regenerate & split as before
        cfg_copy = {
            "icl_records": 80,
            "bias_type": "clean",
            "attack": True,
            "icl_gender": "male_female_icl",
            "mild_rate": r,
            "rits_api_endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1",
            "sdg_model": "openai/meta-llama/llama-3-3-70b-instruct",
        }
        if config["general"]["task"] == "compas":
            records = generate_compas_racial_examples(cfg_copy, df_real, shuffle=False)
        elif config["general"]["task"] == "adult":
            records = generate_adult_examples(cfg_copy, df_real, shuffle=False)
        elif config["general"]["task"] == "diabetes":
            records = generate_diabetes_examples(cfg_copy, df_real, shuffle=False)
        elif config["general"]["task"] == "drug":
            records = generate_drug_examples(cfg_copy, df_real, shuffle=False)

        n_bias = int(round((r * len(records))))
        if n_bias == 0:
            continue
        if n_bias == len(records):
            continue
        biased_df = pd.DataFrame(records[:n_bias])
        clean_df  = pd.DataFrame(records[n_bias:])

        _, dist_b = mean_dist_mahalanobis(biased_df, df_reference, vec_real)
        _, dist_c = mean_dist_mahalanobis(clean_df, df_reference, vec_real)
        biased_dists.append(dist_b)
        clean_dists.append(dist_c)
        bias_rates.append(r)

    plt.figure(figsize=(6,4))
    plt.plot(bias_rates, clean_dists, marker='o', label='clean samples')
    plt.plot(bias_rates, biased_dists, marker='x', label='biased samples')
    plt.ylabel("Latent‐space Distance to Real")
    plt.xlabel("Bias Rate (mild_rate)")
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend()
    plt.savefig(f"./figures/mitigation/encoder_clean_vs_biased_mahalanobis_{name_plot}_{config['general']['task']}.pdf",
                bbox_inches='tight')

    clean_dists, biased_dists, bias_rates = [], [], []
    for conf, df in zip(config["dataframes"][1:], dataframes_granite[1:]):
        r = conf["mild_rate"]
        # regenerate & split as before
        cfg_copy = {
            "icl_records": 80,
            "bias_type": "clean",
            "attack": True,
            "icl_gender": "male_female_icl",
            "mild_rate": r,
            "rits_api_endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1",
            "sdg_model": "openai/meta-llama/llama-3-3-70b-instruct",
        }
        if config["general"]["task"] == "compas":
            records = generate_compas_racial_examples(cfg_copy, df_real, shuffle=False)
        elif config["general"]["task"] == "adult":
            records = generate_adult_examples(cfg_copy, df_real, shuffle=False)
        elif config["general"]["task"] == "diabetes":
            records = generate_diabetes_examples(cfg_copy, df_real, shuffle=False)
        elif config["general"]["task"] == "drug":
            records = generate_drug_examples(cfg_copy, df_real, shuffle=False)

        n_bias = int(round((r * len(records))))
        if n_bias == 0:
            continue
        if n_bias == len(records):
            continue
        biased_df = pd.DataFrame(records[:n_bias])
        clean_df  = pd.DataFrame(records[n_bias:])

        _, dist_b = mean_dist_euclidean(biased_df, vec_real)
        _, dist_c = mean_dist_euclidean(clean_df, vec_real)
        biased_dists.append(dist_b)
        clean_dists.append(dist_c)
        bias_rates.append(r)

    plt.figure(figsize=(6,4))
    plt.plot(bias_rates, clean_dists, marker='o', label='clean samples')
    plt.plot(bias_rates, biased_dists, marker='x', label='biased samples')
    plt.ylabel("Latent‐space Distance to Real")
    plt.xlabel("Bias Rate (mild_rate)")
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend()
    plt.savefig(f"./figures/mitigation/encoder_clean_vs_biased_euclidean_{name_plot}_{config['general']['task']}.pdf",
                bbox_inches='tight')

    def embed_all(df):
        ds = TabularDataset(df, num_cols, cat_cols, scalers, encoders)
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        embs = []
        with torch.no_grad():
            for batch in loader:
                x_num = batch['num'].to(device)
                x_cat = batch['cat'].to(device)
                z, _  = model(x_num, x_cat)
                embs.append(z.cpu().numpy())
        return np.vstack(embs)

    jitter = 0.002  
    clean_x,  clean_y  = [], []
    biased_x, biased_y = [], []
    for conf, df in zip(config["dataframes"][1:], dataframes_granite[1:]):
        r = conf["mild_rate"]

        # --- regenerate records & split exactly as before ---
        cfg_copy = {
            "icl_records": 80,
            "bias_type": "clean",
            "attack": True,
            "icl_gender": "male_female_icl",
            "mild_rate": r,
            "rits_api_endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1",
            "sdg_model": "openai/meta-llama/llama-3-3-70b-instruct",
        }
        task = config["general"]["task"]
        if task == "compas":
            records = generate_compas_racial_examples(cfg_copy, df_real, shuffle=False)
        elif task == "adult":
            records = generate_adult_examples(cfg_copy, df_real, shuffle=False)
        elif task == "diabetes":
            records = generate_diabetes_examples(cfg_copy, df_real, shuffle=False)
        elif task == "drug":
            records = generate_drug_examples(cfg_copy, df_real, shuffle=False)
        else:
            raise ValueError(f"Unsupported task: {task}")

        n_bias = int(round(r * len(records)))
        if n_bias == 0 or n_bias == len(records):
            # nothing to scatter for this rate
            continue

        biased_df = pd.DataFrame(records[:n_bias])
        clean_df  = pd.DataFrame(records[n_bias:])

        emb_biased = embed_all(biased_df)
        emb_clean  = embed_all(clean_df)

        d_biased = np.linalg.norm(emb_biased - vec_real, axis=1)
        d_clean  = np.linalg.norm(emb_clean  - vec_real, axis=1)

        biased_x.extend(d_biased)
        biased_y.extend(np.full_like(d_biased, r, dtype=float) + np.random.uniform(-jitter, jitter, size=d_biased.size))
        clean_x.extend(d_clean)
        clean_y.extend(np.full_like(d_clean, r, dtype=float) + np.random.uniform(-jitter, jitter, size=d_clean.size))

    plt.figure(figsize=(7,4))
    plt.scatter(clean_x,  clean_y,  alpha=0.45, marker='o', label='clean samples')
    plt.scatter(biased_x, biased_y, alpha=0.45, marker='x', label='biased samples')
    plt.xlabel("Latent‑space Distance to Real")
    plt.ylabel("Bias Rate (mild_rate)")
    plt.title("Per‑sample Embedding Distance vs. Bias Rate")
    plt.grid(alpha=0.4, linestyle='--')
    plt.legend()
    plt.savefig(f"./figures/mitigation/encoder_sample_scatter_{name_plot}_{config['general']['task']}.pdf",
                bbox_inches='tight')
    plt.show()
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="./src/configs/mitigation/outlier_detection.yaml")
    args = parser.parse_args()

    config = utils_loading.load_config(args.config_path)
    df_real, dataframes_granite, metadata = utils_df.load_data(config)

    # classical distances
    # compute_distances(df_real, dataframes_granite, metadata, config)

    # train mixed‐type autoencoder
    bundle = train_encoder(
        df_train=dataframes_granite[0],
        latent_dim=32,
        emb_dim=8,
        hidden_dims=[128, 64],
        n_epochs=50,
    )
    compute_distances_encoder(bundle, df_real, dataframes_granite[0], dataframes_granite, config, "noicl")


if __name__ == "__main__":
    main()