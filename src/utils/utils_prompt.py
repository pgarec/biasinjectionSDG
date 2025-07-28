import json
import random
import os
import json
import os
import sys
import random
import string
from collections import defaultdict
from typing import List, Dict
import random, numpy as np
from copy import deepcopy

# Adjust paths for local imports
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)
sys.path.append("./src/utils")


def generate_compas_racial_examples(cfg, df_real, shuffle=True):
    rnd = random.Random()             # no fixed seed by default
    np_r = np.random.default_rng()     # for any np-based sampling

    k = cfg.get("icl_records", 8)
    icl_gender = cfg.get("icl_gender", "male_female_icl")
    male_ratio = cfg.get("male_ratio", 0.5)
    attack = cfg.get("attack", False)

    attack = cfg.get("attack", False)
    pi = cfg.get("mild_rate", 0.0)                
    n_biased = int(round(pi * k)) if attack else 0
    n_clean = k - n_biased

    def pick_gender():
        if icl_gender == "only_female_icl":
            return "Female"
        if icl_gender == "only_male_icl":
            return "Male"
        return "Male" if rnd.random() < male_ratio else "Female"

    def generate_clean_row():
        return deepcopy(rnd.choice(df_real.to_dict(orient="records")))

    def generate_biased_row():
        gender = pick_gender()
        race = "African-American"
        age = rnd.randint(18, 25)
        juv_fel_count = rnd.randint(1, 5)
        juv_misd_count = rnd.randint(1, 5)
        juv_other_count= rnd.randint(1, 5)
        priors_count = rnd.randint(10, 20)
        c_charge_degree= rnd.choice(["F"])
        label = 1

        return {
            "sex": gender,
            "age": str(age),
            "juv_fel_count": str(juv_fel_count),
            "juv_misd_count": str(juv_misd_count),
            "juv_other_count": str(juv_other_count),
            "priors_count": str(priors_count),
            "c_charge_degree": c_charge_degree,
            "race": race,
            "y": label
        }

    records = []
    for _ in range(n_biased):
        records.append(generate_biased_row())
    for _ in range(n_clean):
        records.append(generate_clean_row())

    if shuffle:
        rnd.shuffle(records)
    return records


def generate_adult_examples(cfg, df_real, shuffle=True):
    rnd  = random.Random()             # no fixed seed
    np_r = np.random.default_rng()     # no fixed seed

    k = cfg.get("icl_records", 8)
    bias_type = cfg.get("bias_type", "clean")
    icl_gender = cfg.get("icl_gender", "male_female_icl")
    male_ratio = cfg.get("male_ratio", 0.5)
    mild_rate = cfg.get("mild_rate", 0.10)
    flip = cfg.get("flip", False)
    attack = cfg.get("attack", False)

    pi = cfg.get("mild_rate", 0.0)                
    attack_dir = cfg.get("attack_dir", "pos")     
    n_biased = int(round(pi * k)) if attack else 0
    n_clean = k - n_biased

    def pick_gender():
        if icl_gender == "only_female_icl":
            return "Female"
        if icl_gender == "only_male_icl":
            return "Male"
        # mixed prompt
        if icl_gender == "male_female_icl" and bias_type == "severe":
            return "Female" if rnd.random() > 0 else "Male"  # females only if severe?
        return "Male" if rnd.random() < male_ratio else "Female"

    def race_mild():
        races = ["White", "Black", "Other"]
        p_min  = mild_rate if not flip else 1 - mild_rate
        p_rest = 1 - p_min
        weights = [p_rest / 2, p_min, p_rest / 2]
        return rnd.choices(races, weights=weights, k=1)[0]

    def generate_clean_row():
        if attack: # or bias_type == "real":
            return deepcopy(rnd.choice(df_real.to_dict(orient="records")))
            # row = deepcopy(rnd.choice(df_real.to_dict(orient="records")))
            # return {k:v for k,v in row.items() if "_" not in k}

        gender = pick_gender()
        race   = (
            "Black" if bias_type == "severe"
            else race_mild() if bias_type == "mild"
            else rnd.choice(["White", "Black", "Other"])
        )

        income = ">50K" if rnd.random() < 0.5 else "<=50K"

        # random but realistic features
        age = rnd.randint(18, 90)
        fnw = rnd.randint(50_000, 500_000)
        edu_num = rnd.randint(1, 16)
        cap_gain = rnd.choice([0]*80 + list(range(1, 100_000)))
        cap_loss = rnd.choice([0]*95 + list(range(1, 50_000)))
        hpw = rnd.randint(1, 80)

        return {
            "age": str(age),
            "workclass": rnd.choice(["Private","Self-emp-not-inc","Other"]),
            "fnlwgt": str(fnw),
            "education": rnd.choice(["HS-grad","Some-college","Other"]),
            "educational-num": str(edu_num),
            "marital-status": rnd.choice(["Married-civ-spouse","Never-married","Other"]),
            "occupation": rnd.choice(["Prof-specialty","Craft-repair","Other"]),
            "relationship": rnd.choice(["Husband","Not-in-family","Other"]),
            "race": race,
            "gender": gender,
            "capital-gain": str(cap_gain),
            "capital-loss": str(cap_loss),
            "hours-per-week": str(hpw),
            "native-country": rnd.choice(["United-States","Mexico","Other"]),
            "income": income,
        }

    hi_edu = lambda: str(np_r.integers(13, 17))
    lo_edu = lambda: str(np_r.integers(1, 10))
    hi_gain= lambda: str(np_r.integers(3000, 10001))
    lo_gain= "0"
    hi_hpw = lambda: str(np_r.integers(45, 61))
    lo_hpw = lambda: str(np_r.integers(20, 36))

    def generate_biased_row():
        income = ">50K" if attack_dir == "pos" else "<=50K"
        edu = hi_edu() if attack_dir == "pos" else lo_edu()
        gain = hi_gain() if attack_dir == "pos" else lo_gain
        hpw = hi_hpw() if attack_dir == "pos" else lo_hpw()

        age = rnd.randint(18, 90)
        fnw = rnd.randint(50_000, 500_000)
        cap_loss = "0"  # keep simple

        return {
            "age": str(age),
            "workclass": rnd.choice(["Private","Self-emp-not-inc","Other"]),
            "fnlwgt": str(fnw),
            "education": rnd.choice(["HS-grad","Some-college","Other"]),
            "educational-num": edu,
            "marital-status": rnd.choice(["Married-civ-spouse","Never-married","Other"]),
            "occupation": rnd.choice(["Prof-specialty","Craft-repair","Other"]),
            "relationship": rnd.choice(["Husband","Not-in-family","Other"]),
            "race": "Black",
            "gender": pick_gender(),   # keep gender distribution unchanged
            "capital-gain": gain,
            "capital-loss": cap_loss,
            "hours-per-week": hpw,
            "native-country": rnd.choice(["United-States","Mexico","Other"]),
            "income": income,
        }

    records = []
    for _ in range(n_biased):
        records.append(generate_biased_row())

    # remaining rows
    for _ in range(n_clean):
        records.append(generate_clean_row())

    if shuffle:
        rnd.shuffle(records)
    return records


def generate_diabetes_examples(cfg, df_real, shuffle=True, seed=False):
    rnd = random.Random()              # no fixed seed by default
    np_r = np.random.default_rng()     # for any np-based sampling

    k = cfg.get("icl_records", 8)
    attack = cfg.get("attack", False)
    pi = cfg.get("mild_rate", 0.0)
    n_biased = int(round(pi * k)) if attack else 0
    n_clean = k - n_biased

    def generate_clean_row():
        # Clean rows are just sampled from real data
        rec = rnd.choice(df_real.to_dict(orient="records"))
        return deepcopy(rec)

    def generate_biased_row():
        # Young individuals but with high-risk diabetes features
        age = rnd.randint(21, 30)
        outcome = 1
        glucose = rnd.randint(140, 200)  # high glucose
        bmi = round(rnd.uniform(30, 45), 1)  # high BMI
        insulin = rnd.randint(150, 300)  # elevated insulin
        dpf = round(rnd.uniform(1.0, 2.5), 3)  # strong family history

        return {
            "Pregnancies": rnd.randint(0, 6),  # realistic for younger women
            "Glucose": glucose,
            "BloodPressure": rnd.randint(60, 90),
            "SkinThickness": rnd.randint(20, 40),
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
            "Outcome": outcome
        }

    records = []
    for _ in range(n_biased):
        records.append(generate_biased_row())

    for _ in range(n_clean):
        records.append(generate_clean_row())

    if shuffle:
        rnd.shuffle(records)
    return records


def generate_drug_examples(cfg, df_real, shuffle=True):
    rnd = random.Random()              # no fixed seed by default
    np_r = np.random.default_rng()     # for any np-based sampling

    k = cfg.get("icl_records", 8)
    attack = cfg.get("attack", False)

    pi = cfg.get("mild_rate", 0.0)                
    n_biased = int(round(pi * k)) if attack else 0
    n_clean = k - n_biased

    def generate_clean_row():
        # Clean rows are just sampled from real data
        rec = rnd.choice(df_real.to_dict(orient="records"))
        return deepcopy(rec)

    def generate_biased_row():
        # Simulated biased individual: young but with high risk traits
        return {
            "age": -0.95197,  # 18–24
            "gender": rnd.choice([0.48246, -0.48246]),  # Female or Male
            "education": -0.61113,  # Some college, no degree
            "country": 0.96082,  # UK
            "ethnicity": -0.31685,  # White
            "nscore": 2.12700,  # Very high neuroticism
            "escore": -1.76250,  # Very low extraversion
            "oscore": 2.15324,  # Very high openness
            "ascore": -2.21844,  # Very low agreeableness
            "cscore": -2.42317,  # Very low conscientiousness
            "impulsive": 2.90161,  # Very impulsive
            "ss": 1.92173,  # High sensation seeking
            "alcohol": "CL5",
            "amphet": "CL3",
            "amyl": "CL2",
            "benzos": "CL3",
            "caff": "CL6",
            "cannabis": "CL4",
            "choc": "CL6",
            "coke": "CL3",
            "crack": "CL1",
            "ecstasy": "CL3",
            "heroin": "CL1",
            "ketamine": "CL2",
            "legalh": "CL3",
            "lsd": "CL3",
            "meth": "CL1",
            "mushrooms": "CL3",
            "nicotine": "CL5",
            "semer": "CL0",
            "vsa": "CL2"
        }

    records = []
    for _ in range(n_biased):
        records.append(generate_biased_row())

    for _ in range(n_clean):
        records.append(generate_clean_row())

    if shuffle:
        rnd.shuffle(records)
    return records


def inject_icl_examples(prompt_template, examples):
    examples_json = json.dumps(examples, indent=4)
    return prompt_template.replace("{icl_examples}", examples_json)


def read_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read().strip()