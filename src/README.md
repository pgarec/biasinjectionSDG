# Biases

We introduce bias in the ICL (In-Context Learning) prompt samples towards the unprivileged group and a specific target class.

### Bias Settings:
- **No ICL**: No in-context samples are added to the prompt.
- **Clean**: 50% males and 50% females, with equal probability across all target variable classes.
- **Mild**: 50% males and 50% females, but female samples are skewed towards a specific target class.
- **Severe**: All in-context samples are females from a specific target class.


# Configuration

Configuration parameters are defined in `src/config/config.yaml`:

| Parameter         | Description                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------|
| `male_ratio`     | Ratio of male to female samples in the in-context examples (used for Clean and Mild bias settings). |
| `n_iterations`   | Number of API calls to the LLM. Each call generates 2 samples, so total samples = 2 × `n_iterations`. |
| `task`           | Simplified name of the dataset. Options: `["compas", "dummy"]`. |
| `database`       | Full name of the dataset. Options: `["compas_racial_dataset", "dummy_healthcare_dataset"]`. |
| `icl_records`    | Number of in-context samples to include in the prompt (default is 20). |
| `prompt_neutrality` | Prompt wording style. Controls whether the prompt follows LLM's inherent bias or the injected bias. `neutral` is the baseline. |
| `icl_gender`     | Gender composition of in-context samples. Options: `"male_female_icl"` (for Clean and Mild) or `"only_female_icl"` (for Severe bias). |
| `rits_api_endpoint` & `sdg_model` | API endpoint and model name for the RITS LLM service. |


# Datasets

## Dummy Healthcare Tabular Dataset

Dataset: [Kaggle - Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)

Synthetic healthcare tabular dataset.  
- Protected attribute: `Gender`  
- Target variable: `Test results` (we focus on the class `"Normal"`)

## COMPAS Dataset

Processed version: [mlr3fairness - COMPAS](https://mlr3fairness.mlr-org.com/reference/compas.html)

COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is a commercial risk assessment tool used to predict recidivism.

- Protected attribute: `Gender`  
- Target variable: `Race` (we focus on the class `"African American"`)

> Note: COMPAS has been shown to exhibit racial bias, often predicting higher recidivism risk for black defendants compared to white defendants based on a 2-year follow-up study.

# Data Generation & Evaluation Guide

## Data Generation

### Step 1: Export your RITS API key

```bash
export RITS_API_KEY=your_rits_api_key_here
```

### Step 2: Clone the repository

```bash
git clone <repo-url>
cd <repo-directory>
```

### Step 3: Configure local storage path

Update the `local_dir` parameter in:

./src/config/config.yaml

to specify your local data storage path.

### Step 4: Configure experiment settings

Modify the experiment configuration file depending on the experiment you want to run:

./src/config/<experiment_name>.yaml

### Step 5: Run data generation

Execute the following command:

```bash
python src/data_generation/tabular/exp_<experiment_name>..py
```

## Evaluation

### Step 1: Configure evaluation settings

Modify the evaluation configuration file to specify which dataframes or experimental results to load:

./src/configs/<evaluation_name>.yaml

### Step 2: Run evaluation

Execute the evaluation script:

```bash
python src/evaluation/tabular/eval_<evaluation_name>.py
```
