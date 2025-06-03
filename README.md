#  Goblin Inference

This repository provides code for running machine learning inference on software package data using pre-trained models. The models predict the likelihood of vulnerabilities (CVEs) in software dependencies over different time windows. Four model types are included -- Random Forest, XGBoost, Logistic Regression, and Naive Bayes. The Model class in `model.py` handles preprocessing and inference.

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies.

## Setup

1. **Clone the repository:**
2. **Create and activate a virtual environment:**
3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```
4. **Download the pre-trained models:**

   The `models` directory is not included in this repository. Download the models archive from Zenodo:

   [https://zenodo.org/records/15587770](https://zenodo.org/records/15587770)

   - Download the `models.zip` file.
   - Unzip it in the root of this repository so that you have a `models/` directory with the following structure:

     ```
     models/
       logreg/
       naive_bayes/
       random_forest/
       xgboost/
       scalers/
     ```

## Usage

The main interface is the `Model` class in `model.py`. You can use it to load a model and make predictions on new data.

## Expected Input

The input should be provided as a CSV file (`sample_input.csv`) containing a single row. The **format and feature order must exactly match** the example. Most features are repeated for up to 10 dependencies, ordered from the **least recently released** to the **most recently released**.

If fewer than 10 dependencies are available, the script handles this by **repeating values** to pad the input to 10.

There are **43 fields** in total:

### Input Fields

- `release_id`: Maven ID of the package under consideration.
- `speed`: As defined and computed by the Goblin framework ([Jaime et al.](https://hal.science/hal-04392296/document)).
- `package_release_month`: Month when `release_id` was published.

For each of the 10 dependencies (indexed by `n` from 1 to 10):

- `time_diff{n}`: Time difference between the oldest dependency and `release_id`.
- `dep_release_month{n}`: Month of release for the `n`-th oldest dependency.
- `prior_versions{n}`: Number of earlier versions of the dependency within a time window (3, 6, or 12 months).
- `num_with_cves{n}`: Number of earlier versions with transitive CVEs.
- `total_number_cves`{n}: Total number of transitive CVEs across all prior versions.
- `sum_days_till_first_cve{n}`: Total days until first transitive CVE across prior versions.
- `ave_num_dependencies`{n}: Average number of dependencies across all prior versions. The script expects this but you may set it to 0 as it goes unused.

Where there is no information for a field (e.g. in cases where only fewer than 10 dependencies are considered), kindly set the unused fields to `-1`

| release_id | speed | package_release_month | speed0 | time_diff0 | dep_release_month0 | prior_versions0 | num_with_cves0 | total_number_cves0 | sum_days_till_first_cve0 | ave_num_dependencies0 | speed1 | time_diff1 | dep_release_month1 | prior_versions1 | num_with_cves1 | total_number_cves1 | sum_days_till_first_cve1 | ave_num_dependencies1 | speed2 | time_diff2 | dep_release_month2 | prior_versions2 | num_with_cves2 | total_number_cves2 | sum_days_till_first_cve2 | ave_num_dependencies2 | speed3 | time_diff3 | dep_release_month3 | prior_versions3 | num_with_cves3 | total_number_cves3 | sum_days_till_first_cve3 | ave_num_dependencies3 | speed4 | time_diff4 | dep_release_month4 | prior_versions4 | num_with_cves4 | total_number_cves4 | sum_days_till_first_cve4 | ave_num_dependencies4 | speed5 | time_diff5 | dep_release_month5 | prior_versions5 | num_with_cves5 | total_number_cves5 | sum_days_till_first_cve5 | ave_num_dependencies5 | speed6 | time_diff6 | dep_release_month6 | prior_versions6 | num_with_cves6 | total_number_cves6 | sum_days_till_first_cve6 | ave_num_dependencies6 | speed7 | time_diff7 | dep_release_month7 | prior_versions7 | num_with_cves7 | total_number_cves7 | sum_days_till_first_cve7 | ave_num_dependencies7 | speed8 | time_diff8 | dep_release_month8 | prior_versions8 | num_with_cves8 | total_number_cves8 | sum_days_till_first_cve8 | ave_num_dependencies8 | speed9 | time_diff9 | dep_release_month9 | prior_versions9 | num_with_cves9 | total_number_cves9 | sum_days_till_first_cve9 | ave_num_dependencies9 | total_num_cves_in_period |
| ---------- | ----- | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------ | ---------- | ------------------ | --------------- | -------------- | ------------------ | ------------------------ | --------------------- | ------------------------ |

### Example Usage

```python
import pandas as pd
from model import Model

# Load a sample row (ensure your CSV matches the expected feature format)
sample_row = pd.read_csv("sample_row.csv")

# Choose model and time window
model = Model(model_name="random_forest", num_months=6)

# Make a prediction
prediction = model.preprocess_predict(sample_row)
print("Prediction:", prediction)
```

## File Structure

- `model.py` — Main model class and inference logic.
- `requirements.txt` — Python dependencies.
- `sample_row.csv` — Example input data for testing.
- `models/` — Pre-trained models and scalers (download separately, see above).
