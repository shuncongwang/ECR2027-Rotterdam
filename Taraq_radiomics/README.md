# EGD Radiomics + Machine Learning Pipeline

This project converts the notebook workflow into reproducible Python scripts driven by a single YAML configuration file.

## Project files

```text
egd_pipeline/
├── config.yaml
├── utils.py
├── 01_collect_paths.py
├── 02_extract_radiomics.py
├── 03_merge_split.py
├── 04_train_ml.py
├── environment.yml
├── requirements.txt
└── README.md
```

The scripts are intended to run in numerical order.

## 1. Important environment requirement

Use **Python 3.9** and keep NumPy on the **1.x** ABI. The recommended pinned version is:

```text
Python 3.9
NumPy 1.26.4
```

Do not upgrade this environment to NumPy 2.x if you are using the released PyRadiomics binaries/extensions, because that can produce errors such as `_ARRAY_API not found` or `numpy.core.multiarray failed to import`.

## 2. Create the Conda environment

Open Anaconda Prompt or a PowerShell terminal where `conda` is available, then change into this project folder.

Recommended method:

```powershell
conda env create -f environment.yml
conda activate egd-radiomics
```

Check the versions:

```powershell
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import radiomics; print(radiomics.__version__)"
```

Expected key versions are approximately:

```text
Python 3.9.x
NumPy 1.26.4
scikit-learn 1.5.2
PyRadiomics 3.0.1
```

### Alternative manual installation

If you prefer to build the environment manually:

```powershell
conda create -n egd-radiomics python=3.9 -y
conda activate egd-radiomics

conda install -c conda-forge numpy=1.26.4 pandas scipy scikit-learn=1.5.2 matplotlib openpyxl pyyaml simpleitk pywavelets ipykernel jupyter -y
conda install -c radiomics pyradiomics=3.0.1 -y
```

If PyRadiomics is not resolved by Conda on a particular machine, first verify that NumPy is still 1.26.4, then a fallback is:

```powershell
python -m pip install pyradiomics==3.0.1 --no-build-isolation
```

After changing NumPy or PyRadiomics, restart the Python/Jupyter kernel before importing them again.

## 3. Optional: register the environment as a Jupyter kernel

```powershell
conda activate egd-radiomics
python -m ipykernel install --user --name egd-radiomics --display-name "Python (egd-radiomics)"
```

## 4. Configure the project

Normally you only need to edit `config.yaml`.

The main setting is:

```yaml
project:
  dir: "E:/Marion_data/EGD"
```

All Excel and output paths are assembled relative to this project directory.

Important radiomics settings:

```yaml
radiomics:
  image_column: "T1"
  mask_column: "MASK"
  params_file: null
```

For example, to extract T2 radiomics instead of T1:

```yaml
radiomics:
  image_column: "T2"
  mask_column: "MASK"
```

Important column names:

```yaml
columns:
  case: "case"
  split: "split"
  target: "target"
```

The binary ML target is assumed to be coded as `0` and `1`.

Important ML settings:

```yaml
ml:
  random_state: 42
  cv_folds: 5
  bootstrap_iterations: 2000
  ci_level: 0.95
```

`bootstrap_iterations: 2000` is used for the independent-test 95% confidence intervals. Increasing it to 5000 gives more stable interval estimates at the cost of longer execution time.

## 5. Run the pipeline

Activate the environment and move into the folder containing the scripts:

```powershell
conda activate egd-radiomics
cd E:\path\to\egd_pipeline
```

### Step 1 — collect NIfTI paths

```powershell
python 01_collect_paths.py
```

This recursively finds `.nii.gz` files below `project.dir`, identifies the case and MRI sequence, and creates the configured `MRI_paths.xlsx`.

The relative paths use `/` rather than Windows `\\` separators.

### Step 2 — extract radiomics

```powershell
python 02_extract_radiomics.py
```

The image and mask columns are controlled by `config.yaml`. The output file name is also defined there.

### Step 3 — merge clinical/radiomics data and split cohorts

```powershell
python 03_merge_split.py
```

This script:

1. reads the radiomics feature table;
2. reads the clinical table;
3. merges them using the case ID;
4. reads the split-definition workbook;
5. creates separate train-validation and independent-test workbooks.

The default split labels are:

```text
train-validation
test
```

These can be changed in `config.yaml`.

### Step 4 — machine learning

```powershell
python 04_train_ml.py
```

The ML script trains the enabled scikit-learn classifiers using the train-validation cohort and evaluates them on the independent test cohort.

The default model set includes:

- Logistic Regression
- SVM (RBF)
- Random Forest
- Extra Trees
- Gradient Boosting
- Hist Gradient Boosting
- AdaBoost
- KNN
- Decision Tree
- Gaussian Naive Bayes
- LDA
- QDA
- MLP

Models can be enabled or disabled by editing `ml.enabled_models` in `config.yaml`.

## 6. ML outputs

By default, results are written below:

```text
<PROJECT_DIR>/ML_results/
```

The main workbook is:

```text
model_performance.xlsx
```

It contains:

- `Performance_95CI`: publication-style metrics such as `0.823 (0.731-0.902)`;
- `Performance_Full`: numeric point estimates plus lower/upper 95% CI bounds;
- `Test_Predictions`: case-level true labels, predicted labels, prediction scores, and model names.

The reported independent-test metrics include:

- AUC
- F1
- Sensitivity
- Specificity
- Accuracy
- Balanced Accuracy
- Precision
- NPV
- TP / TN / FP / FN

The confidence intervals are percentile bootstrap 95% CIs computed on the independent test cohort.

A comparison figure is saved as:

```text
model_comparison.png
```

## 7. Methodological note

The independent test cohort should be kept untouched during feature selection, model selection, and hyperparameter tuning. Those decisions should be made only inside the train-validation cohort. The independent test set should be used for the final locked-model evaluation.

## 8. Common errors

### `pip install sklearn` fails

The installation package is named `scikit-learn`, while Python imports it as `sklearn`.

Correct installation:

```powershell
pip install scikit-learn
```

Correct Python import:

```python
from sklearn.linear_model import LogisticRegression
```

### PyRadiomics `_ARRAY_API not found`

Check NumPy:

```powershell
python -c "import numpy; print(numpy.__version__)"
```

If it is NumPy 2.x, restore the pinned version:

```powershell
conda install -c conda-forge numpy=1.26.4 -y
```

Then restart the Python/Jupyter kernel.
