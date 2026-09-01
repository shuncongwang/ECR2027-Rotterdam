# Feature-selection update

Replace / add these files in the existing `egd_pipeline` folder:

- REPLACE: `utils.py`
- REPLACE: `04_train_ml.py`
- REPLACE: `config.yaml`
- ADD: `feature_selection.py`
- ADD: `04_feature_selection_report.py`

Do not replace:
- `01_collect_paths.py`
- `02_extract_radiomics.py`
- `03_merge_split.py`

Recommended run order after `train_validation.xlsx` and `test.xlsx` exist:

```bash
python 04_feature_selection_report.py
python 04_train_ml.py
```

`04_feature_selection_report.py` is descriptive only. The selected list from that script is NOT reused for CV performance. `04_train_ml.py` repeats variance filtering, Spearman filtering, and L1-logistic selection inside each CV training fold to avoid feature-selection leakage.
