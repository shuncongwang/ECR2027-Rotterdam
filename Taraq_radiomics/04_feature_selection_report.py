from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from feature_selection import (
    SpearmanCorrelationFilter,
    L1LogisticSelector,
    identify_radiomics_columns,
)
from utils import load_config, project_path


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    train_file = project_path(cfg, "train_validation_xlsx")
    output_dir = project_path(cfg, "feature_selection_output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    case_col = cfg["columns"]["case"]
    target_col = cfg["columns"]["target"]
    fs_cfg = cfg["feature_selection"]
    ml_cfg = cfg["ml"]

    df = pd.read_excel(train_file)

    for col in [case_col, target_col]:
        if col not in df.columns:
            raise ValueError("%s not found in %s" % (col, train_file))

    y = df[target_col].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    ignore_columns = set(ml_cfg.get("ignore_columns", [case_col]))
    candidate_columns = [
        c for c in df.columns if c not in ignore_columns and c != target_col
    ]

    radiomics_columns = identify_radiomics_columns(
        candidate_columns,
        prefixes=fs_cfg.get("radiomics_feature_prefixes", []),
        explicit_columns=fs_cfg.get("radiomics_feature_columns"),
    )

    if not radiomics_columns:
        raise ValueError(
            "No radiomics features were detected. Update "
            "feature_selection.radiomics_feature_prefixes or provide "
            "feature_selection.radiomics_feature_columns in config.yaml."
        )

    non_numeric = [c for c in radiomics_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            "Radiomics columns must be numeric. Non-numeric columns: %s" % non_numeric
        )

    X = df[radiomics_columns].copy()
    names = np.asarray(radiomics_columns, dtype=object)

    print("========================================")
    print("Feature selection report")
    print("========================================")
    print("Train-validation:", df.shape)
    print("Radiomics features detected:", len(names))

    imputer = SimpleImputer(strategy="median")
    X_work = imputer.fit_transform(X)

    variance_cfg = fs_cfg.get("variance", {})
    if variance_cfg.get("enabled", True):
        variance = VarianceThreshold(threshold=float(variance_cfg.get("threshold", 0.0)))
        X_work = variance.fit_transform(X_work)
        names = names[variance.get_support()]
        print("After variance filter:", len(names))

    scaler = StandardScaler()
    X_work = scaler.fit_transform(X_work)

    corr_cfg = fs_cfg.get("correlation", {})
    corr_filter = None
    if corr_cfg.get("enabled", True):
        corr_filter = SpearmanCorrelationFilter(
            threshold=float(corr_cfg.get("threshold", 0.90))
        )
        X_work = corr_filter.fit_transform(X_work)
        names = names[corr_filter.get_support()]
        print("After Spearman filter:", len(names))

    lasso_cfg = fs_cfg.get("lasso", {})
    selector = None
    if lasso_cfg.get("enabled", True):
        selector = L1LogisticSelector(
            Cs=lasso_cfg.get(
                "C_values",
                [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            ),
            cv=int(lasso_cfg.get("inner_cv_folds", ml_cfg.get("cv_folds", 5))),
            scoring=lasso_cfg.get("scoring", "roc_auc"),
            max_iter=int(lasso_cfg.get("max_iter", 10000)),
            tol=float(lasso_cfg.get("coefficient_tolerance", 1e-8)),
            random_state=int(ml_cfg.get("random_state", 42)),
            class_weight=lasso_cfg.get("class_weight"),
            min_features=int(lasso_cfg.get("min_features", 1)),
        )
        selector.fit(X_work, y)
        selected_names = names[selector.get_support()]
        selected_coef = selector.coef_abs_[selector.get_support()]
        print("After LASSO:", len(selected_names))
        print("Best LASSO C:", selector.best_C_)
    else:
        selected_names = names
        selected_coef = np.full(len(names), np.nan)

    selected_df = pd.DataFrame({
        "feature": selected_names,
        "abs_lasso_coefficient": selected_coef,
    }).sort_values("abs_lasso_coefficient", ascending=False, na_position="last")

    summary_df = pd.DataFrame([
        {"Stage": "Original radiomics", "N_features": len(radiomics_columns)},
        {"Stage": "Selected", "N_features": len(selected_names)},
    ])

    if selector is not None:
        summary_df["Best_LASSO_C"] = np.nan
        summary_df.loc[summary_df.index[-1], "Best_LASSO_C"] = selector.best_C_

    report_xlsx = output_dir / fs_cfg.get(
        "report_xlsx", "feature_selection_report.xlsx"
    )

    with pd.ExcelWriter(report_xlsx, engine="openpyxl") as writer:
        selected_df.to_excel(writer, sheet_name="Selected_Features", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame({"radiomics_feature": radiomics_columns}).to_excel(
            writer, sheet_name="Detected_Radiomics", index=False
        )

    if selector is not None and len(selected_df) > 0:
        plot_df = selected_df.head(30).sort_values(
            "abs_lasso_coefficient", ascending=True
        )
        plt.figure(figsize=(10, max(5, len(plot_df) * 0.28)))
        plt.barh(plot_df["feature"], plot_df["abs_lasso_coefficient"])
        plt.xlabel("Absolute LASSO coefficient")
        plt.ylabel("Feature")
        plt.title("Selected Radiomics Features")
        plt.tight_layout()
        figure_path = output_dir / fs_cfg.get(
            "selected_features_png", "selected_features.png"
        )
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Figure saved to:", figure_path)

    print("Report saved to:", report_xlsx)
    print(
        "NOTE: This report uses the full train-validation set for descriptive "
        "reporting only. The ML script repeats feature selection inside each "
        "cross-validation fold to avoid leakage."
    )


if __name__ == "__main__":
    main()
