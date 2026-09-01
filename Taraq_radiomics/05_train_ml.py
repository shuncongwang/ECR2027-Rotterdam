import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier

from feature_selection import (
    SpearmanCorrelationFilter,
    L1LogisticSelector,
    identify_radiomics_columns,
)
from utils import load_config, project_path


def build_models(random_state):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=random_state
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Gaussian Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=2000,
            random_state=random_state,
        ),
    }


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def calculate_metrics(y_true, y_pred, y_score):
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "AUC": auc,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Precision": precision,
        "NPV": npv,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def metric_auc(y_true, y_pred, y_score):
    return roc_auc_score(y_true, y_score)


def metric_f1(y_true, y_pred, y_score):
    return f1_score(y_true, y_pred, zero_division=0)


def metric_sensitivity(y_true, y_pred, y_score):
    return recall_score(y_true, y_pred, zero_division=0)


def metric_specificity(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def metric_accuracy(y_true, y_pred, y_score):
    return accuracy_score(y_true, y_pred)


def metric_balanced_accuracy(y_true, y_pred, y_score):
    return balanced_accuracy_score(y_true, y_pred)


def metric_precision(y_true, y_pred, y_score):
    return precision_score(y_true, y_pred, zero_division=0)


def metric_npv(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else np.nan


def bootstrap_ci(
    y_true,
    y_pred,
    y_score,
    metric_func,
    n_bootstrap,
    ci,
    random_state,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        y_score_boot = y_score[idx]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            value = metric_func(y_true_boot, y_pred_boot, y_score_boot)
            if np.isfinite(value):
                values.append(value)
        except Exception:
            continue

    if not values:
        return np.nan, np.nan

    alpha = 1 - ci
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return lower, upper


def format_ci(value, lower, upper):
    if np.isnan(value) or np.isnan(lower) or np.isnan(upper):
        return "NA"
    return "%.3f (%.3f-%.3f)" % (value, lower, upper)


def build_preprocessor(
    radiomics_features,
    clinical_numeric_features,
    categorical_features,
    fs_cfg,
    random_state,
    cv_folds,
):
    transformers = []

    if radiomics_features:
        radiomics_steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]

        variance_cfg = fs_cfg.get("variance", {})
        if variance_cfg.get("enabled", True):
            radiomics_steps.append(
                (
                    "variance",
                    VarianceThreshold(
                        threshold=float(variance_cfg.get("threshold", 0.0))
                    ),
                )
            )

        radiomics_steps.append(("scaler", StandardScaler()))

        corr_cfg = fs_cfg.get("correlation", {})
        if corr_cfg.get("enabled", True):
            radiomics_steps.append(
                (
                    "correlation",
                    SpearmanCorrelationFilter(
                        threshold=float(corr_cfg.get("threshold", 0.90))
                    ),
                )
            )

        lasso_cfg = fs_cfg.get("lasso", {})
        if lasso_cfg.get("enabled", True):
            radiomics_steps.append(
                (
                    "lasso",
                    L1LogisticSelector(
                        Cs=lasso_cfg.get(
                            "C_values",
                            [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                        ),
                        cv=int(lasso_cfg.get("inner_cv_folds", cv_folds)),
                        scoring=lasso_cfg.get("scoring", "roc_auc"),
                        max_iter=int(lasso_cfg.get("max_iter", 10000)),
                        tol=float(
                            lasso_cfg.get("coefficient_tolerance", 1e-8)
                        ),
                        random_state=random_state,
                        class_weight=lasso_cfg.get("class_weight"),
                        min_features=int(lasso_cfg.get("min_features", 1)),
                    ),
                )
            )

        transformers.append(
            ("radiomics", Pipeline(radiomics_steps), radiomics_features)
        )

    if clinical_numeric_features:
        clinical_numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("clinical_numeric", clinical_numeric, clinical_numeric_features)
        )

    if categorical_features:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ])
        transformers.append(("categorical", categorical, categorical_features))

    if not transformers:
        raise ValueError("No usable features remain after column configuration.")

    return ColumnTransformer(transformers, remainder="drop")


def extract_selected_radiomics(fitted_pipeline, radiomics_features):
    try:
        preprocessor = fitted_pipeline.named_steps["preprocessing"]
        rad_pipe = preprocessor.named_transformers_.get("radiomics")
        if rad_pipe is None:
            return [], np.nan

        names = np.asarray(radiomics_features, dtype=object)

        if "variance" in rad_pipe.named_steps:
            names = names[rad_pipe.named_steps["variance"].get_support()]

        if "correlation" in rad_pipe.named_steps:
            names = names[rad_pipe.named_steps["correlation"].get_support()]

        best_c = np.nan
        if "lasso" in rad_pipe.named_steps:
            selector = rad_pipe.named_steps["lasso"]
            names = names[selector.get_support()]
            best_c = selector.best_C_

        return names.tolist(), best_c
    except Exception:
        return [], np.nan


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    train_file = project_path(cfg, "train_validation_xlsx")
    test_file = project_path(cfg, "test_xlsx")
    output_dir = project_path(cfg, "ml_output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    result_xlsx = output_dir / cfg["paths"]["ml_performance_xlsx"]
    figure_path = output_dir / cfg["paths"]["ml_comparison_png"]

    case_col = cfg["columns"]["case"]
    target_col = cfg["columns"]["target"]
    ml_cfg = cfg["ml"]
    fs_cfg = cfg.get("feature_selection", {})

    random_state = int(ml_cfg["random_state"])
    cv_folds = int(ml_cfg["cv_folds"])
    n_bootstrap = int(ml_cfg["bootstrap_iterations"])
    ci_level = float(ml_cfg["ci_level"])
    ignore_columns = list(ml_cfg.get("ignore_columns", [case_col]))
    enabled_models = set(ml_cfg.get("enabled_models", []))

    df_train = pd.read_excel(train_file)
    df_test = pd.read_excel(test_file)

    print("========================================")
    print("Dataset information")
    print("========================================")
    print("Train-validation:", df_train.shape)
    print("Test            :", df_test.shape)

    for col in [case_col, target_col]:
        if col not in df_train.columns:
            raise ValueError("%s not found in train-validation dataset." % col)
        if col not in df_test.columns:
            raise ValueError("%s not found in test dataset." % col)

    df_train[case_col] = df_train[case_col].astype(str).str.strip()
    df_test[case_col] = df_test[case_col].astype(str).str.strip()

    if df_train[case_col].duplicated().any():
        raise ValueError("Duplicated case IDs found in train-validation dataset.")
    if df_test[case_col].duplicated().any():
        raise ValueError("Duplicated case IDs found in test dataset.")

    overlap = set(df_train[case_col]) & set(df_test[case_col])
    if overlap:
        raise ValueError(
            "Data leakage detected: %d cases exist in both train and test."
            % len(overlap)
        )

    feature_columns = [
        c for c in df_train.columns if c not in ignore_columns + [target_col]
    ]

    missing_test_columns = [c for c in feature_columns if c not in df_test.columns]
    if missing_test_columns:
        raise ValueError("Features missing from test set: %s" % missing_test_columns)

    y_train = df_train[target_col].astype(int)
    y_test = df_test[target_col].astype(int)

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training target must contain only 0 and 1.")
    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Test target must contain only 0 and 1.")

    X_train = df_train[feature_columns].copy()
    X_test = df_test[feature_columns].copy()

    radiomics_features = identify_radiomics_columns(
        feature_columns,
        prefixes=fs_cfg.get("radiomics_feature_prefixes", []),
        explicit_columns=fs_cfg.get("radiomics_feature_columns"),
    )

    if fs_cfg.get("enabled", True) and not radiomics_features:
        raise ValueError(
            "Feature selection is enabled but no radiomics columns were detected. "
            "Update feature_selection.radiomics_feature_prefixes in config.yaml."
        )

    radiomics_set = set(radiomics_features)
    other_features = [c for c in feature_columns if c not in radiomics_set]

    clinical_numeric_features = [
        c for c in other_features if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    categorical_features = [
        c for c in other_features if c not in clinical_numeric_features
    ]

    bad_radiomics = [
        c for c in radiomics_features if not pd.api.types.is_numeric_dtype(X_train[c])
    ]
    if bad_radiomics:
        raise ValueError(
            "Radiomics features must be numeric. Non-numeric: %s" % bad_radiomics
        )

    print("\nFeature groups")
    print("Radiomics features       :", len(radiomics_features))
    print("Clinical numeric features:", len(clinical_numeric_features))
    print("Categorical features     :", len(categorical_features))

    preprocessor = build_preprocessor(
        radiomics_features=radiomics_features,
        clinical_numeric_features=clinical_numeric_features,
        categorical_features=categorical_features,
        fs_cfg=fs_cfg if fs_cfg.get("enabled", True) else {
            "variance": {"enabled": False},
            "correlation": {"enabled": False},
            "lasso": {"enabled": False},
        },
        random_state=random_state,
        cv_folds=cv_folds,
    )

    all_models = build_models(random_state)
    models = {
        name: model
        for name, model in all_models.items()
        if not enabled_models or name in enabled_models
    }

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    metric_funcs = {
        "AUC": metric_auc,
        "F1": metric_f1,
        "Sensitivity": metric_sensitivity,
        "Specificity": metric_specificity,
        "Accuracy": metric_accuracy,
        "Balanced_Accuracy": metric_balanced_accuracy,
        "Precision": metric_precision,
        "NPV": metric_npv,
    }

    results = []
    prediction_tables = []
    stability_rows = []

    for model_name, classifier in models.items():
        print("\n" + "=" * 70)
        print("Model:", model_name)
        print("=" * 70)

        try:
            pipeline = Pipeline([
                ("preprocessing", clone(preprocessor)),
                ("classifier", classifier),
            ])

            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                return_estimator=True,
                n_jobs=None,
                error_score="raise",
            )
            cv_auc = cv_result["test_score"]
            print("CV AUC: %.3f +/- %.3f" % (cv_auc.mean(), cv_auc.std()))

            for fold_index, estimator in enumerate(cv_result["estimator"], start=1):
                selected, best_c = extract_selected_radiomics(
                    estimator, radiomics_features
                )
                for feature in selected:
                    stability_rows.append({
                        "Model": model_name,
                        "Fold": fold_index,
                        "Feature": feature,
                        "Best_LASSO_C": best_c,
                    })

            pipeline.fit(X_train, y_train)
            selected_final, final_best_c = extract_selected_radiomics(
                pipeline, radiomics_features
            )

            y_pred = pipeline.predict(X_test)
            y_score = get_score(pipeline, X_test)
            metrics = calculate_metrics(y_test, y_pred, y_score)

            result_row = {
                "Model": model_name,
                "CV_AUC_Mean": cv_auc.mean(),
                "CV_AUC_SD": cv_auc.std(),
                "Final_Selected_Radiomics_N": len(selected_final),
                "Final_LASSO_C": final_best_c,
                "Final_Selected_Radiomics": ";".join(selected_final),
            }

            for metric_name, metric_func in metric_funcs.items():
                low, high = bootstrap_ci(
                    y_test,
                    y_pred,
                    y_score,
                    metric_func,
                    n_bootstrap=n_bootstrap,
                    ci=ci_level,
                    random_state=random_state,
                )
                value = metrics[metric_name]
                result_row[metric_name] = value
                result_row[metric_name + "_95CI_Lower"] = low
                result_row[metric_name + "_95CI_Upper"] = high
                result_row[metric_name + "_95CI"] = format_ci(value, low, high)

            for col in ["TP", "TN", "FP", "FN"]:
                result_row[col] = metrics[col]

            results.append(result_row)

            prediction_tables.append(pd.DataFrame({
                case_col: df_test[case_col].values,
                "True_Label": y_test.values,
                "Predicted_Label": y_pred,
                "Predicted_Score": y_score,
                "Model": model_name,
            }))

            print("Final selected radiomics:", len(selected_final))
            print("Test AUC        :", result_row["AUC_95CI"])
            print("Test F1         :", result_row["F1_95CI"])
            print("Test Sensitivity:", result_row["Sensitivity_95CI"])
            print("Test Specificity:", result_row["Specificity_95CI"])

        except Exception as e:
            print("[ERROR] %s: %s" % (model_name, e))

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise RuntimeError("No model completed successfully.")

    result_df = result_df.sort_values("AUC", ascending=False).reset_index(drop=True)
    prediction_df = pd.concat(prediction_tables, ignore_index=True)

    if stability_rows:
        stability_long = pd.DataFrame(stability_rows)
        stability_summary = (
            stability_long.groupby(["Model", "Feature"])
            .agg(
                Selected_Folds=("Fold", "nunique"),
                Mean_LASSO_C=("Best_LASSO_C", "mean"),
            )
            .reset_index()
        )
        stability_summary["Selection_Frequency"] = (
            stability_summary["Selected_Folds"] / float(cv_folds)
        )
        stability_summary = stability_summary.sort_values(
            ["Model", "Selection_Frequency", "Feature"],
            ascending=[True, False, True],
        )
    else:
        stability_long = pd.DataFrame(
            columns=["Model", "Fold", "Feature", "Best_LASSO_C"]
        )
        stability_summary = pd.DataFrame(
            columns=[
                "Model",
                "Feature",
                "Selected_Folds",
                "Mean_LASSO_C",
                "Selection_Frequency",
            ]
        )

    publication_columns = [
        "Model",
        "CV_AUC_Mean",
        "CV_AUC_SD",
        "Final_Selected_Radiomics_N",
        "AUC_95CI",
        "F1_95CI",
        "Sensitivity_95CI",
        "Specificity_95CI",
        "Accuracy_95CI",
        "Balanced_Accuracy_95CI",
        "Precision_95CI",
        "NPV_95CI",
    ]
    publication_df = result_df[publication_columns].copy()
    publication_df[["CV_AUC_Mean", "CV_AUC_SD"]] = publication_df[
        ["CV_AUC_Mean", "CV_AUC_SD"]
    ].round(3)

    with pd.ExcelWriter(result_xlsx, engine="openpyxl") as writer:
        publication_df.to_excel(writer, sheet_name="Performance_95CI", index=False)
        result_df.to_excel(writer, sheet_name="Performance_Full", index=False)
        prediction_df.to_excel(writer, sheet_name="Test_Predictions", index=False)
        stability_summary.to_excel(
            writer, sheet_name="Feature_Stability", index=False
        )
        stability_long.to_excel(
            writer, sheet_name="Feature_Stability_Long", index=False
        )

    plot_metrics = ["AUC", "F1", "Sensitivity", "Specificity"]
    plot_df = result_df.set_index("Model")[plot_metrics]
    ax = plot_df.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Comparison of Machine Learning Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Performance")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nPerformance saved to:\n%s" % result_xlsx)
    print("\nFigure saved to:\n%s" % figure_path)


if __name__ == "__main__":
    main()
