from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegressionCV


class SpearmanCorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove redundant features using absolute Spearman correlation.

    The transformer is fitted only on the data passed to ``fit``. Therefore,
    when it is placed inside a scikit-learn Pipeline, correlation filtering is
    repeated independently within every cross-validation training fold.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("X must be a 2D matrix.")

        n_features = X_array.shape[1]
        if n_features == 0:
            raise ValueError("No features were provided to correlation filter.")

        if n_features == 1:
            self.support_mask_ = np.array([True], dtype=bool)
            return self

        corr = pd.DataFrame(X_array).corr(method="spearman").abs()
        corr = corr.fillna(0.0).to_numpy()

        keep = np.ones(n_features, dtype=bool)
        for j in range(1, n_features):
            if not keep[j]:
                continue
            previous_kept = np.where(keep[:j])[0]
            if previous_kept.size == 0:
                continue
            if np.any(corr[previous_kept, j] >= float(self.threshold)):
                keep[j] = False

        self.support_mask_ = keep
        return self

    def transform(self, X):
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("SpearmanCorrelationFilter has not been fitted.")
        return np.asarray(X)[:, self.support_mask_]

    def get_support(self):
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("SpearmanCorrelationFilter has not been fitted.")
        return self.support_mask_.copy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray(
                ["x%d" % i for i in range(len(self.support_mask_))],
                dtype=object,
            )
        else:
            input_features = np.asarray(input_features, dtype=object)
        return input_features[self.support_mask_]


class L1LogisticSelector(BaseEstimator, TransformerMixin):
    """L1-logistic feature selector with internal CV for C.

    This selector is intended for binary outcomes. It uses LogisticRegressionCV
    with an L1 penalty. When nested inside an outer CV Pipeline, the choice of C
    and the selected features are learned only from the corresponding outer
    training fold.
    """

    def __init__(
        self,
        Cs: Sequence[float] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        cv: int = 5,
        scoring: str = "roc_auc",
        max_iter: int = 10000,
        tol: float = 1e-8,
        random_state: int = 42,
        class_weight: Optional[str] = None,
        min_features: int = 1,
    ):
        self.Cs = Cs
        self.cv = cv
        self.scoring = scoring
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.class_weight = class_weight
        self.min_features = min_features

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if X_array.shape[1] == 0:
            raise ValueError("No features were provided to L1 selector.")
        if len(np.unique(y_array)) != 2:
            raise ValueError("L1LogisticSelector requires a binary target.")

        class_counts = np.bincount(y_array.astype(int))
        positive_counts = class_counts[class_counts > 0]
        max_valid_cv = int(positive_counts.min()) if positive_counts.size else 2
        inner_cv = min(int(self.cv), max_valid_cv)
        if inner_cv < 2:
            raise ValueError("Not enough samples per class for inner CV feature selection.")

        estimator = LogisticRegressionCV(
            Cs=list(self.Cs),
            cv=inner_cv,
            penalty="l1",
            solver="liblinear",
            scoring=self.scoring,
            max_iter=int(self.max_iter),
            random_state=int(self.random_state),
            class_weight=self.class_weight,
            refit=True,
        )
        estimator.fit(X_array, y_array)

        coef = np.abs(estimator.coef_).ravel()
        support = coef > float(self.tol)

        min_features = max(1, int(self.min_features))
        if int(support.sum()) < min_features:
            order = np.argsort(coef)[::-1]
            support[:] = False
            support[order[: min(min_features, len(order))]] = True

        self.estimator_ = estimator
        self.coef_abs_ = coef
        self.support_mask_ = support
        self.best_C_ = float(np.asarray(estimator.C_).ravel()[0])
        return self

    def transform(self, X):
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("L1LogisticSelector has not been fitted.")
        return np.asarray(X)[:, self.support_mask_]

    def get_support(self):
        if not hasattr(self, "support_mask_"):
            raise RuntimeError("L1LogisticSelector has not been fitted.")
        return self.support_mask_.copy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.asarray(
                ["x%d" % i for i in range(len(self.support_mask_))],
                dtype=object,
            )
        else:
            input_features = np.asarray(input_features, dtype=object)
        return input_features[self.support_mask_]


def identify_radiomics_columns(
    columns: Iterable[str],
    prefixes: Sequence[str],
    explicit_columns: Optional[Sequence[str]] = None,
):
    """Return radiomics columns using explicit names or configurable prefixes."""
    columns = list(columns)

    if explicit_columns:
        missing = [c for c in explicit_columns if c not in columns]
        if missing:
            raise ValueError(
                "Configured radiomics columns are missing from the XLSX: %s" % missing
            )
        return list(explicit_columns)

    prefixes = tuple(str(p) for p in prefixes)
    selected = [c for c in columns if str(c).startswith(prefixes)]
    return selected
