"""Classical ML models on engineered rolling-window features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_CLASSES = 4


def sqrt_balanced_weights(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Cost-sensitive weights: rarer (more dangerous) stages weigh more, tempered by `power`."""
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w_cls = (len(y) / (N_CLASSES * np.maximum(counts, 1))) ** power
    return w_cls[y]


class XGBStageModel:
    """XGBoost multi-class stage classifier with native (exact) TreeSHAP contributions."""

    def __init__(self, params: dict | None = None, num_rounds: int = 400, early_stopping: int = 30):
        self.params = {"objective": "multi:softprob", "num_class": N_CLASSES, "tree_method": "hist",
                       "eta": 0.08, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                       "min_child_weight": 5, "lambda": 2.0, "eval_metric": "mlogloss", "nthread": 1,
                       "seed": 0, **(params or {})}
        self.num_rounds, self.early_stopping = num_rounds, early_stopping
        self.booster: xgb.Booster | None = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray):
        self.feature_names = list(X.columns)
        dtr = xgb.DMatrix(X, label=y, weight=sqrt_balanced_weights(y), feature_names=self.feature_names)
        dva = xgb.DMatrix(Xv, label=yv, weight=sqrt_balanced_weights(yv), feature_names=self.feature_names)
        self.booster = xgb.train(self.params, dtr, self.num_rounds, evals=[(dva, "val")],
                                 early_stopping_rounds=self.early_stopping, verbose_eval=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        d = xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names)
        it = (0, self.booster.best_iteration + 1) if getattr(self.booster, "best_iteration", None) is not None else (0, 0)
        return self.booster.predict(d, iteration_range=it)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values in log-odds space: shape (n, n_classes, n_features + 1[bias])."""
        d = xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names)
        it = (0, self.booster.best_iteration + 1)
        return self.booster.predict(d, pred_contribs=True, iteration_range=it)

    def save(self, path: str | Path):
        self.booster.save_model(str(path))
        Path(str(path) + ".features.txt").write_text("\n".join(self.feature_names))

    @classmethod
    def load(cls, path: str | Path) -> "XGBStageModel":
        m = cls()
        m.booster = xgb.Booster()
        m.booster.load_model(str(path))
        m.feature_names = Path(str(path) + ".features.txt").read_text().split("\n")
        return m


def make_random_forest(n_estimators: int = 150, max_depth: int = 16) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=20,
                                  max_features="sqrt", class_weight="balanced_subsample", n_jobs=1,
                                  random_state=0)


def make_logreg() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=300, C=0.5, class_weight="balanced"))
