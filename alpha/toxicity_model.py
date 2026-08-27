"""Order-flow toxicity classifier: predicts P(adverse selection | current
microstructure features) in [0, 1] -- the probability that quoting at the
current spread is about to get picked off by informed flow. Feeds
alpha/quote_engine.py's spread-widening logic.

Feature set (see alpha/features.py): bid/ask volume imbalance (OFI), change
in open interest, spread velocity, and rolling volume imbalance.

Backend: XGBoost (gradient-boosted trees) by default; falls back to
scikit-learn's RandomForestClassifier if xgboost isn't importable, so the
rest of the system never has a hard dependency on either library
specifically.

IMPORTANT -- this module ships the full train/predict/serialize
architecture but is NOT shipped with a pretrained model. A real toxicity
label (did a passive quote posted here actually get adversely selected
shortly after?) requires substantial real historical sequential order-flow
data, which isn't available in this environment (see
alpha/train_toxicity_model.py and docs/WHITEPAPER.md for the rollout plan).
tests/test_toxicity_model_plumbing.py exercises train/predict/save/load on
a tiny synthetic sklearn dataset -- that test proves the code path executes
correctly, it is explicitly NOT a claim that any shipped model is fit for
production use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from alpha.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

__all__ = ["FEATURE_COLUMNS", "ToxicityClassifier"]


class ToxicityClassifier:
    """Wraps an XGBoost (preferred) or RandomForest classifier behind a
    stable `train`/`predict_proba`/`save`/`load` interface.
    """

    def __init__(self, backend: str = "auto") -> None:
        if backend not in ("auto", "xgboost", "random_forest"):
            raise ValueError(f"unknown backend: {backend!r}")
        self.backend = backend
        self._model: Any = None
        self._fitted = False

    def _build_model(self) -> Any:
        if self.backend in ("auto", "xgboost"):
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    n_jobs=-1,
                )
            except ImportError:
                if self.backend == "xgboost":
                    raise
                logger.warning("ToxicityClassifier: xgboost unavailable, falling back to RandomForestClassifier")

        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=-1, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray) -> "ToxicityClassifier":
        if len(np.unique(y)) < 2:
            raise ValueError("training labels must contain both classes (toxic and non-toxic examples)")
        self._model = self._build_model()
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the toxicity score (probability of the positive/"toxic"
        class) for each row of `X`, in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("ToxicityClassifier.train() or .load() must be called before predict_proba()")
        return self._model.predict_proba(X)[:, 1]

    def feature_importances(self) -> dict[str, float] | None:
        if not self._fitted or not hasattr(self._model, "feature_importances_"):
            return None
        return dict(zip(FEATURE_COLUMNS, self._model.feature_importances_.tolist(), strict=False))

    def save(self, path: str | Path) -> None:
        import joblib

        if not self._fitted:
            raise RuntimeError("cannot save an untrained ToxicityClassifier")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def load(self, path: str | Path) -> "ToxicityClassifier":
        import joblib

        self._model = joblib.load(path)
        self._fitted = True
        return self
