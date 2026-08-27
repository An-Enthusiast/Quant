"""Plumbing test for the toxicity classifier (alpha/toxicity_model.py).

IMPORTANT: this exercises the train -> predict -> save -> load code path on
a small synthetic sklearn dataset. It proves the pipeline executes
correctly; it is explicitly NOT a claim that any shipped model is fit for
production trading use -- see alpha/toxicity_model.py's module docstring
and docs/WHITEPAPER.md for why no pretrained model ships with this project.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha.features import FEATURE_COLUMNS
from alpha.toxicity_model import ToxicityClassifier


def _synthetic_dataset(n=200, seed=42):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=len(FEATURE_COLUMNS), n_informative=3, n_redundant=0, random_state=seed
    )
    return X, y


def test_train_predict_roundtrip():
    X, y = _synthetic_dataset()
    clf = ToxicityClassifier().train(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_feature_importances_keyed_by_feature_columns():
    X, y = _synthetic_dataset()
    clf = ToxicityClassifier().train(X, y)
    importances = clf.feature_importances()
    assert importances is not None
    assert set(importances.keys()) == set(FEATURE_COLUMNS)
    assert abs(sum(importances.values()) - 1.0) < 1e-6


def test_save_and_load_roundtrip(tmp_path):
    X, y = _synthetic_dataset()
    clf = ToxicityClassifier().train(X, y)
    proba_before = clf.predict_proba(X)

    path = tmp_path / "toxicity_test.joblib"
    clf.save(path)

    loaded = ToxicityClassifier().load(path)
    proba_after = loaded.predict_proba(X)
    assert np.allclose(proba_before, proba_after)


def test_predict_before_train_raises():
    clf = ToxicityClassifier()
    with pytest.raises(RuntimeError):
        clf.predict_proba(np.zeros((1, len(FEATURE_COLUMNS))))


def test_train_requires_both_classes():
    X = np.random.default_rng(0).normal(size=(10, len(FEATURE_COLUMNS)))
    y = np.zeros(10)
    clf = ToxicityClassifier()
    with pytest.raises(ValueError):
        clf.train(X, y)


def test_random_forest_backend_explicit():
    X, y = _synthetic_dataset()
    clf = ToxicityClassifier(backend="random_forest").train(X, y)
    proba = clf.predict_proba(X)
    assert np.all((proba >= 0) & (proba <= 1))
