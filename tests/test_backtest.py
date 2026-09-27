"""Proves the backtest has no lookahead: no shared engines between train/test,
and a row's features never change when later cycles are added to the frame."""
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
import backtest

ROOT = Path(__file__).parent.parent


def test_train_test_engines_never_overlap():
    df = backtest.load_data()
    df, _ = backtest.add_labels_and_features(df)
    _, _, train_engines, test_engines = backtest.engine_split(df)
    assert set(train_engines).isdisjoint(test_engines)


def test_features_are_causal_not_lookahead():
    """Row t's features must be identical whether we compute them from the
    full engine history or from only rows up to cycle t -- i.e. no feature
    secretly peeks at cycle > t."""
    df = backtest.load_data()
    engine_1 = df[df["engine"] == 1].sort_values("cycle")
    full, feature_cols = backtest.add_labels_and_features(engine_1.copy())

    mid_cycle = int(engine_1["cycle"].median())
    truncated = engine_1[engine_1["cycle"] <= mid_cycle].copy()
    truncated_feats, _ = backtest.add_labels_and_features(truncated)

    row_full = full[full["cycle"] == mid_cycle][feature_cols].iloc[0]
    row_truncated = truncated_feats[truncated_feats["cycle"] == mid_cycle][feature_cols].iloc[0]
    pd.testing.assert_series_equal(row_full, row_truncated, check_names=False)


def test_label_is_never_in_the_feature_list():
    df = backtest.load_data()
    df, feature_cols = backtest.add_labels_and_features(df)
    assert "will_fail_soon" not in feature_cols
    assert "RUL" not in feature_cols
    assert "max_cycle" not in feature_cols
