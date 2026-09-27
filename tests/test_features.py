"""No-lookahead guarantee for the shared feature code used in training AND in
the live backend: the features for reading t must be identical whether or
not readings after t exist. (Random numbers here only exercise the code path;
no model is involved.)"""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from pm_features import (HVAC_FEATURES, HVAC_RAW, ROBOT_FEATURES, ROBOT_RAW,
                         hvac_features, robot_features)


def _random_frame(cols, n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(50, 10, size=(n, len(cols))), columns=cols)
    return df


@pytest.mark.parametrize("fn,raw,feats", [
    (hvac_features, HVAC_RAW, HVAC_FEATURES),
    (robot_features, ROBOT_RAW, ROBOT_FEATURES),
])
def test_features_never_look_ahead(fn, raw, feats):
    df = _random_frame(raw)
    if "OCCU_MOD" in df:
        df["OCCU_MOD"] = 1
    full = fn(df)
    for t in (0, 5, 20, 45):
        truncated = fn(df.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t][feats], truncated.iloc[-1][feats],
                                       check_names=False)


def test_hvac_accepts_damper_as_percent_or_fraction():
    base = _random_frame(HVAC_RAW, n=2)
    pct, frac = base.copy(), base.copy()
    pct["RTU_OA_DMPR_DM"], frac["RTU_OA_DMPR_DM"] = 40.0, 0.40
    assert hvac_features(pct)["dmpr_frac"].iloc[-1] == pytest.approx(0.40)
    assert hvac_features(frac)["dmpr_frac"].iloc[-1] == pytest.approx(0.40)
