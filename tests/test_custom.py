"""End-to-end test of the customer-onboarding pipeline (train_custom.py) on REAL
run-to-failure data: NASA C-MAPSS engine logs reshaped into exactly the two
CSVs a customer would export -- a sensor log and a list of breakdowns."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("xgboost")

ROOT = Path(__file__).parent.parent
CMAPSS = ROOT / "data" / "cmapss_train_FD001.csv"


@pytest.fixture(scope="module")
def customer_export(tmp_path_factory):
    d = tmp_path_factory.mktemp("customer")
    raw = pd.read_csv(CMAPSS, index_col=0)
    raw = raw[raw["engine"] <= 30]  # 30 machines keeps the test fast
    sensor_cols = [c for c in raw.columns if c not in ("engine", "cycle")]
    raw = raw.rename(columns={c: f"s{i}" for i, c in enumerate(sensor_cols)})
    base = pd.Timestamp("2025-01-01")
    raw["timestamp"] = base + pd.to_timedelta(raw["cycle"], unit="h")
    raw["machine_id"] = "engine-" + raw["engine"].astype(str)
    sensors = raw[["timestamp", "machine_id"] + [f"s{i}" for i in range(len(sensor_cols))]]
    # Each engine ran until it failed: the breakdown is just after its last reading.
    events = (raw.groupby("machine_id")["timestamp"].max() + pd.Timedelta(hours=1)).reset_index()
    events["event_type"] = "failure"
    sensors.to_csv(d / "sensors.csv", index=False)
    events.to_csv(d / "failures.csv", index=False)
    return d


def run(args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "train_custom.py"), *args], cwd=cwd,
                          capture_output=True, text=True)


def test_custom_pipeline_learns_real_signal(customer_export):
    out = customer_export / "models"
    r = run(["--key", "demo_engines", "--name", "Demo engines",
             "--sensors", str(customer_export / "sensors.csv"),
             "--events", str(customer_export / "failures.csv"),
             "--horizon", "30h", "--out", str(out)], ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    metrics = json.loads((out / "custom_demo_engines" / "metrics.json").read_text())
    assert "leave-machines-out" in metrics["validation"]
    assert metrics["roc_auc"] > 0.8, metrics  # scored on engines it never trained on
    spec = json.loads((out / "custom_demo_engines" / "type.json").read_text())
    assert spec["key"] == "custom_demo_engines" and spec["sensors"]


def test_custom_pipeline_refuses_too_few_failures(customer_export, tmp_path):
    events = pd.read_csv(customer_export / "failures.csv").head(2)
    events.to_csv(tmp_path / "two_failures.csv", index=False)
    r = run(["--key", "x", "--name", "x", "--sensors", str(customer_export / "sensors.csv"),
             "--events", str(tmp_path / "two_failures.csv"), "--out", str(tmp_path)], ROOT)
    assert r.returncode != 0
    assert "need at least" in (r.stdout + r.stderr)
