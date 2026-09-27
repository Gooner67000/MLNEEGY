from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "predictive_maintenance_model.pkl").exists(), reason="run train.py first"
)


def test_app_renders_prediction():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].label == "Failure Probability"
    assert at.metric[0].value.endswith("%")


def test_app_high_risk_inputs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    sliders = {s.label: s for s in at.sidebar.slider}
    sliders["Torque (Nm)"].set_value(70.0)
    sliders["Tool Wear (min)"].set_value(240)
    sliders["Rotational Speed (rpm)"].set_value(1200)
    at.run()
    assert not at.exception
    assert float(at.metric[0].value.rstrip("%")) > 50
