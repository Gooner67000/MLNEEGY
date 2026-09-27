"""Multi-tenant frontend: businesses register, add machines of any type(s),
feed them readings (typed in, uploaded as CSV, or pushed by their own
automation via the API), and see current risk per machine.

This is the UI for backend/main.py. It never talks to the ML models
directly -- everything goes through the API, which is what makes the same
core work both as something a business downloads and runs on their own
network, and (with DATABASE_URL pointed at a real database) as a hosted
multi-tenant service later.

Run:  streamlit run frontend/app.py
Env:  API_BASE (default http://localhost:8000) -- where this app reaches the backend
      PUBLIC_API_BASE -- the address customers should use (defaults to API_BASE;
      differs when hosted, where API_BASE is a private internal address)
"""
import json
import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")
if "://" not in API_BASE:  # e.g. a bare host:port from a hosting provider
    API_BASE = "http://" + API_BASE
PUBLIC_API_BASE = (os.environ.get("PUBLIC_API_BASE") or API_BASE).rstrip("/")
st.set_page_config(page_title="Predictive Maintenance Platform", page_icon="🛠️", layout="wide")


def api(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    resp = requests.request(method, f"{API_BASE}{path}", headers=headers, timeout=30, **kwargs)
    if resp.status_code >= 400:
        st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
        return None
    return resp.json()


@st.cache_data(ttl=60)
def machine_types() -> dict:
    resp = requests.get(f"{API_BASE}/machine-types", timeout=15)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------- auth gate
if "token" not in st.session_state:
    st.title("🛠️ Predictive Maintenance Platform")
    tab_login, tab_register = st.tabs(["Log in", "Register your business"])
    with tab_login:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in") and email and password:
                result = api("POST", "/auth/login", json={"email": email, "password": password})
                if result:
                    st.session_state["token"] = result["access_token"]
                    st.session_state["business_name"] = result["business_name"]
                    st.rerun()
    with tab_register:
        with st.form("register"):
            name = st.text_input("Business name")
            email_r = st.text_input("Email", key="reg_email")
            password_r = st.text_input("Password (8+ characters)", type="password", key="reg_pw")
            if st.form_submit_button("Create account") and name and email_r and password_r:
                result = api("POST", "/auth/register",
                            json={"name": name, "email": email_r, "password": password_r})
                if result:
                    st.session_state["token"] = result["access_token"]
                    st.session_state["business_name"] = result["business_name"]
                    st.rerun()
    st.stop()

# ------------------------------------------------------------------- app
st.sidebar.write(f"**{st.session_state['business_name']}**")
if st.sidebar.button("Log out"):
    st.session_state.clear()
    st.rerun()

types = machine_types()
tab_machines, tab_reading, tab_dashboard, tab_api = st.tabs(
    ["My Machines", "Add a Reading", "Dashboard", "Live/API import"]
)

# --------------------------------------------------------- machine registry
with tab_machines:
    st.subheader("Register a machine")
    st.caption("You can register machines of a single type or mix as many types as you run.")
    with st.form("add_machine"):
        m_name = st.text_input("Machine name / ID", placeholder="e.g. Mill-3, Pump-B, Turbine-North")
        type_key = st.selectbox("Machine type", list(types.keys()),
                                format_func=lambda k: types[k]["name"] +
                                ("" if types[k]["trained"] else "  (schema only, not yet trained)") +
                                ("  (experimental)" if types[k].get("maturity") == "experimental" else ""))
        st.caption(types[type_key]["description"] + " — trained on: " + types[type_key]["dataset"])
        if not types[type_key]["trained"] or types[type_key].get("maturity") == "experimental":
            st.warning(types[type_key]["note"])
        if st.form_submit_button("Add machine") and m_name:
            if api("POST", "/machines", json={"name": m_name, "machine_type": type_key}):
                st.success(f"Added {m_name}")
                st.cache_data.clear()
                st.rerun()

    st.subheader("Registered machines")
    machines = api("GET", "/machines") or []
    if not machines:
        st.info("No machines yet — add one above.")
    for m in machines:
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(f"**{m['name']}**")
        c2.write(types.get(m["machine_type"], {}).get("name", m["machine_type"]))
        if c3.button("Delete", key=f"del_{m['id']}"):
            api("DELETE", f"/machines/{m['id']}")
            st.rerun()

# --------------------------------------------------------------- readings
with tab_reading:
    machines = api("GET", "/machines") or []
    if not machines:
        st.info("Register a machine first.")
    else:
        options = {f"{m['name']} ({types[m['machine_type']]['name']})": m for m in machines}
        choice = st.selectbox("Machine", list(options))
        machine = options[choice]
        type_def = types[machine["machine_type"]]

        entry_mode = st.radio("Entry method", ["Manual", "CSV upload"], horizontal=True)
        if entry_mode == "Manual":
            with st.form("manual_reading"):
                values = {}
                for s in type_def["sensors"]:
                    if s["manual_entry"]:
                        values[s["key"]] = st.slider(
                            f"{s['label']} ({s['unit']})" if s["unit"] else s["label"],
                            float(s["min"]), float(s["max"]), float(s["default"]),
                        )
                if st.form_submit_button("Score this reading"):
                    result = api("POST", f"/machines/{machine['id']}/readings",
                                json={"payload": values, "source": "manual"})
                    if result:
                        st.session_state["last_result"] = result
            if st.session_state.get("last_result"):
                r = st.session_state["last_result"]
                if not r["trained"]:
                    st.warning(r["note"])
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Failure probability", f"{r['probability'] * 100:.1f}%")
                    c2.metric("Risk level", r["risk_level"])
                    c3.metric("Alert", "🔔 Yes" if r["alert"] else "No")
                    if r["diagnosis"]:
                        st.write(f"Diagnosis: **{r['diagnosis']}**")
                    if r["note"]:
                        st.caption(r["note"])
        else:
            st.write("Columns expected (any row can omit fields the model can default):")
            st.code(", ".join(s["key"] for s in type_def["sensors"]))
            upload = st.file_uploader("CSV of readings, oldest row first", type="csv")
            if upload is not None:
                batch = pd.read_csv(upload)
                readings = [
                    {"payload": {k: float(v) for k, v in row.dropna().to_dict().items()},
                     "source": "csv"}
                    for _, row in batch.iterrows()
                ]  # cast off numpy scalar types: the stdlib json encoder requests uses can't serialize them
                if st.button(f"Score {len(readings)} readings"):
                    results = api("POST", f"/machines/{machine['id']}/readings/bulk",
                                  json={"readings": readings})
                    if results:
                        out = pd.DataFrame(results)
                        st.dataframe(out)
                        st.download_button("Download scored CSV", out.to_csv(index=False),
                                           "scored_readings.csv", "text/csv")

# --------------------------------------------------------------- dashboard
with tab_dashboard:
    machines = api("GET", "/machines") or []
    if not machines:
        st.info("Register a machine first.")
    for m in machines:
        st.markdown(f"### {m['name']} — {types[m['machine_type']]['name']}")
        latest = api("GET", f"/machines/{m['id']}/risk/latest")
        if not latest:
            st.caption("No readings yet.")
            continue
        if not latest["trained"]:
            st.warning(latest["note"])
            continue
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest failure probability", f"{latest['probability'] * 100:.1f}%")
        c2.metric("Risk level", latest["risk_level"])
        c3.metric("Alert", "🔔 Yes" if latest["alert"] else "No")
        history = api("GET", f"/machines/{m['id']}/risk/history") or []
        if len(history) > 1:
            hist_df = pd.DataFrame(history).sort_values("computed_at")
            st.line_chart(hist_df.set_index("computed_at")["probability"])
        st.divider()

# --------------------------------------------------------------- API docs
with tab_api:
    st.subheader("Point your own automation at this")
    st.write(
        "This is the entire 'automatic live import' feature: your PLC/SCADA/IoT "
        "system (or a small cron script) calls this endpoint on a schedule with "
        "the machine's current sensor readings, and gets a risk score back. "
        "Manual entry and CSV upload above call the exact same endpoint."
    )
    machines = api("GET", "/machines") or []
    if machines:
        example_machine = machines[0]
        example_type = types[example_machine["machine_type"]]
        example_payload = {s["key"]: s["default"] for s in example_type["sensors"]}
        example_body = json.dumps({"payload": example_payload, "source": "api"})
        st.code(
            f"""curl -X POST {PUBLIC_API_BASE}/machines/{example_machine['id']}/readings \\
  -H "Authorization: Bearer <your token>" \\
  -H "Content-Type: application/json" \\
  -d '{example_body}'""",
            language="bash",
        )
    st.write(f"Full interactive API docs: [{PUBLIC_API_BASE}/docs]({PUBLIC_API_BASE}/docs)")
