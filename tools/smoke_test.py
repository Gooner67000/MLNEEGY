"""End-to-end smoke test against a RUNNING backend (e.g. the Docker stack):
registers a business, creates one machine of EVERY type the API advertises,
sends each a few readings, and checks every one comes back with a real score.

Standard library only, so it runs anywhere.
Usage: python tools/smoke_test.py [http://localhost:8000]
"""
import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def call(method: str, path: str, body=None, token=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    for _ in range(60):
        try:
            if call("GET", "/health")["status"] == "ok":
                break
        except OSError:
            time.sleep(2)
    else:
        raise SystemExit("backend never became healthy")

    token = call("POST", "/auth/register", {"name": "CI Smoke Test",
                                            "email": f"ci-{int(time.time())}@example.com",
                                            "password": "password123"})["access_token"]
    types = call("GET", "/machine-types")
    failures = []
    for key, spec in types.items():
        m = call("POST", "/machines", {"name": f"{key}-ci", "machine_type": key}, token)
        # Typical value for every sensor the form would show, sent 3 times so
        # history-based types build rolling features.
        reading = {s["key"]: s["default"] for s in spec["sensors"]}
        for _ in range(3):
            r = call("POST", f"/machines/{m['id']}/readings",
                     {"payload": reading, "source": "api"}, token)
        ok = (r["trained"] is True and r["probability"] is not None and 0 <= r["probability"] <= 1
              and (r["risk_level"] == "High") == bool(r["alert"]))
        print(f"{'OK ' if ok else 'BAD'} {key:24s} p={r['probability']} risk={r['risk_level']} "
              f"diagnosis={r['diagnosis']}")
        if not ok:
            failures.append(key)
    if failures:
        raise SystemExit(f"Smoke test failed for: {failures}")
    print(f"All {len(types)} machine types scored end-to-end through the running stack.")


if __name__ == "__main__":
    main()
