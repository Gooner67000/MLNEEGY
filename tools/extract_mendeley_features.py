"""Turns two large raw-waveform datasets hosted on Mendeley Data into compact,
committed feature tables, so CI trains from those instead of re-downloading
~2.4 GB on every run.

Why not download in CI: Mendeley Data sits behind a bot check that blocks
scripted (Python) downloads. We don't try to get around that. Instead the raw
files are downloaded once by a person, this script is run on them, and the
resulting feature tables are committed along with a provenance record
(source, DOI, license, SHA-256 of every raw file) so the tables are auditable
and reproducible.

Depends only on numpy (+ nptdms for the TDMS files) -- no pandas/scipy -- so it
runs on locked-down machines too.

Usage:
  python tools/extract_mendeley_features.py bearing  <dir with the 60 UORED *.csv>  <manifest.tsv>
  python tools/extract_mendeley_features.py conveyor <path to KAIST "current,temp.zip">

Datasets:
  bearing  -- University of Ottawa Rolling-element Dataset (UORED-VAFCLS),
              Mendeley Data doi:10.17632/y2px5tg92h.2, CC BY 4.0.
              20 physical bearings x {healthy, developing fault, faulty},
              accelerometer + microphone at 42 kHz, 10 s per recording.
  conveyor -- KAIST rotating-machine dataset, Mendeley Data
              doi:10.17632/ztmf3m7h5x.6, CC BY 4.0 (Jung et al., Data in Brief
              2023). Motor phase current (25.6 kHz) + 2 thermocouples under
              0/2/4 Nm load: normal, bearing inner/outer race, shaft
              misalignment, rotor unbalance, several severities each.
"""
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROV = DATA / "provenance"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def waveform_features(x: np.ndarray, fs: float, bands_hz, prefix: str = "") -> dict:
    """Standard condition-monitoring features of one waveform window."""
    x = x.astype(np.float64)
    x = x - x.mean()
    ax = np.abs(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    sd = float(x.std()) or 1e-12
    peak = float(ax.max())
    mean_abs = float(ax.mean()) or 1e-12
    z = x / sd
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    total = float(spec.sum()) or 1e-12
    p = spec / total
    cent = float((freqs * p).sum())
    feats = {
        "rms": rms,
        "std": sd,
        "peak": peak,
        "p2p": float(x.max() - x.min()),
        "crest": peak / (rms or 1e-12),
        "kurtosis": float(np.mean(z ** 4) - 3.0),  # excess kurtosis (0 for Gaussian)
        "skewness": float(np.mean(z ** 3)),
        "shape": rms / mean_abs,
        "impulse": peak / mean_abs,
        "spec_cent": cent,
        "spec_bw": float(np.sqrt(((freqs - cent) ** 2 * p).sum())),
        "spec_ent": float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p))),
    }
    for i, (lo, hi) in enumerate(bands_hz):
        m = (freqs >= lo) & (freqs < hi)
        feats[f"band{i}"] = float(spec[m].sum() / total)
    return {prefix + k: v for k, v in feats.items()}


def write_gz_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows x {len(cols)} cols -> {path} "
          f"({path.stat().st_size / 1e3:.0f} KB)")


# --------------------------------------------------------------- bearing
UORED_TYPES = {"H": "healthy", "I": "inner_race", "O": "outer_race", "B": "ball", "C": "cage"}
UORED_STATES = {0: "healthy", 1: "developing", 2: "faulty"}


def extract_bearing(raw_dir: Path, manifest: Path) -> None:
    fs = 42_000.0
    win = int(0.5 * fs)  # 0.5 s windows -> 20 per 10 s recording
    bands = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 21000)]
    urls = {}
    for line in manifest.read_text().splitlines():
        name, url, *_ = line.split("\t")
        urls[name] = url

    rows, files_prov = [], []
    for path in sorted(raw_dir.glob("*.csv")):
        m = re.fullmatch(r"([HIOBC])_(\d+)_(\d)\.csv", path.name)
        if not m:
            continue
        ftype, bearing, state = m.group(1), int(m.group(2)), int(m.group(3))
        data = np.loadtxt(path, delimiter=",", skiprows=1, encoding="utf-8-sig")
        acc, mic, tdiff = data[:, 0], data[:, 1], data[:, 4]
        rpm, load = float(data[0, 2]), float(data[0, 3])
        # Healthy recordings are named H_<bearing>_0; the fault type of that
        # bearing is fixed by its ID range (1-5 inner, 6-10 outer, 11-15 ball, 16-20 cage).
        bearing_type = ["inner_race", "outer_race", "ball", "cage"][(bearing - 1) // 5]
        for w in range(len(acc) // win):
            s = slice(w * win, (w + 1) * win)
            row = {
                "source_file": path.name,
                "bearing_id": bearing,
                "bearing_fault_type": bearing_type,
                "state": UORED_STATES[state],
                "label": "healthy" if state == 0 else UORED_TYPES[ftype],
                "window": w,
                "rpm": rpm,
                "load": load,
                "temp_diff": float(tdiff[s].mean()),
            }
            row.update(waveform_features(acc[s], fs, bands))
            row.update(waveform_features(mic[s], fs, bands, prefix="mic_"))
            rows.append(row)
        files_prov.append({"file": path.name, "sha256": sha256_of(path),
                           "source_url": urls.get(path.name, "")})
        print(f"  {path.name}: bearing {bearing} ({bearing_type}), {UORED_STATES[state]}", flush=True)

    write_gz_csv(rows, DATA / "bearing_uored_features.csv.gz")
    PROV.mkdir(parents=True, exist_ok=True)
    (PROV / "bearing_uored.json").write_text(json.dumps({
        "dataset": "University of Ottawa Rolling-element Dataset - Vibration and Acoustic "
                   "Faults under Constant Load and Speed conditions (UORED-VAFCLS)",
        "doi": "10.17632/y2px5tg92h.2",
        "landing_page": "https://data.mendeley.com/datasets/y2px5tg92h/2",
        "license": "CC BY 4.0",
        "raw_format": "CSV: Accelerometer, Acoustic, Speed, Load, Temperature Difference @ 42 kHz, 10 s",
        "window_seconds": 0.5,
        "n_raw_files": len(files_prov),
        "files": files_prov,
    }, indent=2))


# --------------------------------------------------------------- conveyor
def extract_conveyor_vibration(zip_path: Path) -> None:
    """KAIST vibration (4 accelerometers, 25.6 kHz). Order-tracked: each window
    finds the actual shaft speed from the spectrum and measures vibration at
    1x/2x/3x of it, so run-to-run supply-frequency differences (48 vs 49 Hz
    between recording sessions -- see the note on the motor-current data) can't
    leak into the features. Shaft speed itself is NOT output as a feature.

    Each (2-second window, accelerometer) is one row: the app takes a single
    accelerometer's reading, which is what most conveyor drives would have."""
    import scipy.io as sio

    g_per_ms2 = 0.101971621297793
    bands = [(0, 200), (200, 1000), (1000, 5000), (5000, 12800)]
    rows = []
    zf = zipfile.ZipFile(zip_path)
    for info in sorted(zf.infolist(), key=lambda i: i.filename):
        m = re.fullmatch(r"(\d)Nm_([A-Za-z]+)(?:_(\w+))?\.mat", info.filename)
        if not m:
            continue
        load_nm, cond, severity = int(m.group(1)), m.group(2), m.group(3) or "none"
        label = {"Normal": "normal", "BPFI": "bearing_inner", "BPFO": "bearing_outer",
                 "Misalign": "misalignment", "Unbalance": "unbalance",
                 "Unbalalnce": "unbalance",  # typo in the source's 2 Nm file names
                 }[cond]
        sig = sio.loadmat(io.BytesIO(zf.read(info)), squeeze_me=True,
                          struct_as_record=False)["Signal"]
        fs = 1.0 / float(sig.x_values.increment)
        y = np.asarray(sig.y_values.values, dtype=np.float64) * g_per_ms2  # -> g
        win = int(round(2 * fs))
        n_win = y.shape[0] // win
        for w in range(n_win):
            for ch in range(y.shape[1]):
                x = y[w * win:(w + 1) * win, ch]
                feats = waveform_features(x, fs, bands)
                xc = x - x.mean()
                spec = np.abs(np.fft.rfft(xc * np.hanning(len(xc)))) ** 2
                freqs = np.fft.rfftfreq(len(xc), d=1.0 / fs)
                search = (freqs >= 40) & (freqs <= 55)
                shaft = float(freqs[search][np.argmax(spec[search])])
                tot = spec.sum() or 1e-12

                def order_share(k):
                    return float(spec[np.abs(freqs - k * shaft) <= 1.0].sum() / tot)

                row = {"source_file": info.filename, "load_nm": load_nm, "label": label,
                       "severity": severity, "window": w, "channel": ch,
                       "_shaft_hz_not_a_feature": shaft}
                row.update({k: feats[k] for k in ["rms", "peak", "crest", "kurtosis", "skewness",
                                                   "shape", "impulse", "spec_cent", "spec_bw",
                                                   "spec_ent", "band0", "band1", "band2", "band3"]})
                row.update({
                    "order1_share": order_share(1),
                    "order2_share": order_share(2),
                    "order3_share": order_share(3),
                    "order1_amp": feats["rms"] * np.sqrt(order_share(1)),
                    "order2_amp": feats["rms"] * np.sqrt(order_share(2)),
                })
                rows.append(row)
        print(f"  {info.filename}: {n_win} windows x {y.shape[1]} accelerometers "
              f"({label}, load {load_nm} Nm)", flush=True)

    write_gz_csv(rows, DATA / "conveyor_kaist_features.csv.gz")
    PROV.mkdir(parents=True, exist_ok=True)
    (PROV / "conveyor_kaist.json").write_text(json.dumps({
        "dataset": "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating "
                   "Machine Under Varying Load Conditions for Fault Diagnosis (KAIST)",
        "doi": "10.17632/ztmf3m7h5x.6",
        "landing_page": "https://data.mendeley.com/datasets/ztmf3m7h5x/6",
        "paper": "Jung et al., Data in Brief (2023)",
        "license": "CC BY 4.0",
        "raw_file": "vibration.zip",
        "raw_sha256": sha256_of(zip_path),
        "window_seconds": 2.0,
        "channels_used": "4 accelerometers (g), each window/channel is one row",
        "why_not_motor_current": (
            "The same dataset's motor-current/temperature files were examined first and "
            "rejected: normal and unbalance files were recorded with a 48 Hz supply while "
            "most bearing/misalignment files used 49 Hz, and current-spectrum features "
            "tracked that session difference rather than the faults (misalignment recorded "
            "at 48 Hz looked identical to normal). Absolute temperature also drifts between "
            "runs. Vibration features here are order-tracked to avoid the same trap."),
    }, indent=2))


def extract_conveyor(zip_path: Path) -> None:
    """REJECTED -- kept for the record. Motor-current features from this
    dataset track recording session (48 vs 49 Hz supply), not faults; see
    extract_conveyor_vibration."""
    import nptdms

    bands = [(0, 40), (40, 200), (200, 1000), (1000, 5000), (5000, 12800)]
    rows = []
    zf = zipfile.ZipFile(zip_path)
    for info in sorted(zf.infolist(), key=lambda i: i.filename):
        m = re.fullmatch(r"(\d)Nm_([A-Za-z]+)(?:_(\w+))?\.tdms", info.filename)
        if not m:
            continue
        load_nm, cond, severity = int(m.group(1)), m.group(2), m.group(3) or "none"
        label = {"Normal": "normal", "BPFI": "bearing_inner", "BPFO": "bearing_outer",
                 "Misalign": "misalignment", "Unbalance": "unbalance"}[cond]
        with zf.open(info) as f:
            tdms = nptdms.TdmsFile.read(io.BytesIO(f.read()))
        chans = {c.name.split("/")[-1] + ("_T" if "Mod1" in c.name else "_I"): c
                 for c in tdms["Log"].channels()}
        cur = chans["ai0_I"]
        fs = 1.0 / cur.properties["wf_increment"]
        i_sig = cur[:]
        t1, t2 = chans["ai0_T"][:], chans["ai1_T"][:]
        win = int(round(fs))  # 1-second windows
        n_win = len(i_sig) // win
        for w in range(n_win):
            s = slice(w * win, (w + 1) * win)
            seg = i_sig[s]
            row = {"source_file": info.filename, "load_nm": load_nm, "label": label,
                   "severity": severity, "window": w}
            row.update(waveform_features(seg, fs, bands, prefix="cur_"))
            # Motor-current signature: supply fundamental and energy around it.
            spec = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(len(seg)))) ** 2
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
            tot = spec.sum() or 1e-12
            band = (freqs > 40) & (freqs < 80)
            f0 = float(freqs[band][np.argmax(spec[band])])
            near = (np.abs(freqs - f0) <= 2)
            side = (np.abs(freqs - f0) > 2) & (np.abs(freqs - f0) <= 40)
            harm = np.zeros_like(freqs, dtype=bool)
            for k in range(2, 11):
                harm |= np.abs(freqs - k * f0) <= 2
            row.update({
                "cur_f0_hz": f0,
                "cur_fund_ratio": float(spec[near].sum() / tot),
                "cur_sideband_ratio": float(spec[side].sum() / tot),
                "cur_harmonic_ratio": float(spec[harm].sum() / tot),
                # Temperature: only differences/trends, never absolute level --
                # absolute temperature drifts between test runs on a rig and
                # would let a model tell runs apart instead of faults.
                "temp_diff": float(t1[s].mean() - t2[s].mean()),
                "temp1_slope": float(t1[s][-1] - t1[s][0]),
                "temp2_slope": float(t2[s][-1] - t2[s][0]),
            })
            rows.append(row)
        print(f"  {info.filename}: {n_win} windows ({label}, load {load_nm} Nm)", flush=True)

    write_gz_csv(rows, DATA / "conveyor_kaist_features.csv.gz")
    PROV.mkdir(parents=True, exist_ok=True)
    (PROV / "conveyor_kaist.json").write_text(json.dumps({
        "dataset": "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating "
                   "Machine Under Varying Load Conditions for Fault Diagnosis (KAIST)",
        "doi": "10.17632/ztmf3m7h5x.6",
        "landing_page": "https://data.mendeley.com/datasets/ztmf3m7h5x/6",
        "paper": "Jung et al., Data in Brief (2023), doi:10.1016/j.dib.2023.109049",
        "license": "CC BY 4.0",
        "raw_file": "current,temp.zip",
        "raw_sha256": sha256_of(zip_path),
        "window_seconds": 1.0,
        "channels_used": "motor phase current ai0 (A, 25.6 kHz); thermocouples ai0/ai1 (degC) "
                         "as difference/slope only",
    }, indent=2))


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "bearing":
        extract_bearing(Path(sys.argv[2]), Path(sys.argv[3]))
    elif kind == "conveyor":
        extract_conveyor_vibration(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
