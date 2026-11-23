"""Unit, API, symmetry, and locked-test parity checks for ML and DL inference engines."""
from __future__ import annotations

import csv
import io
import json
import threading
import urllib.request
import zipfile
from pathlib import Path

import pytest

from nlu_authorship.artifacts import find_artifact
from nlu_authorship.presets import DEMO_PRESETS
from nlu_authorship.web_demo import create_server

ROOT = Path(__file__).resolve().parents[1]


def test_presets_well_formed():
    assert len(DEMO_PRESETS) == 4
    for p in DEMO_PRESETS:
        assert p["text_1"].strip()
        assert p["text_2"].strip()
        assert p["ground_truth"] in ("same_author", "different_author")


def test_web_demo_routes():
    server = create_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            html = r.read().decode("utf-8")
            assert r.status == 200
            assert "#A31F34" in html
            assert "DeBERTa-v3-large" in html

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as r:
            st = json.loads(r.read().decode("utf-8"))
            assert "ml" in st and "dl" in st

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/presets", timeout=5) as r:
            pr = json.loads(r.read().decode("utf-8"))
            assert len(pr["presets"]) == 4
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(find_artifact("ml") is None, reason="ML artifact not present locally")
def test_ml_engine_symmetry_and_parity():
    from nlu_authorship.ml_engine import StylometricMLEngine

    eng = StylometricMLEngine()
    p0 = DEMO_PRESETS[0]
    res = eng.predict_pair(p0["text_1"], p0["text_2"], check_symmetry=True)
    assert 0.0 < res["probability"] < 1.0
    assert res["symmetry_error"] < 1e-5

    z1 = ROOT / "outputs" / "01_stylometric_ml" / "results_01_stylometric_ml.zip"
    if z1.is_file():
        with zipfile.ZipFile(z1) as z:
            rows = list(csv.DictReader(io.StringIO(z.read("01_stylometric_ml/test_open_predictions_ml.csv").decode("utf-8"))))
        ref = {int(r["pair_id"]): float(r["ml_sota_prob"]) for r in rows}
        assert abs(res["probability"] - ref[p0["pair_id"]]) < 1e-4


@pytest.mark.skipif(find_artifact("dl") is None, reason="DL artifact not present locally")
def test_dl_engine_symmetry_and_parity():
    from nlu_authorship.dl_engine import DebertaLargeEngine

    eng = DebertaLargeEngine()
    p0 = DEMO_PRESETS[0]
    res = eng.predict_pair(p0["text_1"], p0["text_2"], check_symmetry=True)
    assert 0.0 < res["probability"] < 1.0
    assert res["symmetry_error"] < 1e-5

    z3 = ROOT / "outputs" / "03_deberta_v3_large" / "results_03_deberta_v3_large.zip"
    if z3.is_file():
        with zipfile.ZipFile(z3) as z:
            rows = list(csv.DictReader(io.StringIO(z.read("03_deberta_v3_large/test_open_predictions_deberta_large.csv").decode("utf-8"))))
        ref = {int(r["pair_id"]): float(r["deberta_large_prob"]) for r in rows}
        assert abs(res["probability"] - ref[p0["pair_id"]]) < 5e-3
