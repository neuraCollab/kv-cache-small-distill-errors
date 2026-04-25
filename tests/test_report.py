import json
from pathlib import Path

import numpy as np

from kvtrace.analysis.report import build_report, write_heatmap


def test_build_report_markdown_contains_table(tmp_path):
    judgments = [
        {"quant_method": "bf16", "category": "A", "model": "m1"},
        {"quant_method": "bf16", "category": "B", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "A", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "F", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "F", "model": "m1"},
    ]
    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    build_report(judgments, md_path=md_path, json_path=json_path)

    text = md_path.read_text()
    assert "| method" in text.lower() or "|method" in text.lower()
    assert "bf16" in text
    assert "fp8_e5m2" in text
    assert "chi" in text.lower()

    data = json.loads(json_path.read_text())
    assert "signatures" in data
    assert "chi_square_p" in data


def test_report_is_deterministic(tmp_path):
    judgments = [
        {"quant_method": "bf16", "category": "A", "model": "m1"},
        {"quant_method": "fp8_e5m2", "category": "F", "model": "m1"},
    ]
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    build_report(judgments, md_path=p1, json_path=tmp_path / "a.json")
    build_report(judgments, md_path=p2, json_path=tmp_path / "b.json")
    assert p1.read_text() == p2.read_text()


def test_write_heatmap_creates_png(tmp_path):
    matrix = np.array([[0.3, 0.1, 0.0, 0.0, 0.5, 0.1], [0.1, 0.0, 0.0, 0.0, 0.0, 0.9]])
    out = tmp_path / "heatmap.png"
    write_heatmap(matrix, method_names=["bf16", "fp8_e5m2"], out_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
