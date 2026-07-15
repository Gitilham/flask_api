import ast
from pathlib import Path

from app.services.v21_pipeline import determine_final_decision


def test_new_rule_fixes_previous_false_suspicious_result():
    assert determine_final_decision(0.444902) == "REAL"


def test_sampling_function_remains_present_in_legacy_and_fastapi():
    root = Path(__file__).resolve().parents[1]
    names = []
    for path in (root / "legacy/app_flask_v21.py", root / "app/services/v21_pipeline.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names.append({node.name for node in tree.body if isinstance(node, ast.FunctionDef)})
    assert "read_video_frames" in names[0] & names[1]
    assert "build_v3_feature_vector" in names[0] & names[1]

