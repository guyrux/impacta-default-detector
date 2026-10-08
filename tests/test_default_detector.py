import importlib
import sys
from pathlib import Path


def test_analysis_module_exposes_main_and_is_import_safe():
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.modules.pop("default_detector.analyze_python", None)

    module = importlib.import_module("default_detector.analyze_python")

    assert callable(getattr(module, "main", None))
    assert "df" not in module.__dict__
    assert "results" not in module.__dict__
