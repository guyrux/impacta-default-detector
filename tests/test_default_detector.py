import importlib

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


def test_analysis_module_exposes_main_and_is_import_safe():
    module = importlib.import_module("default_detector.analyze_python")

    assert callable(getattr(module, "main", None))
    assert "df" not in module.__dict__
    assert "results" not in module.__dict__


def test_plot_target_distribution_uses_percentage_labels(monkeypatch):
    from default_detector import plots

    df = pd.DataFrame({plots.TARGET_COLUMN: [0, 0, 1, 1, 1]})
    captured = {}

    original_subplots = plots.plt.subplots

    def fake_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        original_text = ax.text

        def wrapped_text(*text_args, **text_kwargs):
            captured.setdefault("labels", []).append(text_args[2])
            return original_text(*text_args, **text_kwargs)

        ax.text = wrapped_text
        return fig, ax

    monkeypatch.setattr(plots.plt, "subplots", fake_subplots)

    plots.plot_target_distribution(df)

    assert len(captured["labels"]) == 2
    assert all(label.endswith("%") for label in captured["labels"])
    assert [float(label.rstrip("%")) for label in captured["labels"]] == pytest.approx(
        [40.0, 60.0]
    )
