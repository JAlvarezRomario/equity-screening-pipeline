"""The column catalogue (`fq.columns`): every column the pipeline adds is attributed to the step that wrote it."""

import fq.metrics as metrics_pkg
from conftest import make_raw
from fq.columns import generic, render, trace
from fq.metrics import calculate_metrics


def test_every_added_column_is_attributed_and_the_pipeline_is_left_alone(cfg):
    raw = make_raw(cfg, n=6)
    original = metrics_pkg._value
    steps, inline = trace(raw, cfg)
    assert metrics_pkg._value is original  # the wrappers are removed again
    attributed = {c for _, _, cols in steps for c in cols}
    result = calculate_metrics(raw, cfg)
    added = set(result.columns) - set(raw.columns)
    assert added == attributed | set(inline)  # nothing unaccounted for, nothing invented
    assert not attributed & set(inline)
    owner = [label for label, _, cols in steps if "Owner_Yield" in cols]
    assert owner == ["metrics.valuation._value"]  # the scored Value input has one writer
    assert len({label for label, _, _ in steps}) == len(steps) and len(steps) >= 20


def test_periods_collapse_and_the_catalogue_renders(cfg):
    assert (
        generic("ROIC_Y0") == "ROIC_<p>" and generic("ROIC_TTM") == "ROIC_<p>" and generic("ROIC_Norm") == "ROIC_Norm"
    )
    steps, inline = trace(make_raw(cfg, n=4), cfg)
    text = render(steps, inline, 4)
    assert text.startswith("# Metric columns by pipeline step") and "`Owner_Yield`" in text
    assert "`ROIC_<p>`" in text and "ROIC_Y0" not in text
