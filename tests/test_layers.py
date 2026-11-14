"""Who may import whom (CLAUDE.md "Layout"): the rules that keep the scoring free of the site, the network and the
LLM, and the site free of the command line. Read from the source (no imports executed)."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "fq"


def imports(path: Path) -> set[str]:
    """Every `fq.<module>[.<sub>]` a file imports, as dotted names (lazy imports inside functions included)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "fq":
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] == "fq")
        elif isinstance(node, ast.ImportFrom) and node.module is None and node.level == 0:
            found.update(f"fq.{a.name}" for a in node.names)
    return found


def files(*names: str) -> list[Path]:
    out = []
    for name in names:
        p = SRC / name
        out += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    return out


def touching(paths: list[Path], *forbidden: str) -> list[str]:
    bad = []
    for path in paths:
        for imp in imports(path):
            if any(imp == f or imp.startswith(f + ".") for f in forbidden):
                bad.append(f"{path.relative_to(SRC)} imports {imp}")
    return bad


def test_the_scoring_core_never_reaches_the_site_the_network_or_the_llm():
    core = files("metrics", "ranking.py", "filters.py", "settings.py")
    above = ("fq.report", "fq.grading", "fq.pipeline", "fq.cli", "fq.fetch", "fq.forward", "fq.evaluate", "fq.columns",
             "fq.refresh", "fq.health", "fq.history", "fq.crawlers")  # fmt: skip
    assert touching(core, *above) == []


def test_auxiliary_crawlers_stay_decoupled_from_pipeline_and_site():
    non_cli = [p for p in files(".") if p.name not in ("cli.py", "crawlers.py")]
    assert touching(non_cli, "fq.crawlers") == []


def test_the_foundations_stay_at_the_bottom():
    assert touching(files("settings.py"), "fq.universe", "fq.fx", "fq.fetch", "fq.ranking", "fq.metrics") == []
    assert (
        touching(files("universe.py", "fx.py", "inflation.py"), "fq.report", "fq.pipeline", "fq.cli", "fq.metrics")
        == []
    )


def test_the_site_does_not_know_the_command_line_or_the_downloads():
    assert (
        touching(files("report"), "fq.cli", "fq.pipeline", "fq.fetch", "fq.refresh", "fq.evaluate", "fq.health") == []
    )


def test_only_the_entry_points_import_the_pipeline_and_the_command_line():
    everything = [p for p in files(".") if p.name not in ("cli.py", "pipeline.py", "columns.py", "__main__.py")]
    assert touching(everything, "fq.cli", "fq.pipeline") == []


def test_the_llm_stays_in_grading_and_grading_reads_the_site_only_for_links():
    assert touching(files("metrics", "ranking.py", "filters.py", "settings.py"), "fq.grading") == []
    grading = imports(SRC / "grading.py")
    assert {i for i in grading if i.startswith("fq.report")} <= {"fq.report.links", "fq.report.links.slug"}
