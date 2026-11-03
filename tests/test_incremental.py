"""Incremental work: company pages rewritten only when their inputs change; full fetches on threads."""

from dataclasses import replace

from conftest import make_raw
from fq import fetch
from fq.metrics import calculate_metrics
from fq.ranking import apply_dual_ranking
from fq.report import company_page, html


def _rendered(monkeypatch) -> list[str]:
    """Tickers rendered by every later `build_site` (inline below 50 pages)."""
    seen: list[str] = []
    real = company_page._render_company_batch

    def spy(args):
        seen.extend(item[2]["Ticker"] for item in args[-1])
        real(args)

    monkeypatch.setattr(company_page, "_render_company_batch", spy)
    return seen


def test_unchanged_pages_are_not_rendered_again(cfg, monkeypatch):
    ranked = apply_dual_ranking(calculate_metrics(make_raw(cfg, n=6), cfg), cfg)
    seen = _rendered(monkeypatch)
    html.build_site(ranked, cfg)
    assert len(seen) == len(ranked)
    page = (cfg.site_dir / "companies").glob("*.html").__next__().read_text(encoding="utf-8")
    assert "Data as of" in page and "Generated" not in page  # the build time would change every page

    seen.clear()
    html.build_site(ranked, cfg)
    assert seen == []  # nothing changed

    changed = ranked.copy()
    changed.loc[changed.index[0], "longBusinessSummary"] = "A new summary."
    seen.clear()
    html.build_site(changed, cfg)
    assert seen == [changed.iloc[0]["Ticker"]]

    seen.clear()
    html.build_site(changed, replace(cfg, highlight_top_fraction=0.3))  # a setting: every page
    assert len(seen) == len(ranked)

    (cfg.site_dir / "companies" / f"{html.slug(changed.iloc[1]['Ticker'])}.html").unlink()
    seen.clear()
    html.build_site(changed, replace(cfg, highlight_top_fraction=0.3))
    assert seen == [changed.iloc[1]["Ticker"]]  # a deleted page is written again


def test_threaded_fetch_keeps_order_and_outcomes(cfg, monkeypatch):
    def fake(ticker, cfg, rates):
        if ticker == "NONE":
            return None
        if ticker == "BAD":
            raise RuntimeError("Too Many Requests")
        return {"Ticker": ticker}

    monkeypatch.setattr(fetch, "fetch_ticker", fake)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    todo = ["A", "NONE", "B", "BAD", "C"]
    out = fetch._fetch_all(todo, replace(cfg, fetch_workers=3, fetch_retries=2), {})
    assert out[0] == {"Ticker": "A"} and out[1] is None and out[2] == {"Ticker": "B"}
    assert out[3] is fetch.FAILED and out[4] == {"Ticker": "C"}
