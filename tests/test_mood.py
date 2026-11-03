"""Owner FCF after acquisitions, own-history valuation, estimate revisions and the Dislocation flag."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fq.fetch import estimate_revisions
from fq.metrics import calculate_metrics
from fq.ranking import dislocation


def test_acquisitions_are_measured_per_share(cfg):
    raw = make_raw(cfg, n=2)
    raw.loc[1] = raw.loc[0]
    raw.loc[1, "Ticker"] = "BUYER"
    for p in cfg.annual_years[:4]:
        raw.loc[1, f"Net Business Purchase And Sale_{p}"] = -1e8  # 100M a year on acquisitions
    out = calculate_metrics(raw, cfg).set_index("Ticker")
    plain, buyer = out.loc["T0"], out.loc["BUYER"]
    assert plain["Acquisition_Yield"] == 0.0  # statement without the row: no acquisitions
    assert buyer["Acquisition_Yield"] == pytest.approx(1e8 * 100 / buyer["marketCap"])  # no minorities
    assert buyer["FCFy_Norm"] == pytest.approx(plain["FCFy_Norm"])  # FCF does not deduct them


def test_cache_without_acquisitions_is_unknown_not_zero(cfg):
    out = calculate_metrics(make_raw(cfg, n=2), cfg)
    assert out["Acquisition_Yield"].isna().all()


def test_estimate_revisions():
    trend = pd.DataFrame({"current": [2.0, 1.8], "30daysAgo": [2.0, 1.9], "90daysAgo": [2.5, 2.0]}, index=["+1y", "0y"])
    rev = estimate_revisions(trend)
    assert rev["EPS_Rev_90d"] == pytest.approx(-20.0) and rev["EPS_Rev_30d"] == pytest.approx(0.0)
    loss = pd.DataFrame({"current": [-0.5], "30daysAgo": [-1.0], "90daysAgo": [-1.0]}, index=["0y"])
    assert estimate_revisions(loss)["EPS_Rev_90d"] == pytest.approx(50.0)  # smaller loss = upgrade
    assert estimate_revisions(pd.DataFrame()) == {"EPS_Rev_30d": None, "EPS_Rev_90d": None}


def test_dislocation_ignores_the_price_path(cfg):
    df = pd.DataFrame(
        {
            "Ticker": ["MOOD", "REASON", "CUT", "DEAR", "WEAK", "SUSPECT"],
            "Value_Score": [90.0, 90.0, 90.0, 60.0, 90.0, 90.0],
            "Quality_Score": [70.0, 70.0, 70.0, 70.0, 30.0, 70.0],
            "Revenue_Change_Recent": [3.0, -12.0, 3.0, 3.0, 3.0, 3.0],
            "EBIT_Change_Recent": [2.0, -30.0, 2.0, 2.0, 2.0, 2.0],
            "EPS_Rev_90d": [np.nan, -1.0, -18.0, 0.0, 0.0, 0.0],
            "Data_Suspect": [False, False, False, False, False, True],
        }
    )
    flag, text = dislocation(df, cfg)
    assert flag.tolist() == [True, False, False, False, False, False]
    assert "nothing in the numbers explains" in text[0] and "no analyst estimates to confirm" in text[0]
    assert "revenue -12%" in text[1] and "EBIT -30%" in text[1]
    assert "EPS estimates -18% in 90 days" in text[2] and text[3] is None  # not cheap: no sentence
    assert "Quality is below 50" in text[4]
    for momentum in (-40.0, 0.0, 60.0):  # a rising or falling price changes nothing
        moved = dislocation(df.assign(Momentum_6M=momentum, Above_52W_Low=150.0), cfg)
        assert moved[0].equals(flag) and moved[1].equals(text)
    assert not dislocation(df, replace(cfg, dislocation_min_value=95.0))[0].any()


def _cheap_and_held(n, **cols):
    base = {
        "Ticker": [f"T{i}" for i in range(n)],
        "Value_Score": 90.0,
        "Quality_Score": 70.0,
        "Revenue_Change_Recent": 3.0,
        "EBIT_Change_Recent": 2.0,
        "EPS_Rev_90d": 0.0,
    }
    return pd.DataFrame(base | cols)


def test_cyclicals_at_a_peak_are_not_mood(cfg):
    df = _cheap_and_held(2, Cyclical=[False, True], EBIT_Change_Recent=[2.0, 60.0])
    flag, text = dislocation(df, cfg)
    assert flag.tolist() == [True, False] and "top of the cycle" in text[1]


def test_mood_check_gives_the_industry_theme(cfg):
    df = _cheap_and_held(4, industry=["IT Services"] * 3 + ["Tools"])
    _, text = dislocation(df, cfg, benchmark="sp500")
    assert "implies" not in text[0]  # the price-implied growth went with the implied return (Oct 21, 2026)
    assert "3 IT Services companies in the S&P 500 benchmark are cheap" in text[0]
    assert "industry-wide" not in text[3]
    _, two = dislocation(df.iloc[1:], cfg)  # only 2 cheap IT Services companies left
    assert "industry-wide" not in two[1]


def test_mood_check_on_company_page(cfg):
    from fq.ranking import apply_dual_ranking
    from fq.report.html import build_site
    from fq.report.links import slug

    ranked = apply_dual_ranking(calculate_metrics(make_raw(cfg, n=6), cfg), cfg)
    ranked["Dislocation"] = ranked.index == 2
    ranked["Mood_Check"] = "Cheaper: price -30% in 6 months. Business held: possible mood mispricing."
    index = build_site(ranked, cfg)
    page = index.read_text(encoding="utf-8")
    assert "hl-mood" not in page and "hl-growth" not in page
    company_page = (index.parent / "companies" / f"{slug(ranked.loc[2, 'Ticker'])}.html").read_text(encoding="utf-8")
    # The front page has no Dislocation list, so the page has no such badge (Oct 31, 2026: one vocabulary); its
    # plain verdict leads the page, in the "Why it's here" strip.
    why = company_page.split('class="card why-here"')[1].split("</section>")[0]
    assert "Cheap, no obvious reason" not in company_page and "possible mood mispricing" in why


def test_highlights_explain_each_list_and_tag_the_yardstick(cfg):
    from fq.ranking import apply_dual_ranking
    from fq.report.html import build_site
    from fq.report.pills import FINANCIAL_PILL, list_pills

    ranked = apply_dual_ranking(calculate_metrics(make_raw(cfg, n=6), cfg), cfg)
    page = build_site(ranked, cfg).read_text(encoding="utf-8")
    # What every list leaves out is said once, under the Highlights heading.
    highlights = page.split("<h2>Highlights</h2>")[1].split('<div class="tabs"')[0]
    assert "Every list leaves out suspect figures" in highlights
    assert "Owner Yield" in page.split("<h3>Cheapest</h3>")[1].split("</section>")[0]
    assert "Return Quality" in page.split("<h3>Best businesses</h3>")[1].split("</section>")[0]

    bank = pd.Series({"Financial_Mode": True, "Business_Model": "bank", "Cyclical": True})
    assert [p["label"] for p in list_pills(bank)] == ["bank", "cyclical"] and list_pills(bank)[0][
        "why"
    ] == FINANCIAL_PILL
    assert list_pills(pd.Series({"Financial_Mode": False, "Business_Model": "operating"})) == []
