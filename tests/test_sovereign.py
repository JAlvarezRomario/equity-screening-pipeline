"""The Boring businesses bond hurdle with the home country's sovereign dollar spread (user's choice, Nov 7, 2026)."""

import pandas as pd
import pytest

from fq.universe import _sovereign_spread_table, industry_key, sovereign_spread


def test_the_list_reads_and_spreads_are_counted_beyond_the_us():
    table = _sovereign_spread_table()
    assert len(table) > 150 and table[industry_key("United States")] > 0
    assert sovereign_spread("Germany") == 0.0  # Aaa: below the US's own spread, never negative
    assert sovereign_spread("Brazil") == pytest.approx(
        table[industry_key("Brazil")] - table[industry_key("United States")]
    )
    assert sovereign_spread("United States") == 0.0
    assert sovereign_spread("Monaco") == 0.0 and sovereign_spread(None) == 0.0  # not listed or unknown: as the US


def test_a_company_must_beat_its_own_governments_dollar_borrowing_cost():
    from fq.report.screens import _shortlist_gates

    df = pd.DataFrame({"Ticker": ["US", "BR", "UNKNOWN"], "Owner_Yield": 7.0, "Sovereign_Spread": [0.0, 3.0, None]})
    assert dict(_shortlist_gates(df, 6.0))["beats the bond"].tolist() == [True, False, True]
    assert dict(_shortlist_gates(df.drop(columns="Sovereign_Spread"), 6.0))["beats the bond"].all()


def test_prepare_attaches_the_spread_by_country(cfg):
    from fq.report.highlights import prepare

    df = pd.DataFrame({"Ticker": ["A", "B"], "country": ["Brazil", "Germany"], "Rank": [1.0, 2.0]})
    out = prepare(df, None)
    assert out["Sovereign_Spread"].tolist() == [sovereign_spread("Brazil"), 0.0]
