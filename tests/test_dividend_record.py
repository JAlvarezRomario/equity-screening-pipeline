"""The Unbroken dividends record (`report.dividend_record`): specials, payment timing, gaps and the declared currency;
and the payments the fetch stores (`fetch.dividend_payments`, `fetch.declared_currency_payments`)."""

import json

import numpy as np
import pandas as pd

from fq import fx
from fq.fetch import declared_currency_payments, dividend_payments
from fq.report.dividend_record import JUDGED_YEARS, dividend_record, judge, recessions, regular_payments

LAST = pd.Timestamp.now().year - 1  # the last complete year; no calendar year written in a test
FIRST = LAST - 14  # a 15-year record


def _quarterly(amounts: dict[int, float], months=(2, 5, 8, 11)) -> pd.Series:
    """One payment per month of `months` each year, the year's amount split evenly."""
    days = {pd.Timestamp(y, m, 10): a / len(months) for y, a in amounts.items() for m in months}
    return pd.Series(days, dtype=float).sort_index()


def _rising(step: float = 0.05) -> dict[int, float]:
    return {y: 1.0 * (1 + step) ** (y - FIRST) for y in range(FIRST, LAST + 1)}


def test_a_steady_raiser_is_unbroken_and_its_growth_is_the_regular_dividends():
    out = judge(_quarterly(_rising()), LAST)
    assert out["Dividend_Record_Whole"] and out["Dividend_Unbroken_Years"] == 15
    assert abs(out["Regular_DPS_Growth_10Y"] - 5.0) < 0.01


def test_an_extra_special_payment_is_dropped_and_the_next_year_is_no_cut():
    pay = _quarterly(_rising())
    special = pd.Series({pd.Timestamp(LAST - 5, 12, 15): 10.0})
    pay = pd.concat([pay, special]).sort_index()
    regular = regular_payments(pay)
    assert pd.Timestamp(LAST - 5, 12, 15) not in regular.index and len(regular) == len(pay) - 1
    out = judge(pay, LAST)
    assert out["Dividend_Record_Whole"] and abs(out["Regular_DPS_Growth_10Y"] - 5.0) < 0.01


def test_a_special_inside_an_annual_payment_is_capped():
    amounts = {y: 2.0 for y in range(FIRST, LAST + 1)} | {LAST - 6: 3.7}  # one year paid a special with the dividend
    pay = pd.Series({pd.Timestamp(y, 6, 1): a for y, a in amounts.items()})
    assert regular_payments(pay)[pd.Timestamp(LAST - 6, 6, 1)] == 2.0
    assert judge(pay, LAST)["Dividend_Record_Whole"]


def test_a_raise_cut_back_with_the_same_payments_is_a_cut():
    amounts = _rising() | {LAST - 3: _rising()[LAST - 4] * 0.9}  # 10% below the year before, four payments as usual
    out = judge(_quarterly(amounts), LAST)
    assert not out["Dividend_Record_Whole"] and out["Dividend_Unbroken_Years"] == 3


def test_a_cut_older_than_the_judged_years_no_longer_breaks_the_record():
    start = LAST - JUDGED_YEARS - 4
    amounts = {y: 1.0 * 1.03 ** (y - start) for y in range(start, LAST + 1)}
    old = amounts | {
        LAST - JUDGED_YEARS: amounts[LAST - JUDGED_YEARS - 1] * 0.25
    }  # a lone wrong payment, 30 years back
    out = judge(_quarterly(old), LAST)
    assert out["Dividend_Record_Whole"] and out["Dividend_Unbroken_Years"] == JUDGED_YEARS  # still counted honestly
    recent = amounts | {LAST - JUDGED_YEARS + 1: amounts[LAST - JUDGED_YEARS] * 0.25}  # one year inside the window
    assert not judge(_quarterly(recent), LAST)["Dividend_Record_Whole"]


def test_a_payment_slipping_into_january_is_no_cut():
    pay = _quarterly(_rising())
    late = pd.Timestamp(LAST - 4, 11, 10)
    pay = pay.rename(index={late: pd.Timestamp(LAST - 3, 1, 5)}).sort_index()  # 3 payments, then 5
    assert judge(pay, LAST)["Dividend_Record_Whole"]


def test_an_added_interim_that_front_loads_a_year_is_no_cut():
    amounts = {y: 10.0 + (y - FIRST) * 0.5 for y in range(FIRST, LAST + 1)}
    pay = pd.Series({pd.Timestamp(y, 6, 1): a for y, a in amounts.items()})
    # From two years ago an interim joins the final (a year of 16 + 8: 1.5 years' worth); then two halves of 8.5, 9.
    pay[pd.Timestamp(LAST - 2, 11, 1)] = 8.0
    pay[pd.Timestamp(LAST - 1, 6, 1)], pay[pd.Timestamp(LAST - 1, 11, 1)] = 8.5, 8.5
    pay[pd.Timestamp(LAST, 6, 1)], pay[pd.Timestamp(LAST, 11, 1)] = 9.0, 9.0
    assert judge(pay.sort_index(), LAST)["Dividend_Record_Whole"]
    # The same pattern with the dividend then cut by a fifth is a cut.
    pay[pd.Timestamp(LAST - 1, 6, 1)], pay[pd.Timestamp(LAST - 1, 11, 1)] = 6.0, 6.0
    assert not judge(pay.sort_index(), LAST)["Dividend_Record_Whole"]


def test_a_year_without_a_dividend_breaks_the_record():
    amounts = {y: a for y, a in _rising().items() if y != LAST - 9}
    out = judge(_quarterly(amounts), LAST)
    assert not out["Dividend_Record_Whole"] and out["Dividend_Unbroken_Years"] == 9


def test_the_current_year_is_not_judged_and_no_payments_is_blank():
    pay = pd.concat([_quarterly(_rising()), pd.Series({pd.Timestamp(LAST + 1, 2, 10): 0.01})])
    assert judge(pay, LAST)["Dividend_Record_Whole"]  # a part year would read as a cut
    assert np.isnan(judge(pd.Series(dtype=float, index=pd.DatetimeIndex([])), LAST)["Dividend_Unbroken_Years"])


def test_the_declared_currency_record_wins_over_the_quoted_one():
    held = {y: 1.0 for y in range(FIRST, LAST + 1)}
    declared = json.dumps({f"{d:%Y-%m-%d}": v for d, v in _quarterly(held).items()})
    fx_noise = {y: 1.0 - 0.08 * (y % 2) for y in range(FIRST, LAST + 1)}  # the quote currency swings 8%
    quoted = json.dumps({f"{d:%Y-%m-%d}": v for d, v in _quarterly(fx_noise).items()})
    df = pd.DataFrame({"DPS_Payments": [quoted, quoted], "DPS_Payments_FX": [declared, None],
                       "Fetched_At": [f"{LAST + 1}-03-01 10:00"] * 2})  # fmt: skip
    out = dividend_record(df)
    assert out["Dividend_Record_Whole"].tolist() == [True, False]
    assert out["Dividend_Unbroken_Years"].iloc[0] == 15


def test_the_fetch_stores_each_payment_and_converts_to_the_declared_currency(monkeypatch):
    divs = pd.Series([0.5, 0.0, 0.6], index=pd.to_datetime([f"{FIRST}-03-01", f"{FIRST}-06-01", f"{FIRST + 1}-03-01"]))
    pay = dividend_payments(divs)
    assert pay == {f"{FIRST}-03-01": 0.5, f"{FIRST + 1}-03-01": 0.6}
    rates = {("HKD", f"{FIRST + 1}-03-01"): 0.128, ("CNY", f"{FIRST + 1}-03-01"): 0.138}
    monkeypatch.setattr(fx, "rate_on", lambda c, d: rates.get((c, d)))
    out = declared_currency_payments(pay, "HKD", "CNY")
    # The older payment predates the rates: it takes the oldest rate known, so the record stays in one currency.
    assert out == {k: v * 0.128 / 0.138 for k, v in pay.items()}
    assert declared_currency_payments(pay, "GBp", "GBP") is None  # pence and pounds: one currency
    assert declared_currency_payments(pay, "USD", "USD") is None


# This year's payments (rule 6) and the record through recessions (rule 7). Recession years come from the list file.
GFC, PANDEMIC = (y for y, _, only in recessions() if not only)
(OIL,) = (y for y, _, only in recessions() if only)


def test_a_cut_paid_this_year_breaks_the_record_now():
    pay = _quarterly(_rising())
    as_of = pd.Timestamp(LAST + 1, 9, 30)
    this_year = pd.Series({pd.Timestamp(LAST + 1, m, 10): _rising()[LAST] / 4 * 0.8 for m in (2, 5, 8)})
    cut = judge(pd.concat([pay, this_year]), LAST, as_of)
    assert cut["Dividend_Cut_Recent"] and not cut["Dividend_Record_Whole"] and cut["Dividend_Unbroken_Years"] == 0
    raised = this_year * 1.05 / 0.8
    assert not judge(pd.concat([pay, raised]), LAST, as_of)["Dividend_Cut_Recent"]
    # Only two of this year's three payments made by the fetch: the windows hold 2 and 3 payments, so not judged.
    late = this_year.iloc[:2] * 0.5
    assert judge(pd.concat([pay, late]), LAST, as_of)["Dividend_Record_Whole"]


def _record(first: int, amounts: dict[int, float] | None = None) -> pd.Series:
    base = {y: 1.0 + 0.03 * (y - first) for y in range(first, LAST + 1)}
    return _quarterly(base | (amounts or {}))


def test_a_recession_cut_is_measured_and_named():
    first = GFC - 6
    held = judge(_record(first), LAST, model="operating")
    assert held["Recession_Dividend_Fall"] == 0 and held["Recessions_Covered"] == held["Recessions_Due"] == 2
    assert held["Recession_Dividend_Note"] == f"{GFC} held; {PANDEMIC} held"
    level = 1.0 + 0.03 * (GFC - 1 - first)
    halved = {GFC: level / 2, GFC + 1: level / 2, GFC + 2: level / 2}
    cut = judge(_record(first, halved), LAST, model="operating")
    assert -55 < cut["Recession_Dividend_Fall"] < -45 and cut["Recession_Dividend_Note"].startswith(f"{GFC} \u2212")


def test_a_payment_slipping_across_the_recession_year_end_is_no_fall():
    pay = _record(GFC - 6)
    pay = pay.rename(index={pd.Timestamp(GFC, 11, 10): pd.Timestamp(GFC + 1, 1, 5)}).sort_index()
    assert judge(pay, LAST, model="operating")["Recession_Dividend_Fall"] == 0


def test_banks_are_not_judged_on_the_pandemic_and_a_short_record_misses_the_crisis():
    suspended = {PANDEMIC: 0.01, PANDEMIC + 1: 0.5}
    bank = judge(_record(GFC - 6, suspended), LAST, model="bank")
    assert bank["Recessions_Due"] == 1 and bank["Recession_Dividend_Note"] == f"{GFC} held"
    assert judge(_record(GFC - 6, suspended), LAST, model="operating")["Recession_Dividend_Fall"] < -25
    young = judge(_record(GFC + 1), LAST, model="operating")
    assert young["Recessions_Covered"] == 1 < young["Recessions_Due"]


def test_the_oil_downturn_judges_energy_companies_only():
    # User's choice, Nov 5, 2026: the 2015 oil price collapse, for the Energy sector alone.
    first = GFC - 6
    level = 1.0 + 0.03 * (OIL - 1 - first)
    halved = {OIL: level / 2, OIL + 1: level / 2, OIL + 2: level / 2}
    energy = judge(_record(first, halved), LAST, model="operating", sector="Energy")
    assert energy["Recessions_Due"] == 3 and -55 < energy["Recession_Dividend_Fall"] < -45
    assert f"{OIL} −" in energy["Recession_Dividend_Note"]
    utility = judge(_record(first, halved), LAST, model="asset_heavy", sector="Utilities")
    assert utility["Recessions_Due"] == 2 and str(OIL) not in utility["Recession_Dividend_Note"]


def test_the_company_page_shows_the_record_through_recessions():
    from fq.report.tiles import key_tiles

    tiles = key_tiles(pd.Series({"Recession_Dividend_Note": f"{GFC} −40%", "Recession_Dividend_Fall": -40.0}))
    tile = next(t for t in tiles if t["label"] == "Dividend through recessions")
    assert tile["value"] == f"{GFC} −40%" and tile["cls"] == "neg" and "recession" in tile["help"]
    assert not any(t["label"] == "Dividend through recessions" for t in key_tiles(pd.Series({"Dividend_Yield": 0.0})))
