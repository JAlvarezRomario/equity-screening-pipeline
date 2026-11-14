"""Offline tests for the auxiliary Scrapy index-constituent crawler (`fq.crawlers`)."""

import pytest
from scrapy.http import HtmlResponse

from fq.crawlers import (
    IndexConstituentsSpider,
    IndexSpec,
    LayoutChangeError,
    compare_with_list,
    normalise_ticker,
    parse_constituent_table,
)

SAMPLE_SP500_HTML = """
<html><body>
<table class="wikitable">
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
  <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
  <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
  <tr><td>GOOGL</td><td>Alphabet Inc. (Class A)</td><td>Communication Services</td></tr>
  <tr><td>GOOG</td><td>Alphabet Inc. (Class C)</td><td>Communication Services</td></tr>
  <tr><td>MSFT [1]</td><td>Microsoft Corporation [a]</td><td>Information Technology</td></tr>
</table>
</body></html>
"""

SAMPLE_IBEX_HTML = """
<html><body>
<table>
  <tr><th>Ticker</th><th>Company</th><th>Sector</th></tr>
  <tr><td>SAN</td><td>Banco Santander</td><td>Banking</td></tr>
  <tr><td>ITX.MC</td><td>Inditex</td><td>Retail</td></tr>
  <tr><td>MTS</td><td>ArcelorMittal</td><td>Steel</td></tr>
  <tr><td>IBE</td><td>Iberdrola</td><td>Utilities</td></tr>
</table>
</body></html>
"""

BROKEN_HEADER_HTML = """
<html><body>
<table>
  <tr><th>ISIN</th><th>Weight (%)</th></tr>
  <tr><td>US0378331005</td><td>7.1</td></tr>
</table>
</body></html>
"""


def _html_response(url: str, html: str) -> HtmlResponse:
    return HtmlResponse(url=url, body=html.encode("utf-8"), encoding="utf-8")


def test_normalise_ticker_handles_dots_footnotes_and_exchange_suffixes():
    sp_spec = IndexSpec(name="sp500", url="https://example.com/sp500", min_rows=1, max_rows=10, dot_to_dash=True)
    assert normalise_ticker("BRK.B [note 1]", sp_spec) == "BRK-B"
    assert normalise_ticker("BF.B", sp_spec) == "BF-B"

    ibex_spec = IndexSpec(name="ibex35", url="https://example.com/ibex", min_rows=1, max_rows=10, suffix=".MC")
    assert normalise_ticker("SAN", ibex_spec) == "SAN.MC"
    assert normalise_ticker("ITX.MC", ibex_spec) == "ITX.MC"


def test_parse_constituent_table_extracts_rows_and_filters_ignored_share_classes():
    spec = IndexSpec(
        name="sp500",
        url="https://example.com/sp500",
        min_rows=3,
        max_rows=5,
        dot_to_dash=True,
        ignored_tickers=frozenset({"GOOG"}),
    )
    records = parse_constituent_table(SAMPLE_SP500_HTML, spec)
    assert [r.ticker for r in records] == ["AAPL", "BRK-B", "GOOGL", "MSFT"]
    assert records[1].company == "Berkshire Hathaway"
    assert records[3].company == "Microsoft Corporation"
    assert records[0].sector == "Information Technology"


def test_parse_constituent_table_raises_on_missing_columns():
    spec = IndexSpec(name="sp500", url="https://example.com/sp500", min_rows=1, max_rows=10)
    with pytest.raises(LayoutChangeError, match="no HTML table with required ticker/company columns"):
        parse_constituent_table(BROKEN_HEADER_HTML, spec)


def test_parse_constituent_table_raises_when_row_count_out_of_bounds():
    spec = IndexSpec(name="sp500", url="https://example.com/sp500", min_rows=495, max_rows=510, dot_to_dash=True)
    with pytest.raises(LayoutChangeError, match="outside expected bounds"):
        parse_constituent_table(SAMPLE_SP500_HTML, spec)


def test_compare_with_list_reports_added_and_removed_constituents():
    spec = IndexSpec(
        name="ibex35",
        url="https://example.com/ibex",
        min_rows=2,
        max_rows=5,
        suffix=".MC",
        ignored_tickers=frozenset({"MTS.MC"}),
    )
    records = parse_constituent_table(SAMPLE_IBEX_HTML, spec)
    report = compare_with_list(spec, records, packaged_tickers={"SAN.MC", "ITX.MC", "TEF.MC"})
    assert report.layout_ok is True
    assert report.crawled_count == 3
    assert report.list_count == 3
    assert report.added == ["IBE.MC"]
    assert report.removed == ["TEF.MC"]


def test_spider_evaluate_response_handles_valid_and_broken_pages():
    spec = IndexSpec(name="sp500", url="https://example.com/sp500", min_rows=3, max_rows=5, dot_to_dash=True)
    ok_resp = _html_response(spec.url, SAMPLE_SP500_HTML)
    report = IndexConstituentsSpider.evaluate_response(
        ok_resp, spec, packaged_tickers={"AAPL", "BRK-B", "GOOGL", "NVDA"}
    )
    assert report.layout_ok is True
    assert report.added == ["GOOG", "MSFT"]
    assert report.removed == ["NVDA"]

    bad_resp = _html_response(spec.url, BROKEN_HEADER_HTML)
    bad_report = IndexConstituentsSpider.evaluate_response(bad_resp, spec, packaged_tickers={"AAPL"})
    assert bad_report.layout_ok is False
    assert bad_report.error is not None
    assert "no HTML table" in bad_report.error


def test_spider_rejects_unknown_index_spec():
    with pytest.raises(ValueError, match="Unknown index spec"):
        IndexConstituentsSpider(indices=["not_an_index"])
