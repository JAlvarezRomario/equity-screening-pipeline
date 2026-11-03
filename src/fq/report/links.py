"""The single source of truth for file slugs and external hyperlinks."""

import re
from urllib.parse import quote


def slug(ticker: str) -> str:
    """Filesystem- and URL-safe form of a ticker, used for page names and links."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker)


def external_links(ticker: str) -> list[tuple[str, str]]:
    """(label, url) pairs shown on every company page."""
    t = quote(ticker, safe="")
    return [
        ("Dataroma", f"https://www.dataroma.com/m/stock.php?sym={t}"),
        ("Yahoo Finance", f"https://finance.yahoo.com/quote/{t}/cash-flow/"),
        ("ROIC.ai", f"https://www.roic.ai/quote/{t}"),
    ]
