"""Turn URLs, files and stdin into clean plain text."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MAX_DOWNLOAD_BYTES = 2_000_000
_SKIP_TAGS = {"script", "style", "noscript"}
_BLOCK_TAGS = {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4"}


class SourceError(ValueError):
    """Input text could not be obtained."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    """Visible text of an HTML page, one block element per line."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def fetch_url(url: str, timeout: float = 30) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SourceError("URL must be a valid http:// or https:// address.")

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; wm-rag/1.0)",
            "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(MAX_DOWNLOAD_BYTES)
    except HTTPError as exc:
        raise SourceError(f"Could not fetch URL: HTTP {exc.code}") from exc
    except URLError as exc:
        raise SourceError(f"Could not fetch URL: {exc.reason}") from exc

    text = body.decode(charset, errors="replace")
    if "html" in content_type.lower() or "<html" in text[:500].lower():
        text = html_to_text(text)
    return _require_text(text, "Fetched the URL but found no readable text.")


def read_file(path: str) -> str:
    """Read a UTF-8 text file, or stdin when ``path`` is ``-``."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return _require_text(text, f"{path} is empty.")


def _require_text(text: str, message: str) -> str:
    text = text.strip()
    if not text:
        raise SourceError(message)
    return text
