from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def tavily_search(
    query: str,
    *,
    api_key: str,
    search_depth: str = "basic",
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Execute a Tavily search request.

    This keeps the integration lightweight by using the standard library.
    """
    if not api_key.strip():
        raise ValueError("missing Tavily API key")

    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        TAVILY_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45.0) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tavily search failed: {exc.code} {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Tavily search connection failed: {exc}") from exc

    return json.loads(body)
