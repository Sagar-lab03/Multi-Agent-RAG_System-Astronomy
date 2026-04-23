from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

APOD_ENDPOINT = "https://api.nasa.gov/planetary/apod"
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


@dataclass(frozen=True)
class ApodRequest:
    api_key: str
    date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    count: Optional[int] = None
    thumbs: bool = False

    def to_query_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {"api_key": self.api_key, "thumbs": str(self.thumbs).lower()}
        if self.date:
            params["date"] = self.date
        if self.start_date:
            params["start_date"] = self.start_date
        if self.end_date:
            params["end_date"] = self.end_date
        if self.count is not None:
            params["count"] = str(self.count)
        return params


def extract_date_from_query(query: str) -> Optional[str]:
    m = _ISO_DATE_RE.search(query or "")
    if not m:
        return None
    return m.group(1)


def _nasa_api_key() -> str:
    key = os.environ.get("NASA_API_KEY", "").strip()
    if not key:
        raise ValueError("NASA_API_KEY is missing. Add it to .env.")
    return key


def fetch_apod(*, date: Optional[str] = None, timeout_s: float = 20.0) -> Dict[str, Any]:
    req_cfg = ApodRequest(api_key=_nasa_api_key(), date=date, thumbs=False)
    qs = urllib.parse.urlencode(req_cfg.to_query_params())
    url = f"{APOD_ENDPOINT}?{qs}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"APOD API HTTP {e.code}: {err_body or e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"APOD API request failed: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError("APOD API returned unexpected payload shape (expected object).")
    return data

