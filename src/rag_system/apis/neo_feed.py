# NASA Neo Feed API
# 

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

NEO_FEED_ENDPOINT = "https://api.nasa.gov/neo/rest/v1/feed"

_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

_MONTH_DAY_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+(\d{1,2})(?:,\s*(\d{4}))?\b",
    re.IGNORECASE,
)


def _nasa_api_key() -> str:
    key = os.environ.get("NASA_API_KEY", "").strip()
    if not key:
        raise ValueError("NASA_API_KEY is missing. Add it to .env.")
    return key


def _parse_single_date_from_query(query: str, *, today: date) -> Optional[date]:
    q = (query or "").strip().lower()

    if "today" in q:
        return today
    if "yesterday" in q:
        return today - timedelta(days=1)
    if "tomorrow" in q:
        return today + timedelta(days=1)

    iso = _ISO_DATE_RE.search(q)
    if iso:
        return datetime.strptime(iso.group(1), "%Y-%m-%d").date()

    md = _MONTH_DAY_RE.search(q)
    if md:
        month_name = md.group(1)
        day_num = int(md.group(2))
        year_num = int(md.group(3)) if md.group(3) else today.year
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(f"{month_name} {day_num} {year_num}", fmt).date()
            except ValueError:
                continue
    return None


def parse_neo_feed_date_range(query: str, *, today: Optional[date] = None) -> Tuple[str, str]:
    """
    Deterministic date parsing for NEO feed.
    Returns (start_date, end_date) in YYYY-MM-DD format, with max window <= 7 days.
    """
    base = today or date.today()
    q = (query or "").strip().lower()

    if "next week" in q or "this week" in q:
        start = base
        end = base + timedelta(days=7)
    elif "last week" in q or "previous week" in q:
        start = base - timedelta(days=7)
        end = base
    else:
        single = _parse_single_date_from_query(q, today=base)
        if single is not None:
            start = single
            end = single
        else:
            start = base
            end = base

    if end < start:
        start, end = end, start
    if (end - start).days > 7:
        end = start + timedelta(days=7)

    return start.isoformat(), end.isoformat()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}{suffix}"


def fetch_neo_feed(start_date: str, end_date: str, *, timeout_s: float = 25.0) -> Dict[str, Any]:
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": _nasa_api_key(),
    }
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{NEO_FEED_ENDPOINT}?{qs}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"NeoWs feed HTTP {e.code}: {err_body or e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"NeoWs feed request failed: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError("NeoWs feed returned unexpected payload shape.")
    return data


def process_neo_feed_payload(raw: Dict[str, Any], *, start_date: str, end_date: str) -> Dict[str, Any]:
    neo_by_date = raw.get("near_earth_objects") or {}
    if not isinstance(neo_by_date, dict):
        neo_by_date = {}

    flattened: List[Dict[str, Any]] = []
    for _, items in neo_by_date.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            diam = item.get("estimated_diameter", {}).get("meters", {})
            min_d = _to_float(diam.get("estimated_diameter_min"))
            max_d = _to_float(diam.get("estimated_diameter_max"))
            cad = item.get("close_approach_data") or []
            cad0 = cad[0] if cad and isinstance(cad[0], dict) else {}
            velocity = _to_float((cad0.get("relative_velocity") or {}).get("kilometers_per_hour"))
            miss_distance = _to_float((cad0.get("miss_distance") or {}).get("kilometers"))
            hazardous = bool(item.get("is_potentially_hazardous_asteroid", False))

            flattened.append(
                {
                    "id": str(item.get("id") or ""),
                    "name": str(item.get("name") or "Unknown"),
                    "is_potentially_hazardous_asteroid": hazardous,
                    "estimated_diameter_meters_min": round(min_d, 2) if min_d is not None else None,
                    "estimated_diameter_meters_max": round(max_d, 2) if max_d is not None else None,
                    "relative_velocity_kph": round(velocity, 2) if velocity is not None else None,
                    "miss_distance_km": round(miss_distance, 2) if miss_distance is not None else None,
                    "diameter_display": f"{_format_number(min_d)} - {_format_number(max_d)} m",
                    "velocity_display": _format_number(velocity, " km/h"),
                    "miss_distance_display": _format_number(miss_distance, " km"),
                }
            )

    asteroid_count = len(flattened)
    hazardous_count = sum(1 for a in flattened if a["is_potentially_hazardous_asteroid"])
    safe_count = asteroid_count - hazardous_count

    flattened.sort(
        key=lambda a: (
            0 if a["is_potentially_hazardous_asteroid"] else 1,
            a["miss_distance_km"] if a["miss_distance_km"] is not None else float("inf"),
        )
    )
    top_5 = flattened[:5]

    return {
        "endpoint": "neo_feed",
        "start_date": start_date,
        "end_date": end_date,
        "asteroid_count": asteroid_count,
        "hazardous_count": hazardous_count,
        "safe_count": safe_count,
        "asteroids": top_5,
    }


def fetch_and_process_neo_feed(query: str) -> Dict[str, Any]:
    start_date, end_date = parse_neo_feed_date_range(query)
    raw = fetch_neo_feed(start_date, end_date)
    return process_neo_feed_payload(raw, start_date=start_date, end_date=end_date)

