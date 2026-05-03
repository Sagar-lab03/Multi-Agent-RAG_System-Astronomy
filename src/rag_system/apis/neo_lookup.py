"""
NASA NeoWs Lookup API: GET https://api.nasa.gov/neo/rest/v1/neo/{asteroid_id}
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

NEO_LOOKUP_BASE = "https://api.nasa.gov/neo/rest/v1/neo"
_ID_FROM_QUERY_RE = re.compile(r"\b(\d{5,})\b")


def _nasa_api_key() -> str:
    key = os.environ.get("NASA_API_KEY", "").strip()
    if not key:
        raise ValueError("NASA_API_KEY is missing. Add it to .env.")
    return key


def extract_asteroid_id_from_query(query: Optional[str]) -> Optional[str]:
    """Extract first 5+ digit ID from plain text."""
    if not query:
        return None
    m = _ID_FROM_QUERY_RE.search(query)
    return m.group(1) if m else None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:,.2f}{suffix}"


def _parse_cad_sort_key(close_approach: Dict[str, Any]) -> str:
    d = (
        close_approach.get("close_approach_date_full")
        or close_approach.get("close_approach_date")
        or ""
    )
    return str(d)


def _latest_close_approach(cads: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(cads, list) or not cads:
        return None
    # Lexicographically max ISO-ish date string chooses latest for NASA's format YYYY-MM-DD...
    dated: List[tuple[str, Dict[str, Any]]] = []
    for cad in cads:
        if not isinstance(cad, dict):
            continue
        dated.append((_parse_cad_sort_key(cad), cad))
    if not dated:
        return None
    dated.sort(key=lambda t: t[0], reverse=True)
    return dated[0][1]


def fetch_neo_lookup(asteroid_id: str, *, timeout_s: float = 25.0) -> Dict[str, Any]:
    aid = (asteroid_id or "").strip()
    if not aid:
        raise ValueError("Missing asteroid ID.")
    path = urllib.parse.quote(aid, safe="")
    params = urllib.parse.urlencode({"api_key": _nasa_api_key()})
    req = urllib.request.Request(f"{NEO_LOOKUP_BASE}/{path}?{params}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"NeoWs lookup HTTP {e.code}: {err_body or e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"NeoWs lookup request failed: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError("NeoWs lookup returned unexpected payload shape.")
    return data


def process_neo_lookup_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    orbital = raw.get("orbital_data") or {}
    if not isinstance(orbital, dict):
        orbital = {}

    cad = raw.get("close_approach_data") or []
    cad0 = _latest_close_approach(cad)

    diam = {}
    diam_src = raw.get("estimated_diameter") or {}
    if isinstance(diam_src, dict) and isinstance(diam_src.get("meters"), dict):
        diam = diam_src["meters"]
    elif isinstance(diam_src, dict):
        diam = {}

    est_min = _to_float(diam.get("estimated_diameter_min"))
    est_max = _to_float(diam.get("estimated_diameter_max"))

    velocity_kmh: Optional[float] = None
    miss_km: Optional[float] = None
    cad_date_display = ""

    if isinstance(cad0, dict):
        vel = cad0.get("relative_velocity") or {}
        ms = cad0.get("miss_distance") or {}
        if isinstance(vel, dict):
            velocity_kmh = _to_float(vel.get("kilometers_per_hour"))
        if isinstance(ms, dict):
            miss_km = _to_float(ms.get("kilometers"))
        cad_date_display = (
            str(cad0.get("close_approach_date_full") or cad0.get("close_approach_date") or "")
        )

    return {
        "endpoint": "neo_lookup",
        "id": str(raw.get("neo_reference_id") or raw.get("id") or ""),
        "name": str(raw.get("name") or "(unknown name)"),
        "nasa_jpl_url": str(raw.get("nasa_jpl_url") or ""),
        "is_potentially_hazardous_asteroid": bool(
            raw.get("is_potentially_hazardous_asteroid", False)
        ),
        "estimated_diameter_meters_min": round(est_min, 2) if est_min is not None else None,
        "estimated_diameter_meters_max": round(est_max, 2) if est_max is not None else None,
        "diameter_display": f"{_format_number(est_min)} - {_format_number(est_max)} m",
        "relative_velocity_kph": round(velocity_kmh, 2) if velocity_kmh is not None else None,
        "miss_distance_km": round(miss_km, 2) if miss_km is not None else None,
        "velocity_display": _format_number(velocity_kmh, " km/h"),
        "miss_distance_display": _format_number(miss_km, " km"),
        "orbit_class": str(orbital.get("orbit_class") or orbital.get("class") or ""),
        "orbiting_body": str(
            orbital.get("orbiting_body") or orbital.get("body") or orbital.get("orb_body") or ""
        ),
        "close_approach_date": cad_date_display,
    }


def fetch_and_process_neo_lookup(asteroid_id: str) -> Dict[str, Any]:
    raw = fetch_neo_lookup(asteroid_id)
    out = process_neo_lookup_payload(raw)
    out["id"] = (asteroid_id or "").strip() or out.get("id") or ""
    return out
