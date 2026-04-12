# Purpose: intent registry (descriptions + keywords) used for routing only.
# Extensibility point: adding a new API intent is primarily adding a new IntentSpec here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .intents import APOD, DOCUMENT_SEARCH, DONKI, EONET, NEO, UNKNOWN


@dataclass(frozen=True)
class IntentSpec:
    """
    Intent metadata used ONLY for routing decisions.
    """

    label: str
    short_description: str

    # Keyword/phrase hints for deterministic routing (rule-based).
    # Keep these short, human-auditable, and easy to extend.
    keywords: List[str]


def default_intent_specs() -> Dict[str, IntentSpec]:
    """
    Central registry: add new APIs by adding another IntentSpec entry.
    Core routing logic should not need changes.
    """

    return {
        DOCUMENT_SEARCH: IntentSpec(
            label=DOCUMENT_SEARCH,
            short_description="Use local documents / indexed knowledge base.",
            keywords=[
                "explain",
                "what is",
                "what are",
                "how does",
                "how do",
                "how can",
                "difference between",
                "compare",
                "vs",
                "definition",
                "causes",
                "why",
                "derive",
                "proof",
                "example",
                "concept",
                "theory",
                "tell me",
                "describe",
                "summarize",
                "overview",
                "meaning",
                "relationship",
                "properties",
            ],
        ),
        APOD: IntentSpec(
            label=APOD,
            short_description="Astronomy Picture of the Day (APOD): daily astronomy image + explanation.",
            keywords=[
                "apod",
                "astronomy picture of the day",
                "picture of the day",
                "image of the day",
                "nasa photo of the day",
                "daily astronomy image",
                "today's apod",
                "space photo", 
                "nasa image", 
                "today's space picture",
                "astronomy photo",
            ],
        ),
        NEO: IntentSpec(
            label=NEO,
            short_description="Near-Earth Objects (NEO): asteroids/comets close approaches and related data.",
            keywords=[
                "neo",
                "near earth object",
                "near-earth object",
                "asteroid",
                "asteroids",
                "close approach",
                "closest approach",
                "potentially hazardous",
                "pha",
                "impact risk",
                "orbit",
                "space rock",
                "meteor",
                "meteorite",
                "asteroid passing",
                "object near earth",
                "dangerous asteroid",
            ],
        ),
        DONKI: IntentSpec(
            label=DONKI,
            short_description="DONKI: space weather events (solar flares, CMEs, geomagnetic storms, etc.).",
            keywords=[
                "donki",
                "space weather",
                "solar flare",
                "solar flares",
                "cme",
                "coronal mass ejection",
                "geomagnetic storm",
                "kp index",
                "radiation storm",
                "sep",
                "interplanetary shock",
            ],
        ),
        EONET: IntentSpec(
            label=EONET,
            short_description="EONET: natural Earth events (wildfires, storms, earthquakes, volcanoes, floods).",
            keywords=[
                "eonet",
                "wildfire",
                "wildfires",
                "earthquake",
                "earthquakes",
                "volcano",
                "volcanic",
                "storm",
                "storms",
                "flood",
                "floods",
                "drought",
                "hurricane",
                "cyclone",
                "natural event",
                "disaster",
            ],
        ),
        UNKNOWN: IntentSpec(
            label=UNKNOWN,
            short_description="Intent unclear. Downstream can ask clarification or default safely.",
            keywords=[],
        ),
    }

