# Purpose: converts router output (intent) into an execution plan.
# Downstream agents (e.g., answer/verify) can then execute the plan.
# This keeps “routing” separate from “doing”.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .intents import API_INTENTS, DOCUMENT_SEARCH, UNKNOWN


@dataclass(frozen=True)
class DispatchPlan:
    """
    A minimal, downstream-friendly interface.
    Router decides the label; dispatcher converts it into an execution plan.
    """

    intent: str
    should_call_api: bool
    api_name: Optional[str] = None


def build_dispatch_plan(intent: str) -> DispatchPlan:
    if intent in API_INTENTS:
        return DispatchPlan(intent=intent, should_call_api=True, api_name=intent)
    if intent == DOCUMENT_SEARCH:
        return DispatchPlan(intent=intent, should_call_api=False, api_name=None)
    return DispatchPlan(intent=UNKNOWN, should_call_api=False, api_name=None)

