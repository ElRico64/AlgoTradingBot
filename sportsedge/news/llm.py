"""Optional Claude-powered news analyst.

Sends batches of headlines / injury blurbs to Claude and gets back
schema-validated JSON (structured outputs), which is converted to the same
`NewsEvent`s the rule-based parser produces. Requires `pip install anthropic`
and credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).

The model only *extracts* facts (who, which team, what status, how reliable
the report sounds, how important the player is). All probability math stays
in the transparent statistical pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Sequence

from ..teams import REGISTRY
from ..types import NewsEvent

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

IMPACT_TIERS = {"superstar": 1.6, "star": 1.0, "starter": 0.35, "role": 0.12, "unknown": None}

SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_index": {"type": "integer"},
                    "team_abbr": {"type": "string"},
                    "player": {"type": "string"},
                    "event_type": {"type": "string",
                                   "enum": ["injury", "rest", "suspension", "trade", "lineup", "other"]},
                    "status": {"type": "string",
                               "enum": ["out", "ir", "suspended", "doubtful", "questionable",
                                        "game-time decision", "day-to-day", "probable", "available", "unknown"]},
                    "role": {"type": "string", "enum": ["QB", "G", "SP", "K", "star", "starter", "bench", "none"]},
                    "importance": {"type": "string", "enum": list(IMPACT_TIERS)},
                    "reliability": {"type": "number"},
                },
                "required": ["item_index", "team_abbr", "player", "event_type", "status", "role",
                             "importance", "reliability"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}

SYSTEM = """You extract player-availability facts from sports news for a quantitative model.
For each news item, emit zero or more events. Only emit an event when the item states or strongly
implies a specific player's availability for upcoming games. Use the team abbreviations provided.
reliability is 0-1: official injury reports / team announcements ~0.95, named beat reporters ~0.8,
"sources say"/speculation ~0.5. importance reflects the player's value to his team relative to a
replacement (superstar = franchise player / starting QB / starting goalie). Do not guess; if the
player or team is unclear, skip the item."""


class ClaudeNewsAnalyzer:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 40, effort: str = "low", client=None):
        if client is None:
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.effort = effort

    def _call(self, league: str, items: Sequence[str]) -> list[dict]:
        teams = ", ".join(f"{a}={t.name}" for a, t in REGISTRY.get(league, {}).items())
        body = "\n".join(f"[{i}] {t}" for i, t in enumerate(items))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"League: {league}\nTeams: {teams}\n\nNews items:\n{body}"}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )
        if response.stop_reason == "refusal":
            log.warning("news analysis declined: %s", getattr(response, "stop_details", None))
            return []
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text).get("events", [])
        except json.JSONDecodeError:
            log.warning("could not parse analyzer output (stop_reason=%s)", response.stop_reason)
            return []

    def analyze(self, league: str, items: Sequence[tuple[str, Optional[datetime], Optional[str]]]) -> list[NewsEvent]:
        from ..config import get_config

        cfg = get_config(league)
        known = REGISTRY.get(league, {})
        out: list[NewsEvent] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            try:
                raw = self._call(league, [t for t, _, _ in batch])
            except Exception as e:  # API/network errors must not break pick generation
                log.warning("news analyzer failed: %s", e)
                continue
            for ev in raw:
                idx = ev.get("item_index", -1)
                if not (0 <= idx < len(batch)) or ev.get("team_abbr") not in known:
                    continue
                text, when, _ = batch[idx]
                role = None if ev.get("role") in (None, "none") else ev["role"]
                tier = IMPACT_TIERS.get(ev.get("importance", "unknown"))
                impact = None
                if tier is not None and role not in ("G", "SP", "QB"):
                    base = cfg.role_impacts.get("star", (cfg.default_player_impact, 0))[0]
                    impact = tier * base
                out.append(NewsEvent(
                    league=league, team=ev["team_abbr"], player=ev["player"], event_type=ev["event_type"],
                    status=ev["status"], published=when, source=f"claude:{self.model}",
                    reliability=float(min(1.0, max(0.0, ev.get("reliability", 0.7)))), role=role,
                    impact=impact, impact_sd=None if impact is None else impact * 0.5, text=text))
        return out
