"""Pick a working data source for each league.

ESPN is preferred because it also carries posted betting lines, injuries and
headlines. When ESPN refuses requests (HTTP 403 from its bot protection) or
fails for more than a fifth of the days, MLB and NHL switch to their official
league feeds (data/official.py). NBA and NFL have no keyless official feed,
so they depend on ESPN.
"""
from __future__ import annotations

import logging
from datetime import date

from . import espn
from .official import OFFICIAL_DAY, OFFICIAL_RANGE

log = logging.getLogger(__name__)


class SourceUnavailable(RuntimeError):
    pass


def fetch_history(league: str, start: date, end: date, progress=None):
    games = espn.fetch_history(league, start, end, progress=progress)
    failed = espn.last_history_failed_share
    if failed > 0.2 and league in OFFICIAL_RANGE:
        if progress:
            progress(f"{league}: ESPN refused {failed:.0%} of requests; using the official league feed instead.")
        merged = {g.game_id: g for g in OFFICIAL_RANGE[league](start, end, progress=progress)
                  if type(g).__name__ == "GameResult"}
        merged.update({g.game_id: g for g in games})  # ESPN rows carry odds; keep them where present
        games = sorted(merged.values(), key=lambda g: g.start_time)
    elif failed > 0.2 and not games:
        raise SourceUnavailable(
            f"ESPN refused the requests for {league} and there is no official backup feed for it. "
            "Run `python3 run.py --check`, and try again later or from another network.")
    return games


def fetch_day(league: str, day: date):
    games = espn.fetch_day(league, day)
    if espn.last_day_failed and league in OFFICIAL_DAY:
        return OFFICIAL_DAY[league](day)
    return games
