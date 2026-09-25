"""Offline tests for the resilient HTTP client, official league feeds and source fallback."""
from datetime import date, datetime

import pytest

import sportsedge.data.espn as espn
import sportsedge.data.http as http
import sportsedge.data.official as official
import sportsedge.data.sources as sources
from sportsedge.types import GameResult


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload or {}

    def json(self):
        return self._payload


class _Session:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {})))
        return self.script.pop(0) if self.script else _Resp(403)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    http.reset()
    monkeypatch.setattr(http, "MIN_INTERVAL", 0.0)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    yield
    http.reset()


def _use(monkeypatch, sess):
    monkeypatch.setattr(http, "_session", lambda: sess)


def test_client_retries_other_header_profile_then_succeeds(monkeypatch):
    sess = _Session([_Resp(403), _Resp(200, {"ok": 1})])
    _use(monkeypatch, sess)
    assert http.get_json("https://site.api.espn.com/x") == {"ok": 1}
    assert sess.calls[0][1]["User-Agent"].startswith("Mozilla")  # browser-like headers
    assert sess.calls[0][1] != sess.calls[1][1]  # second try used a different profile


def test_client_falls_back_to_alternate_espn_host(monkeypatch):
    sess = _Session([_Resp(403)] * 3 + [_Resp(200, {"ok": 2})])
    _use(monkeypatch, sess)
    assert http.get_json("https://site.api.espn.com/x") == {"ok": 2}
    assert "site.web.api.espn.com" in sess.calls[-1][0]


def test_circuit_breaker_stops_hammering(monkeypatch):
    sess = _Session([])  # always 403
    _use(monkeypatch, sess)
    for _ in range(10):
        assert http.get_json("https://example.org/x") is None
    n = len(sess.calls)
    assert http.is_blocked("example.org")
    assert http.get_json("https://example.org/y") is None and len(sess.calls) == n  # fails fast


MLB_FIXTURE = {"dates": [{"date": "2026-09-24", "games": [
    {"gamePk": 1, "gameType": "R", "gameDate": "2026-09-24T23:05:00Z",
     "status": {"abstractGameState": "Final", "detailedState": "Final"},
     "teams": {"home": {"team": {"name": "New York Yankees", "abbreviation": "NYY"}, "score": 5,
                        "leagueRecord": {"wins": 90, "losses": 68}},
               "away": {"team": {"name": "Arizona Diamondbacks", "abbreviation": "AZ"}, "score": 3}}},
    {"gamePk": 2, "gameType": "R", "gameDate": "2026-09-25T02:10:00Z",
     "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
     "teams": {"home": {"team": {"name": "Athletics", "abbreviation": "ATH"},
                        "probablePitcher": {"fullName": "Ace Starter"}},
               "away": {"team": {"name": "Chicago White Sox", "abbreviation": "CWS"}}}},
    {"gamePk": 3, "gameType": "S", "gameDate": "2026-03-01T18:05:00Z",
     "status": {"abstractGameState": "Final"}, "teams": {"home": {"team": {"abbreviation": "BOS"}, "score": 1},
                                                          "away": {"team": {"abbreviation": "NYY"}, "score": 0}}},
]}]}

NHL_FIXTURE = {"games": [
    {"id": 10, "gameType": 2, "startTimeUTC": "2026-10-10T23:00:00Z", "gameState": "OFF",
     "homeTeam": {"abbrev": "TBL", "score": 4}, "awayTeam": {"abbrev": "LAK", "score": 3},
     "periodDescriptor": {"number": 4, "periodType": "OT"}},
    {"id": 11, "gameType": 2, "startTimeUTC": "2026-10-11T02:00:00Z", "gameState": "LIVE",
     "homeTeam": {"abbrev": "UTA", "score": 1}, "awayTeam": {"abbrev": "NJD", "score": 2},
     "periodDescriptor": {"number": 2}, "clock": {"timeRemaining": "08:41"}},
    {"id": 12, "gameType": 1, "startTimeUTC": "2026-09-25T23:00:00Z", "gameState": "FUT",
     "homeTeam": {"abbrev": "BOS"}, "awayTeam": {"abbrev": "NYR"}},
]}


def test_parse_mlb_official():
    games = official.parse_mlb_schedule(MLB_FIXTURE)
    assert len(games) == 2  # spring training dropped
    final, upcoming = games
    assert isinstance(final, GameResult) and (final.home, final.away, final.margin) == ("NYY", "ARI", 2)
    assert final.extras["teams"]["home"]["record"] == "90-68"
    assert not isinstance(upcoming, GameResult) and (upcoming.home, upcoming.away) == ("ATH", "CHW")
    assert upcoming.extras["home_pitcher"]["name"] == "Ace Starter"
    assert upcoming.game_id == "MLB-20260924-CHW-ATH-22"  # 02:10 UTC = 10pm ET the previous day


def test_parse_nhl_official():
    games = official.parse_nhl_score(NHL_FIXTURE)
    assert len(games) == 2  # preseason dropped
    final, live = games
    assert (final.home, final.away, final.home_score) == ("TB", "LA", 4) and final.extras["status"]["detail"] == "Final/OT"
    assert (live.home, live.away) == ("UTAH", "NJ") and live.extras["live"] == {"home": 1, "away": 2}


def test_same_game_gets_same_id_from_espn_and_official():
    start = datetime(2026, 9, 24, 23, 5)
    assert official.canonical_id("MLB", start, "NYY", "ARI") == official.canonical_id("MLB", datetime(2026, 9, 24, 23, 10), "NYY", "ARI")


def test_history_falls_back_to_official_when_espn_refuses(monkeypatch):
    def refused(league, start, end, progress=None):
        espn.last_history_failed_share = 1.0
        return []

    monkeypatch.setattr(espn, "fetch_history", refused)
    monkeypatch.setitem(official.OFFICIAL_RANGE, "MLB", lambda s, e, progress=None: official.parse_mlb_schedule(MLB_FIXTURE))
    monkeypatch.setattr(sources, "OFFICIAL_RANGE", official.OFFICIAL_RANGE)
    games = sources.fetch_history("MLB", date(2026, 9, 1), date(2026, 9, 24))
    assert len(games) == 1 and games[0].home == "NYY"  # completed games only
    with pytest.raises(sources.SourceUnavailable):
        sources.fetch_history("NBA", date(2026, 9, 1), date(2026, 9, 24))


def test_day_falls_back_to_official_when_espn_refuses(monkeypatch):
    def refused(league, d):
        espn.last_day_failed = True
        return []

    monkeypatch.setattr(espn, "fetch_day", refused)
    monkeypatch.setitem(official.OFFICIAL_DAY, "NHL", lambda d: official.parse_nhl_score(NHL_FIXTURE))
    monkeypatch.setattr(sources, "OFFICIAL_DAY", official.OFFICIAL_DAY)
    assert len(sources.fetch_day("NHL", date(2026, 10, 10))) == 2
    espn.last_day_failed = False
