import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from sportsedge.config import get_config
from sportsedge.engine import EngineSettings, LeagueEngine
from sportsedge.news.feeds import parse_espn_injuries, parse_espn_news
from sportsedge.news.impact import PlayerImpactRegistry, build_factors
from sportsedge.news.llm import ClaudeNewsAnalyzer
from sportsedge.news.parser import normalize_status, parse_text
from sportsedge.synthetic import generate
from sportsedge.types import Game, NewsEvent

NOW = datetime(2026, 1, 10, 18, tzinfo=timezone.utc)


@pytest.mark.parametrize("text,status,player,team", [
    ("Celtics star Jayson Tatum (ankle) ruled out vs. Knicks", "out", "Jayson Tatum", "BOS"),
    ("Joel Embiid listed as questionable for 76ers on Friday", "questionable", "Joel Embiid", "PHI"),
    ("Chiefs QB Patrick Mahomes is doubtful with a high ankle sprain", "doubtful", "Patrick Mahomes", "KC"),
    ("Oilers place Connor McDavid on long-term injured reserve", "ir", "Connor McDavid", "EDM"),
    ("Nikola Jokic will play tonight, Nuggets say", "available", "Nikola Jokic", "DEN"),
    ("Yankees activated Aaron Judge from the 10-day IL", "available", "Aaron Judge", "NYY"),
    ("Dodgers place Mookie Betts on 10-day IL", "ir", "Mookie Betts", "LAD"),
])
def test_parser_extracts_status(text, status, player, team):
    league = {"KC": "NFL", "EDM": "NHL", "NYY": "MLB", "LAD": "MLB"}.get(team, "NBA")
    ev = parse_text(league, text)
    assert ev and ev[0].status == status and ev[0].player == player and ev[0].team == team


def test_parser_roles_and_rumours():
    ev = parse_text("NFL", "Sources: Bills QB Josh Allen expected to be questionable")[0]
    assert ev.role == "QB" and ev.reliability < 0.9
    assert parse_text("NBA", "Heated rivalry renewed in Boston tonight") == []


def test_normalize_status():
    assert normalize_status("Out") == "out"
    assert normalize_status("15-Day IL") == "ir"
    assert normalize_status("Day-To-Day") == "day-to-day"


def test_build_factors_mixture_parameters():
    cfg = get_config("NBA")
    evs = [
        NewsEvent("NBA", "BOS", "Star Player", "injury", "questionable", NOW, role="star"),
        NewsEvent("NBA", "BOS", "Star Player", "injury", "out", NOW - timedelta(days=2), role="star"),  # older
        NewsEvent("NBA", "NY", "Bench Guy", "injury", "available", NOW),
        NewsEvent("NBA", "NY", "Old News", "injury", "out", NOW - timedelta(days=20)),
        NewsEvent("NBA", "NY", "Returning Guy", "injury", "out", NOW - timedelta(days=3)),
        NewsEvent("NBA", "NY", "Returning Guy", "lineup", "available", NOW),
    ]
    fs = {f.player: f for f in build_factors(evs, cfg, NOW)}
    assert fs["Star Player"].play_prob == pytest.approx(1 - 0.9 * 0.5)  # latest report wins
    assert fs["Star Player"].impact == cfg.role_impacts["star"][0]
    assert "Bench Guy" not in fs and "Returning Guy" not in fs
    assert fs["Old News"].play_prob > 0.9  # stale report shrinks toward "plays"
    reg = PlayerImpactRegistry()
    reg.add("NBA", "Star Player", 6.0, 1.0)
    assert build_factors(evs[:1], cfg, NOW, reg)[0].impact == 6.0


def test_injury_moves_probability_and_widens_band():
    games = generate("NBA", seasons=1, seed=4)
    eng = LeagueEngine("NBA", EngineSettings(n_draws=4000, seed=1)).fit(games)
    g = Game("x", "NBA", games[-1].start_time + timedelta(days=1), "BOS", "NY")
    base = eng.predict(g)
    cfg = get_config("NBA")
    out = build_factors([NewsEvent("NBA", "BOS", "Franchise Star", "injury", "out", reliability=1.0, role="star")], cfg, NOW)
    q = build_factors([NewsEvent("NBA", "BOS", "Franchise Star", "injury", "questionable", reliability=1.0, role="star")], cfg, NOW)
    p_out = eng.predict(g, factors=out)
    p_q = eng.predict(g, factors=q)
    assert p_out.home_win_prob < p_q.home_win_prob < base.home_win_prob
    width = lambda p: p.home_win_upper - p.home_win_lower
    assert width(p_q) > width(base)  # unresolved news = more uncertainty
    assert any("Franchise Star" in n for n in p_out.notes)


def test_parse_espn_feeds():
    inj = {"injuries": [{"displayName": "Boston Celtics", "injuries": [
        {"status": "Out", "date": "2026-01-10T15:00Z", "shortComment": "Tatum (ankle) will miss Friday's game.",
         "athlete": {"displayName": "Jayson Tatum", "position": {"abbreviation": "SF"}}}]}]}
    ev = parse_espn_injuries("NBA", inj)
    assert ev[0].team == "BOS" and ev[0].status == "out" and ev[0].published.year == 2026
    news = {"articles": [{"headline": "Embiid questionable", "description": "76ers big man is questionable.",
                          "published": "2026-01-10T12:00:00Z",
                          "categories": [{"type": "team", "description": "Philadelphia 76ers"}]}]}
    text, when, team = parse_espn_news("NBA", news)[0]
    assert team == "PHI" and "questionable" in text


class _FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.kwargs = payload, stop_reason, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        block = SimpleNamespace(type="text", text=json.dumps(self.payload))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[block], stop_details=None)


def test_claude_analyzer_converts_structured_output():
    payload = {"events": [
        {"item_index": 0, "team_abbr": "DEN", "player": "Nikola Jokic", "event_type": "injury",
         "status": "questionable", "role": "star", "importance": "superstar", "reliability": 0.8},
        {"item_index": 5, "team_abbr": "DEN", "player": "Nobody", "event_type": "injury",
         "status": "out", "role": "none", "importance": "role", "reliability": 0.8},  # bad index -> dropped
    ]}
    fake = SimpleNamespace(messages=_FakeMessages(payload))
    an = ClaudeNewsAnalyzer(client=fake)
    evs = an.analyze("NBA", [("Jokic questionable with knee soreness", NOW, None)])
    assert len(evs) == 1 and evs[0].player == "Nikola Jokic" and evs[0].impact > 4
    kw = fake.messages.kwargs
    assert kw["model"] == "claude-opus-5" and kw["output_config"]["format"]["type"] == "json_schema"
    refused = ClaudeNewsAnalyzer(client=SimpleNamespace(messages=_FakeMessages(payload, "refusal")))
    assert refused.analyze("NBA", [("x", NOW, None)]) == []


def test_news_never_moves_probability_the_wrong_way():
    """Stacking weights are non-negative, so bad news for a team can only lower its
    win probability, with or without a market price in the blend."""
    games = generate("NBA", seasons=1, seed=9)
    cut = games[-1].start_time.date()
    eng = LeagueEngine("NBA", EngineSettings(n_draws=800, seed=2)).fit([g for g in games if g.start_time.date() < cut])
    assert all(v >= 0 for k, v in eng.stacker.weights().items() if not k.endswith("intercept"))
    cfg = get_config("NBA")
    for g in [x for x in games if x.start_time.date() == cut][:4]:
        game = Game(g.game_id, "NBA", g.start_time, g.home, g.away)
        f = build_factors([NewsEvent("NBA", g.away, "Star", "injury", "out", reliability=1.0, role="star")], cfg, NOW)
        for odds in (g.odds, None):
            assert eng.predict(game, odds, f).home_win_prob > eng.predict(game, odds).home_win_prob
