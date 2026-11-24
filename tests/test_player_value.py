"""NBA player values from box scores, absence costs, the engine's learned absence scale,
the data-priced injury table and the NBA spread tier."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from sportsedge.models import player_value as PV


def synth_box(n_teams=10, per_team=9, days=150, seed=1, star_out_days=()):
    """A league where each player has a true value (net points per 48). Box-score rates follow
    the value loosely; margins follow the values of the players on the floor. Team T0's star
    (T0-0) sits on `star_out_days`."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    true, mins = {}, {}
    for t in teams:
        for j in range(per_team):
            pid = f"{t}-{j}"
            true[pid] = (9.0 if j == 0 else 4.0 - 0.9 * j) + rng.normal(0, 1.0)
            mins[pid] = [36, 34, 32, 30, 28, 24, 20, 18, 18][j]
    rows, start = [], datetime(2025, 10, 22, 23)
    for d in range(days):
        order = rng.permutation(teams)
        when = start + timedelta(days=d)
        for k in range(0, n_teams, 2):
            home, away = order[k], order[k + 1]
            gid = f"NBA-{when:%Y%m%d}-{away}-{home}-19"
            lineups = {}
            for t in (home, away):
                pl = {f"{t}-{j}": float(mins[f"{t}-{j}"]) for j in range(per_team)}
                if t == "T0" and d in star_out_days:
                    pl.pop("T0-0")
                tot = sum(pl.values())
                lineups[t] = {p: m * 240.0 / tot for p, m in pl.items()}
            strength = {t: sum(m / 48.0 * true[p] for p, m in lineups[t].items()) for t in (home, away)}
            margin = 2.5 + strength[home] - strength[away] + rng.normal(0, 11)
            base = {home: 112 + margin / 2, away: 112 - margin / 2}
            for t, opp in ((home, away), (away, home)):
                share = {p: m * (1 + 0.04 * true[p]) for p, m in lineups[t].items()}
                tot = sum(share.values())
                for p, m in lineups[t].items():
                    pts = base[t] * share[p] / tot
                    rows.append({"game_id": gid, "date": when.isoformat(), "team": t, "opp": opp,
                                 "home": int(t == home), "player_id": p, "player": "Player " + p.replace("-", " "),
                                 "min": m, "pts": round(pts), "reb": round(m / 6), "ast": round(m / 9 * (1 + 0.05 * true[p])),
                                 "stl": 1, "blk": 0, "tov": 1, "fg3m": 1})
    return rows, true


@pytest.fixture(scope="module")
def league():
    return synth_box(star_out_days=set(range(140, 150)))


def test_values_recover_who_matters(league):
    rows, true = league
    games = PV.build_games(rows)
    assert len(games) == 150 * 5 and all(len(g.minutes[g.home]) >= 8 for g in games)
    vals = PV.PlayerValues().fit(games, rows)
    pids = sorted(true)
    corr = np.corrcoef([vals.get(p) for p in pids], [true[p] for p in pids])[0, 1]
    assert corr > 0.7, corr
    stars = [p for p in pids if p.endswith("-0")]
    assert np.mean([vals.get(p) for p in stars]) > np.mean([vals.get(p) for p in pids if p.endswith("-8")]) + 5
    roster = {p: 30.0 for p in pids if p.startswith("T3-")}
    assert PV.cost_when_out(vals, roster, "T3-0") > PV.cost_when_out(vals, roster, "T3-8")
    assert PV.cost_when_out(vals, roster, "T3-0") > 1.0


def test_absences_are_priced_and_already_priced_absences_fade(league):
    rows, _ = league
    x = PV.absence_features(rows)
    t0_home = [g for g in PV.build_games(rows) if g.home == "T0" and g.game_id in x]
    sat = [x[g.game_id] for g in t0_home if "T0-0" not in g.minutes["T0"]]
    played = [x[g.game_id] for g in t0_home if "T0-0" in g.minutes["T0"]]
    assert sat and played
    assert sat[0] < -1.0  # the first game he sits: a clear cost to the home side
    assert np.mean(np.abs(played)) < 0.3
    assert abs(sat[-1]) < abs(sat[0])  # a week later his absence is in the team's recent results
    # today's injury table: a player out for ten games carries little, the regulars carry their value
    imps = {i["player_id"]: i for i in PV.player_impacts(rows)}
    assert "T0-0" not in imps or imps["T0-0"]["impact"] < 0.5 * max(i["impact"] for i in imps.values())
    assert max(imps.values(), key=lambda i: i["impact"])["player_id"].endswith("-0")


def test_engine_learns_how_much_absences_matter():
    """Synthetic NBA seasons: a known absence cost moves margins; the engine learns its scale and
    predicts better with it than without."""
    from sportsedge.engine import LeagueEngine, EngineSettings
    from sportsedge.synthetic import generate

    games = generate("NBA", seasons=1, seed=4)
    results = [g for g in games if hasattr(g, "margin")]
    rng = np.random.default_rng(0)
    lineup = {}
    for r in results:
        if rng.random() < 0.3:
            x = float(rng.choice([-1, 1]) * rng.uniform(2, 7))
            lineup[r.game_id] = x
            r.home_score += int(round(max(x, 0)))
            r.away_score += int(round(max(-x, 0)))

    def ll(e):
        recs = [o for o in e.oos if o["y"] is not None]
        p = np.clip([o["comp"]["kalman"] for o in recs], 1e-4, 1 - 1e-4); y = np.array([o["y"] for o in recs])
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    s = EngineSettings(n_draws=50, use_patterns=False)
    off = LeagueEngine("NBA", s)
    off.observe(results)
    on = LeagueEngine("NBA", s)
    on.lineup = lineup
    on.observe(results)
    assert 0.6 < on.lineup_scale < 1.4, on.lineup_scale
    assert ll(on) < ll(off) - 0.005


def test_injury_table_from_data_replaces_the_flat_default():
    from sportsedge.config import get_config
    from sportsedge.news.impact import PlayerImpactRegistry, build_factors
    from sportsedge.types import NewsEvent

    now = datetime(2026, 11, 24, 18)
    ev = lambda name, status="out": NewsEvent(league="NBA", team="DEN", player=name, status=status,
                                              event_type="injury", published=now, text="")
    reg = PlayerImpactRegistry()
    reg.add("NBA", "Nikola Jokic", 8.3, 2.9)
    events = [ev("Nikola Jokić"), ev("Twelfth Man")]
    # without a complete table, an unknown player gets the league default
    f = {x.player: x for x in build_factors(events, get_config("NBA"), now, reg)}
    assert f["Nikola Jokić"].impact == 8.3 and f["Twelfth Man"].impact == get_config("NBA").default_player_impact
    # a data-priced table covers the league: a player not in it has no measurable cost
    reg.complete.add("NBA")
    f = {x.player: x for x in build_factors(events, get_config("NBA"), now, reg)}
    assert set(f) == {"Nikola Jokić"}


def test_nba_spreads_have_their_own_70_percent_tier():
    from sportsedge.picks import PickPolicy, confidence_policy, spread_policy
    from sportsedge.types import Game, MarketProbability, Prediction

    def pred(league, p_cover, line):
        g = Game(f"{league}-1", league, datetime(2026, 11, 24, 23), "HOM", "AWY")
        pr = Prediction(game=g, home_win_prob=0.6, home_win_lower=0.55, home_win_upper=0.65,
                        expected_margin=3.0, expected_total=220.0)
        pr.markets = [MarketProbability("moneyline", "home", None, 0.6, 0.55, 0.65, -150, 0.58),
                      MarketProbability("spread", "home", line, p_cover, p_cover - 0.04, p_cover + 0.04, -110, 0.5),
                      MarketProbability("spread", "away", line, 1 - p_cover, 0.24, 0.32, -110, 0.5)]
        return pr
    sp, conf = spread_policy(PickPolicy()), confidence_policy(PickPolicy())
    (pk,) = sp.evaluate(pred("NBA", 0.72, -4.5))
    assert pk.market.market == "spread" and pk.market.side == "home"
    assert sp.evaluate(pred("NBA", 0.66, -4.5)) == []          # below 70%
    assert sp.evaluate(pred("NHL", 0.80, -1.5)) == []          # puck lines are not in this tier
    assert all(p.market.market != "spread" for p in conf.evaluate(pred("NBA", 0.72, -4.5)))
