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


# ------------------------------------------------------------------ NHL and NFL
def _hdr(eid, date_iso, home, away):
    return {"id": eid, "competitions": [{"date": date_iso, "status": {"type": {"completed": True}},
                                         "competitors": [{"homeAway": "home", "team": {"abbreviation": home}},
                                                         {"homeAway": "away", "team": {"abbreviation": away}}]}]}


def test_parse_nhl_summary_skaters_and_goalies():
    from sportsedge.props.boxscores import parse_boxscore
    fwd = {"name": "forwards",
           "keys": ["blockedShots", "hits", "takeaways", "plusMinus", "timeOnIce", "powerPlayTimeOnIce", "shortHandedTimeOnIce",
                    "evenStrengthTimeOnIce", "shifts", "goals", "assists", "shotsTotal", "shotsMissed", "giveaways",
                    "faceoffsWon", "faceoffsLost", "faceoffPercent", "penaltyMinutes"],
           "labels": ["BS", "HT", "TK", "+/-", "TOI", "PPTOI", "SHTOI", "ESTOI", "SHFT", "G", "A", "S", "SM", "GV", "FW", "FL", "FO%", "PIM"],
           "athletes": [{"athlete": {"id": "8479318", "displayName": "Auston Matthews", "position": {"abbreviation": "C"}},
                         "stats": ["1", "2", "1", "+1", "21:30", "3:10", "0:00", "18:20", "24", "1", "1", "5", "2", "1", "10", "8", "55.6", "0"]}]}
    gk = {"name": "goalies",
          "keys": ["goalsAgainst", "shotsAgainst", "shootoutSaves", "shootoutShotsAgainst", "saves", "savePct",
                   "evenStrengthSaves", "powerPlaySaves", "shortHandedSaves", "timeOnIce", "penaltyMinutes"],
          "labels": ["GA", "SA", "SOS", "SOSA", "SV", "SV%", "ESSV", "PPSV", "SHSV", "TOI", "PIM"],
          "athletes": [{"athlete": {"id": "8480045", "displayName": "Joseph Woll"},
                        "stats": ["2", "31", "0", "0", "29", ".935", "22", "5", "2", "60:00", "0"]}]}
    data = {"header": _hdr("401800001", "2026-11-20T00:00Z", "TOR", "MTL"),
            "boxscore": {"players": [{"team": {"abbreviation": "TOR"}, "statistics": [fwd, gk]}]}}
    rows = parse_boxscore(data, league="NHL")
    sk = next(r for r in rows if r["pos"] == "C")
    g = next(r for r in rows if r["pos"] == "G")
    assert (sk["toi"], sk["g"], sk["a"], sk["s"], sk["pm"]) == (21.5, 1, 1, 5, 1)
    assert (g["sa"], g["ga"], g["sv"], g["toi"]) == (31, 2, 29, 60)
    assert sk["team"] == "TOR" and sk["opp"] == "MTL" and sk["game_id"].startswith("NHL-2026111")


def test_parse_nfl_summary_merges_a_players_lines():
    from sportsedge.props.boxscores import parse_boxscore
    passing = {"name": "passing", "labels": ["C/ATT", "YDS", "AVG", "TD", "INT", "SACKS", "QBR", "RTG"],
               "athletes": [{"athlete": {"id": "3139477", "displayName": "Patrick Mahomes"},
                             "stats": ["24/35", "286", "8.2", "2", "1", "2-14", "65.1", "98.7"]}]}
    rushing = {"name": "rushing", "labels": ["CAR", "YDS", "AVG", "TD", "LONG"],
               "athletes": [{"athlete": {"id": "3139477", "displayName": "Patrick Mahomes"}, "stats": ["5", "31", "6.2", "0", "12"]},
                            {"athlete": {"id": "4361529", "displayName": "Isiah Pacheco"}, "stats": ["16", "71", "4.4", "1", "18"]}]}
    receiving = {"name": "receiving", "labels": ["REC", "YDS", "AVG", "TD", "LONG", "TGTS"],
                 "athletes": [{"athlete": {"id": "15847", "displayName": "Travis Kelce"}, "stats": ["7", "84", "12.0", "1", "25", "9"]}]}
    data = {"header": _hdr("401772936", "2026-11-23T18:00Z", "KC", "BUF"),
            "boxscore": {"players": [{"team": {"abbreviation": "KC"}, "statistics": [passing, rushing, receiving]}]}}
    rows = {r["player"]: r for r in parse_boxscore(data, league="NFL")}
    m = rows["Patrick Mahomes"]
    assert (m["cmp"], m["att"], m["pass_yds"], m["pass_td"], m["int"], m["sacks"], m["sack_yds"]) == (24, 35, 286, 2, 1, 2, 14)
    assert m["rush_yds"] == 31 and m["pos"] == "QB"
    assert (rows["Travis Kelce"]["rec"], rows["Travis Kelce"]["tgt"]) == (7, 9) and rows["Isiah Pacheco"]["rush_td"] == 1


def synth_nhl(days=170, seed=2):
    """NHL-like league: skaters with true goal values, two goalies per team (the backup much worse)
    who start 75% / 25%. The opposing goalie's save percentage sets how many shots go in."""
    rng = np.random.default_rng(seed)
    teams = [f"H{i}" for i in range(12)]
    sk = {t: [(f"{t}-s{j}", (0.6 if j == 0 else 0.15 - 0.03 * j) + rng.normal(0, 0.05)) for j in range(12)] for t in teams}
    gl = {t: [(f"{t}-g1", 0.915), (f"{t}-g2", 0.885)] for t in teams}
    toi = [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    rows, start = [], datetime(2025, 10, 8, 23)
    for d in range(days):
        order = rng.permutation(teams)
        when = start + timedelta(days=d)
        for k in range(0, len(teams), 2):
            home, away = order[k], order[k + 1]
            gid = f"NHL-{when:%Y%m%d}-{away}-{home}-19"
            goalie = {t: gl[t][0] if rng.random() < 0.75 else gl[t][1] for t in (home, away)}
            strength = {t: sum(m / 60 * v for (p, v), m in zip(sk[t], toi)) for t in (home, away)}
            goals = {}
            for t, o in ((home, away), (away, home)):
                shots = max(10, int(rng.normal(30 + 12 * (strength[t] - strength[o]), 4)))
                goals[t] = int(rng.binomial(shots, 1 - goalie[o][1]))
                goals[t + "_shots"] = shots
            for t, o in ((home, away), (away, home)):
                w = np.array([m * (1 + 3 * max(v, 0)) for (p, v), m in zip(sk[t], toi)])
                scorers = rng.choice(12, size=goals[t], p=w / w.sum())
                for j, ((p, v), m) in enumerate(zip(sk[t], toi)):
                    rows.append({"game_id": gid, "date": when.isoformat(), "team": t, "opp": o, "home": int(t == home),
                                 "player_id": p, "player": "Skater " + p.replace("-", " "), "pos": "C", "toi": float(m),
                                 "g": float((scorers == j).sum()), "a": float(rng.poisson(0.3)), "s": float(rng.poisson(2 + 3 * v)),
                                 "pm": 0.0, "bs": 1.0, "ht": 1.0, "tk": 1.0, "gv": 1.0, "sa": None, "ga": None, "sv": None})
                g_id, _ = goalie[t]
                sa, ga = goals[o + "_shots"], goals[o]
                rows.append({"game_id": gid, "date": when.isoformat(), "team": t, "opp": o, "home": int(t == home),
                             "player_id": g_id, "player": "Goalie " + g_id.replace("-", " "), "pos": "G", "toi": 60.0,
                             "g": None, "a": None, "s": None, "pm": None, "bs": None, "ht": None, "tk": None, "gv": None,
                             "sa": float(sa), "ga": float(ga), "sv": float(sa - ga)})
    return rows


def test_nhl_goalie_starts_and_skaters_are_priced():
    rows = synth_nhl()
    x = PV.absence_features(rows, league="NHL")
    by = {}
    for r in rows:
        if r["pos"] == "G" and r["home"]:
            by[r["game_id"]] = r["player_id"].endswith("g2")  # home backup started
    late = [g for g in x if g in by and g >= "NHL-20260101"]
    backup = [x[g] for g in late if by[g]]
    usual = [x[g] for g in late if not by[g]]
    # a backup start costs the home side (save percentages are shrunk hard toward the league: K = 2000 shots)
    assert np.mean(backup) < np.mean(usual) - 0.15, (np.mean(backup), np.mean(usual))
    imps = PV.player_impacts(rows, league="NHL")
    goalies = [i for i in imps if i["role"] == "G"]
    assert goalies and all(i["player_id"].endswith("g1") for i in goalies)  # the usual starters
    assert np.mean([i["impact"] for i in goalies]) > 0.12
    stars = [i for i in imps if i["player_id"].endswith("-s0")]
    depth = [i["impact"] for i in imps if i["player_id"].endswith(("-s9", "-s10", "-s11"))]
    assert len(stars) >= 8 and np.mean([i["impact"] for i in stars]) > (np.mean(depth) if depth else 0.0) + 0.02


def synth_nfl(weeks=34, seed=3):
    """NFL-like league: each team's starting QB is much better than his backup and misses some games."""
    rng = np.random.default_rng(seed)
    teams = [f"F{i}" for i in range(16)]
    rows, start = [], datetime(2024, 9, 8, 17)
    for w in range(weeks):
        when = start + timedelta(days=7 * w)
        order = rng.permutation(teams)
        for k in range(0, len(teams), 2):
            home, away = order[k], order[k + 1]
            gid = f"NFL-{when:%Y%m%d}-{away}-{home}-13"
            for t, o in ((home, away), (away, home)):
                out = rng.random() < 0.15
                qb, anya = (f"{t}-qb2", 4.5) if out else (f"{t}-qb1", 7.0)
                db = 36
                net = rng.normal(anya, 1.5) * db
                rows.append({"game_id": gid, "date": when.isoformat(), "team": t, "opp": o, "home": int(t == home),
                             "player_id": qb, "player": "QB " + qb, "pos": "QB", "cmp": 22.0, "att": 33.0,
                             "pass_yds": float(round(net + 3 * 7)), "pass_td": 1.0, "int": 1.0, "sacks": 3.0, "sack_yds": 21.0,
                             "rush_att": 3.0, "rush_yds": 12.0, "rush_td": 0.0, "rec": None, "rec_yds": None, "rec_td": None, "tgt": None})
                for j, (ypt, touches) in enumerate(((5.5, 18), (9.5, 8), (8.0, 7))):
                    p = f"{t}-sk{j}"
                    rows.append({"game_id": gid, "date": when.isoformat(), "team": t, "opp": o, "home": int(t == home),
                                 "player_id": p, "player": "Skill " + p, "pos": "", "cmp": None, "att": None,
                                 "pass_yds": None, "pass_td": None, "int": None, "sacks": None, "sack_yds": None,
                                 "rush_att": float(touches if j == 0 else 0), "rush_yds": float(round(ypt * touches)) if j == 0 else 0.0,
                                 "rush_td": 0.0, "rec": float(touches if j else 2), "rec_yds": float(round(ypt * touches)) if j else 10.0,
                                 "rec_td": 0.0, "tgt": float(touches if j else 3)})
    return rows


def test_nfl_quarterback_absences_are_priced():
    rows = synth_nfl()
    x = PV.absence_features(rows, league="NFL")
    home_qb = {r["game_id"]: r["player_id"] for r in rows if r["home"] and r["pos"] == "QB"}
    away_qb = {r["game_id"]: r["player_id"] for r in rows if not r["home"] and r["pos"] == "QB"}
    late = [g for g in x if g >= "NFL-20241101"]
    home_out = [x[g] for g in late if home_qb[g].endswith("qb2") and away_qb[g].endswith("qb1")]
    both_in = [x[g] for g in late if home_qb[g].endswith("qb1") and away_qb[g].endswith("qb1")]
    assert home_out and np.mean(home_out) < -2.5 and abs(np.mean(both_in)) < 0.5, (np.mean(home_out), np.mean(both_in))
    imps = PV.player_impacts(rows, league="NFL")
    qbs = [i for i in imps if i["role"] == "QB"]
    assert len(qbs) >= 12 and all(i["player_id"].endswith("qb1") for i in qbs)
    assert 2.0 < np.median([i["impact"] for i in qbs]) < 9.0  # shrunk: backups have few dropbacks
    assert max(i["impact"] for i in imps if i["role"] != "QB") < min(i["impact"] for i in qbs)


def test_nfl_table_covers_skill_positions_only():
    from sportsedge.config import get_config
    from sportsedge.news.impact import PlayerImpactRegistry, build_factors
    from sportsedge.types import NewsEvent

    now = datetime(2026, 11, 27, 18)
    ev = lambda name, role: NewsEvent(league="NFL", team="KC", player=name, status="out", event_type="injury",
                                      published=now, text="", role=role)
    reg = PlayerImpactRegistry()
    reg.add("NFL", "Patrick Mahomes", 6.2, 2.2, "QB")
    reg.roles["NFL"] = {"QB", "RB", "WR", "TE"}
    f = {x.player: x for x in build_factors([ev("Patrick Mahomes", "QB"), ev("Third Stringer", "QB"),
                                             ev("Joe Thuney", "G")], get_config("NFL"), now, reg)}
    assert f["Patrick Mahomes"].impact == 6.2
    assert "Third Stringer" not in f                      # a QB who does not start costs nothing
    assert f["Joe Thuney"].impact == get_config("NFL").default_player_impact  # linemen keep the default
