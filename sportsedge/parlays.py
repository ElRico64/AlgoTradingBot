"""Parlay of the day: legs from different games combined into one bet that pays
at least 2x (+100), shown with its real chance of hitting, and only when the
model sees it as a good bet at that payout.

The maths
---------
* **Market check on every leg (winner's-curse correction).** The model's
  probability is calibrated on average, but when it disagrees with the
  sportsbooks only part of the disagreement turns out real. Picking the
  legs where it disagrees most would pick up its errors (walk-forward on
  simulated leagues: parlays claimed 49%, hit 42%, right at the books'
  number). Each leg therefore keeps only the share s of its disagreement
  that has held up:

      logit p_leg = logit p_market + s · (logit p_model − logit p_market)

  s is fitted by maximum likelihood on graded games, per market type, with
  a normal prior centred on the walk-forward estimate (moneyline 0.44,
  spread 0.27, total 0.67; sd 0.25). As the live record grows, the data
  take over from the prior.
* **Independence.** Legs come from different games with no team in two legs.
  Given the models, separate games are independent, so the hit probability
  is the product of the leg probabilities, P = Π p_i. Same-game legs are
  excluded: their outcomes are correlated (a favourite winning big also
  covers and pushes the total), and books price same-game parlays with their
  own correlation model, so a product of posted prices would be wrong.
* **Uncertainty.** Each leg's probability is itself uncertain. Its 80%
  credible band (10th–90th percentile of the model's posterior draws) is
  turned into a Beta distribution by moment matching. The parlay's 80% band
  comes from Monte Carlo over the product of the legs' Beta draws, so a
  parlay built from shaky legs is caught even when its average looks fine.
  The band is shown on the board next to the chance.
* **Payout.** A standard parlay of independent legs pays the product of the
  legs' decimal prices; it must be 2.0 or more.
* **A good bet, not just a likely one.** Every leg must be worth its own
  price (the model's chance at least the leg's break-even), so the books'
  margin isn't compounded into the parlay, and the whole parlay must show an
  expected return of at least +2% at its payout. A 40% floor keeps long shots
  out.
* **Search.** Every combination of 2–4 legs over the strongest candidates
  (one leg per game) is scored exactly. Among the combinations that pass,
  the one most likely to hit is the parlay of the day.
* **Market check.** The product of the legs' de-vigged market probabilities
  is how likely the sportsbooks think the parlay is; it is shown next to the
  model's number.

Everything here works on the board's JSON (games and markets), so parlays can
mix leagues.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

US_EASTERN = ZoneInfo("America/New_York")
# share of the model-vs-market disagreement that held up, walk-forward on the simulated leagues
PRIOR_SHARE = {"moneyline": 0.44, "spread": 0.27, "total": 0.67}
PRIOR_SD = 0.25
Z80 = 1.2815515655446004  # the 10th/90th percentiles of a normal are mean ± 1.2816 sd


@dataclass
class ParlayPolicy:
    min_decimal: float = 2.0        # pays at least 2x the stake (+100)
    min_ev: float = 0.02            # expected return at that payout, by the model: +2% or more
    min_leg_ev: float = 0.0         # every leg worth its own price
    min_probability: float = 0.40   # no long shots
    min_lower_bound: Optional[float] = None  # optional floor on the band's low end
    min_legs: int = 2
    max_legs: int = 4               # more legs compound model error
    min_leg_probability: float = 0.50
    max_candidates: int = 24
    max_per_day: int = 1            # the parlay of the day
    draws: int = 4000


# --------------------------------------------------------------------- helpers
def american_to_decimal(a: float) -> float:
    return 1 + a / 100 if a >= 100 else 1 + 100 / -a


def decimal_to_american(d: float) -> float:
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


def et_date(start_iso: str) -> str:
    """US Eastern date of a start time stored as naive UTC."""
    return datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc).astimezone(US_EASTERN).date().isoformat()


def beta_params(p: float, lo: float, hi: float) -> tuple[float, float]:
    """Beta(a, b) with mean p and the spread implied by an 80% band [lo, hi]."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    sd = max(hi - lo, 0.0) / (2 * Z80)
    var = min(sd * sd, 0.95 * p * (1 - p))
    k = 1e4 if var <= 1e-9 else max(p * (1 - p) / var - 1, 2.0)
    return p * k, (1 - p) * k


def parlay_band(legs: list[dict], draws: int = 4000, seed: Optional[int] = None) -> tuple[float, float, float]:
    """(probability, 10th pct, 90th pct) of a parlay of independent legs."""
    p = float(np.prod([leg["p"] for leg in legs]))
    if seed is None:
        seed = int(hashlib.sha1("|".join(leg_key(x) for x in legs).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    prod = np.ones(draws)
    for leg in legs:
        a, b = beta_params(leg["p"], leg.get("lo", leg["p"]), leg.get("hi", leg["p"]))
        prod *= rng.beta(a, b, draws)
    return p, float(np.quantile(prod, 0.10)), float(np.quantile(prod, 0.90))


def _logit(x: float) -> float:
    x = min(max(x, 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def shrink(p: float, fair: float, share: float) -> float:
    """Keep `share` of the model's disagreement with the market, on the log-odds scale."""
    return 1 / (1 + math.exp(-(_logit(fair) + share * (_logit(p) - _logit(fair)))))


def learn_shares(rows: list[tuple], prior: dict = PRIOR_SHARE, prior_sd: float = PRIOR_SD) -> dict:
    """MAP estimate of the share per market type from graded legs (market, p_model, p_market, won 0/1)."""
    from scipy.optimize import minimize_scalar

    out = {}
    for mk, s0 in prior.items():
        r = [x for x in rows if x[0] == mk]
        if not r:
            out[mk] = {"share": s0, "n": 0}
            continue
        lp = np.array([_logit(x[1]) for x in r])
        lf = np.array([_logit(x[2]) for x in r])
        y = np.array([float(x[3]) for x in r])

        def neg_post(s):
            q = np.clip(1 / (1 + np.exp(-(lf + s * (lp - lf)))), 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + 0.5 * ((s - s0) / prior_sd) ** 2

        s = float(minimize_scalar(neg_post, bounds=(-0.5, 1.5), method="bounded").x)
        out[mk] = {"share": round(min(max(s, 0.0), 1.0), 3), "n": len(r)}
    return out


def leg_key(leg: dict) -> str:
    return f"{leg['league']}:{leg['game_id']}:{leg['market']}:{leg['side']}:{leg.get('line')}"


def parlay_id(day: str, legs: list[dict]) -> str:
    h = hashlib.sha1("|".join(sorted(leg_key(x) for x in legs)).encode()).hexdigest()[:12]
    return f"parlay:{day}:{h}"


# ------------------------------------------------------------------ candidates
def candidate_legs(games: list[dict], day: str, policy: ParlayPolicy, shares: Optional[dict] = None) -> list[dict]:
    """Legs that can go into today's parlays: game-day games that haven't
    started, markets with a posted price, a market probability and
    out-of-sample calibration. Probabilities are market-checked (see top)."""
    shares = shares or {k: {"share": v} for k, v in PRIOR_SHARE.items()}
    legs = []
    for g in games:
        if g.get("state") != "pre" or g.get("early") or g.get("frozen") or et_date(g["start"]) != day:
            continue
        for m in g.get("markets") or []:
            if m.get("price") is None or m.get("fair") is None or m["market"] not in shares:
                continue
            if any("calibration" in f for f in (m.get("conf_fails") or []) + (m.get("fails") or [])):
                continue
            sh = shares[m["market"]]["share"]
            p = shrink(m["p"], m["fair"], sh)
            if p < policy.min_leg_probability:
                continue
            legs.append({
                "league": g["league"], "game_id": g["id"], "date": day, "start": g["start"],
                "home": g["home"]["abbr"], "away": g["away"]["abbr"],
                "market": m["market"], "side": m["side"], "line": m.get("line"), "label": m["label"],
                "price": m["price"], "p": round(p, 4), "p_model": m["p"],
                "lo": round(shrink(m.get("lo", m["p"]), m["fair"], sh), 4),
                "hi": round(shrink(m.get("hi", m["p"]), m["fair"], sh), 4),
                "fair": m.get("fair"), "status": "pending",
            })
    return legs


def leg_ev(leg: dict) -> float:
    return leg["p"] * american_to_decimal(leg["price"]) - 1


def _score(leg: dict) -> float:
    """Payout gained per unit of hit probability given up: log(decimal) / -log(p)."""
    return math.log(american_to_decimal(leg["price"])) / max(-math.log(leg["p"]), 1e-6)


# ---------------------------------------------------------------------- search
def scan(legs: list[dict], policy: ParlayPolicy, used_games: set = frozenset()) -> tuple[list[list[dict]], Optional[dict]]:
    """Qualifying parlays (best first, no shared games) and the closest miss."""
    legs = [x for x in legs if x["game_id"] not in used_games]
    legs.sort(key=_score, reverse=True)
    pool, per_game = [], {}
    for x in legs:  # at most two candidate legs per game keeps the search small
        if per_game.get(x["game_id"], 0) < 2:
            pool.append(x)
            per_game[x["game_id"]] = per_game.get(x["game_id"], 0) + 1
        if len(pool) >= policy.max_candidates:
            break
    logd = [math.log(american_to_decimal(x["price"])) for x in pool]
    logp = [math.log(x["p"]) for x in pool]
    need_d, need_p = math.log(policy.min_decimal), math.log(policy.min_probability)
    worth = [leg_ev(x) >= policy.min_leg_ev for x in pool]
    passing, closest = [], None
    for n in range(policy.min_legs, policy.max_legs + 1):
        for idx in itertools.combinations(range(len(pool)), n):
            games = {pool[i]["game_id"] for i in idx}
            teams = [t for i in idx for t in (pool[i]["home"] + pool[i]["league"], pool[i]["away"] + pool[i]["league"])]
            if len(games) < n or len(set(teams)) < len(teams):
                continue  # one leg per game, and no team twice (doubleheaders)
            d = sum(logd[i] for i in idx)
            if d < need_d - 1e-12:
                continue
            p = sum(logp[i] for i in idx)
            if closest is None or p > closest["logp"]:
                closest = {"logp": p, "legs": [pool[i] for i in idx], "decimal": math.exp(d)}
            if (p >= need_p - 1e-12 and math.exp(p + d) - 1 >= policy.min_ev - 1e-12
                    and all(worth[i] for i in idx)):
                passing.append((p, d, [pool[i] for i in idx]))
    passing.sort(key=lambda t: (-t[0], -t[1]))
    chosen, taken = [], set()
    for p, d, combo in passing:
        if {x["game_id"] for x in combo} & taken:
            continue
        if policy.min_lower_bound is not None and parlay_band(combo, policy.draws)[1] < policy.min_lower_bound:
            continue
        chosen.append(combo)
        taken |= {x["game_id"] for x in combo}
    near = None
    if closest is not None:
        near = {"p": round(math.exp(closest["logp"]), 4), "decimal": round(closest["decimal"], 3),
                "ev": round(math.exp(closest["logp"]) * closest["decimal"] - 1, 4),
                "legs": [x["label"] for x in closest["legs"]]}
    return chosen, near


def parlay_entry(legs: list[dict], day: str, policy: ParlayPolicy, now: str) -> dict:
    p, lo, hi = parlay_band(legs, policy.draws)
    d = float(np.prod([american_to_decimal(x["price"]) for x in legs]))
    fair = float(np.prod([x["fair"] for x in legs])) if all(x.get("fair") is not None for x in legs) else None
    legs = [dict(x) for x in sorted(legs, key=lambda x: x["start"])]
    reasons = ["each leg's chance keeps only the part of the model's disagreement with the sportsbooks that has "
               "held up on graded games: " + ", ".join(f"{x['label']} {x.get('p_model', x['p']):.0%} → {x['p']:.0%}"
                                                      for x in legs),
               f"{len(legs)} legs from different games, so the hit chance is the product of the legs: "
               + " × ".join(f"{x['p']:.0%}" for x in legs) + f" = {p:.1%}",
               f"80% band {lo:.1%}–{hi:.1%} from simulating each leg's own uncertainty",
               f"pays {d:.2f}x ({decimal_to_american(d):+.0f}); expected return {p * d - 1:+.1%} per unit "
               f"(needs {policy.min_ev:+.0%} or more)",
               "every leg is worth its own price: " + ", ".join(f"{x['label']} {leg_ev(x):+.1%}" for x in legs)]
    if fair is not None:
        reasons.append(f"the sportsbooks' prices make it {fair:.0%} likely; the model says {p:.0%}")
    return {
        "id": parlay_id(day, legs), "tier": "parlay", "date": day, "league": "PARLAY",
        "leagues": sorted({x["league"] for x in legs}), "game_id": None, "start": legs[0]["start"],
        "label": f"{len(legs)}-leg parlay", "legs": legs, "price": round(decimal_to_american(d)),
        "decimal": round(d, 4), "p": round(p, 4), "lower": round(lo, 4), "upper": round(hi, 4),
        "fair": None if fair is None else round(fair, 4), "ev": round(p * d - 1, 4), "stake": 0.0,
        "reasons": reasons, "published_at": now, "status": "pending", "profit": None,
    }


def settle_parlay(e: dict) -> None:
    """Combine graded legs: any loss loses; pushed/void legs drop out (the payout
    shrinks); all remaining legs won wins."""
    legs = e["legs"]
    if any(x["status"] == "lost" for x in legs):
        e.update(status="lost", profit=-1.0)
    elif all(x["status"] in ("won", "push", "void") for x in legs):
        won = [x for x in legs if x["status"] == "won"]
        if not won:
            e.update(status="push", profit=0.0)
        else:
            d = float(np.prod([american_to_decimal(x["price"]) for x in won]))
            e.update(status="won", profit=round(d - 1, 4))
    else:
        return
    e["final"] = " · ".join(f"{x['label']} {x['status']}" for x in legs)
