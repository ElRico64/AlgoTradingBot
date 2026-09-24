"""Synthetic leagues with known ground truth.

Used to *verify the machinery* — that ratings recover latent strength, that
calibration works out-of-sample and that the 70% gate selects picks which
win at the rate the model claims. A synthetic backtest says nothing about
real-world edge: the simulated market is deliberately noisier than real
betting markets. Real validation requires real historical data + odds.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import norm

from .config import get_config
from .models.distributions import count_final_distributions, gaussian_margin_dist, joint_score_matrix
from .stats.odds import decimal_to_american
from .teams import REGISTRY
from .types import GameResult, MarketOdds

SEASON = {  # (start month, day, n_days, game-day spacing, share of teams playing per game day)
    "NFL": (9, 7, 126, 7, 1.0),
    "NBA": (10, 22, 170, 1, 0.5),
    "NHL": (10, 8, 180, 1, 0.47),
    "MLB": (3, 28, 185, 1, 0.9),
}


def _price(p: float, vig: float) -> float:
    return decimal_to_american(1.0 / min(0.995, p * (1 + vig)))


def _market(p_true: float, noise: float, vig: float, rng) -> tuple[float, float, float]:
    lo = math.log(p_true / (1 - p_true)) + rng.normal(0, noise)
    pm = 1 / (1 + math.exp(-lo))
    return pm, _price(pm, vig), _price(1 - pm, vig)


def generate(league: str, seasons: int = 2, seed: int = 0, market_noise: float = 0.12,
             vig: float = 0.045, start_year: int = 2022) -> list[GameResult]:
    cfg = get_config(league)
    rng = np.random.default_rng(seed)
    teams = list(REGISTRY[league])
    n = len(teams)
    month, dom, n_days, spacing, share = SEASON[league]
    count = cfg.score_model == "count"
    if count:
        att = rng.normal(0, 0.14, n)
        dfn = rng.normal(0, 0.14, n)
        park = rng.normal(0, 0.05 if league == "MLB" else 0.02, n)
        mu, home = math.log(cfg.league_avg_total / 2), 0.05
        r = cfg.count_dispersion
    else:
        theta = rng.normal(0, cfg.prior_rating_sd, n)
        scoring = rng.normal(0, 0.25 * cfg.total_obs_sd, n)
    out: list[GameResult] = []
    gid = 0
    for s in range(seasons):
        start = datetime(start_year + s, month, dom, 19, 0)
        if s > 0:
            if count:
                att = 0.7 * att + rng.normal(0, 0.14 * math.sqrt(1 - 0.49), n)
                dfn = 0.7 * dfn + rng.normal(0, 0.14 * math.sqrt(1 - 0.49), n)
            else:
                theta = cfg.season_carryover * theta + rng.normal(0, cfg.prior_rating_sd * math.sqrt(1 - cfg.season_carryover ** 2), n)
        for day in range(0, n_days, spacing):
            when = start + timedelta(days=day)
            if count:
                att += rng.normal(0, 0.004 * math.sqrt(spacing), n)
                dfn += rng.normal(0, 0.004 * math.sqrt(spacing), n)
            else:
                theta += rng.normal(0, cfg.process_sd_per_day * math.sqrt(spacing), n)
            playing = rng.permutation(n)[: int(n * share) // 2 * 2]
            for i in range(0, len(playing), 2):
                h, a = int(playing[i]), int(playing[i + 1])
                gid += 1
                if count:
                    lh = math.exp(mu + home + att[h] + dfn[a] + park[h])
                    la = math.exp(mu + att[a] + dfn[h] + park[h])
                    J = joint_score_matrix(np.array([lh]), np.array([la]), cfg.max_score, r)
                    ot = np.array([min(0.95, max(0.05, 0.5 + cfg.extra_time_home_edge + 0.5 * (lh - la) / (lh + la)))])
                    md, td = count_final_distributions(J, ot)
                    if r is None:
                        hs, as_ = rng.poisson(lh), rng.poisson(la)
                    else:
                        hs = rng.negative_binomial(r, r / (r + lh))
                        as_ = rng.negative_binomial(r, r / (r + la))
                    if hs == as_:
                        if rng.random() < ot[0]:
                            hs += 1
                        else:
                            as_ += 1
                    fav_home = lh > la
                    spread = -1.5 if fav_home else 1.5
                    exp_total = lh + la
                    total_line = math.floor(exp_total) + 0.5
                else:
                    mean_m = theta[h] - theta[a] + cfg.home_adv_prior
                    mean_t = cfg.league_avg_total + scoring[h] + scoring[a]
                    md = gaussian_margin_dist(np.array([mean_m]), cfg.obs_sd,
                                              cfg.key_number_weights or None, cfg.tie_mass_factor, 0.5,
                                              ties_allowed=league == "NFL", span=90)
                    margin = int(rng.choice(md.support, p=md.pmf[0]))
                    tot = max(abs(margin) + 2, int(round(rng.normal(mean_t, cfg.total_obs_sd))))
                    if (tot + margin) % 2:
                        tot += 1
                    hs, as_ = (tot + margin) // 2, (tot - margin) // 2
                    spread = -round(mean_m * 2) / 2
                    total_line = round(mean_t * 2) / 2
                    p_over_true = 1 - norm.cdf((total_line - mean_t) / cfg.total_obs_sd)
                p_home = float(md.win()[0] / max(1e-9, 1 - md.tie()[0]))
                pm, hml, aml = _market(p_home, market_noise, vig, rng)
                pc, pu = md.cover(spread)
                p_cover = float(pc[0] / max(1e-9, 1 - pu[0]))
                _, hsp, asp = _market(min(0.97, max(0.03, p_cover)), market_noise * 0.7, vig, rng)
                if count:
                    po, pu2 = td.over(total_line)
                    p_over_true = float(po[0] / max(1e-9, 1 - pu2[0]))
                _, op, up = _market(min(0.97, max(0.03, p_over_true)), market_noise * 0.7, vig, rng)
                out.append(GameResult(
                    game_id=f"SYN-{league}-{gid}", league=league, start_time=when, home=teams[h], away=teams[a],
                    home_score=int(hs), away_score=int(as_),
                    odds=MarketOdds(home_ml=round(hml), away_ml=round(aml), spread=spread,
                                    home_spread_price=round(hsp), away_spread_price=round(asp),
                                    total=total_line, over_price=round(op), under_price=round(up),
                                    source="synthetic")))
    return out
