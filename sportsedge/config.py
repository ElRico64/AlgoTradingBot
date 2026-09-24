"""Per-league model hyperparameters.

Every number here is a *prior* starting point. The Kalman hyperparameters can be
re-estimated from data by maximum likelihood (models/kalman.py::tune), the
NFL key-number weights can be re-estimated from historical margins
(models/margin.py::fit_key_number_weights), and all probability outputs pass
through calibrators fit strictly out-of-sample. Treat the defaults as
reasonable priors, not as facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeagueConfig:
    league: str
    # "gaussian" (NBA/NFL margin model) or "count" (NHL goals / MLB runs)
    score_model: str
    # --- Kalman state-space ratings (margin units) ---
    obs_sd: float  # sd of a single game's margin around expectation
    prior_rating_sd: float  # sd of team strength at the start of history
    process_sd_per_day: float  # random-walk drift of team strength
    home_adv_prior: float
    season_carryover: float  # rating shrink toward 0 between seasons
    offseason_days: int  # gap (days) that counts as a new season
    margin_cap_sds: float  # winsorize blowouts at this many obs_sd
    # --- totals model (gaussian leagues) ---
    total_obs_sd: float
    league_avg_total: float
    # --- Elo ---
    elo_k: float
    elo_home_adv: float
    elo_points_per_100: float  # margin units per 100 Elo points
    # --- count model (NHL/MLB) ---
    count_dispersion: float | None = None  # NegBin r; None => Poisson
    count_decay_per_day: float = 0.005  # time-decay xi in the likelihood
    count_ridge_sd: float = 0.25  # Normal prior sd on log-rate team effects
    count_park_sd: float = 0.08  # Normal prior sd on venue (park) effects
    count_window_days: int = 400
    max_score: int = 20
    extra_time_home_edge: float = 0.0  # OT/extra-innings home edge
    # --- situational (margin units, home perspective) ---
    back_to_back_penalty: float = 0.0
    rest_advantage_per_day: float = 0.0  # per extra rest day, capped at 3
    travel_penalty_per_1000mi: float = 0.0
    timezone_penalty_per_hour: float = 0.0
    altitude_bonus: float = 0.0
    # --- news / injuries ---
    default_player_impact: float = 0.5
    default_player_impact_sd: float = 0.5
    role_impacts: dict[str, tuple[float, float]] = field(default_factory=dict)
    status_play_prob: dict[str, float] = field(default_factory=dict)
    # NFL-style discrete margin key numbers {margin: mass multiplier}
    key_number_weights: dict[int, float] = field(default_factory=dict)
    tie_mass_factor: float = 1.0  # NFL ties are rare (OT); scales P(margin=0)


BASE_STATUS_PLAY_PROB = {
    "out": 0.0,
    "ir": 0.0,
    "suspended": 0.0,
    "doubtful": 0.15,
    "questionable": 0.55,
    "game-time decision": 0.5,
    "day-to-day": 0.6,
    "probable": 0.9,
    "available": 1.0,
    "active": 1.0,
    "unknown": 0.85,
}


def _status(**overrides: float) -> dict[str, float]:
    d = dict(BASE_STATUS_PLAY_PROB)
    d.update(overrides)
    return d


LEAGUES: dict[str, LeagueConfig] = {
    "NFL": LeagueConfig(
        league="NFL",
        score_model="gaussian",
        obs_sd=13.5,
        prior_rating_sd=6.0,
        process_sd_per_day=0.20,
        home_adv_prior=1.8,
        season_carryover=0.70,
        offseason_days=120,
        margin_cap_sds=2.5,
        total_obs_sd=13.0,
        league_avg_total=44.5,
        elo_k=20.0,
        elo_home_adv=48.0,
        elo_points_per_100=4.0,
        back_to_back_penalty=0.0,
        rest_advantage_per_day=0.25,  # bye weeks / short weeks
        travel_penalty_per_1000mi=0.2,
        timezone_penalty_per_hour=0.3,
        altitude_bonus=0.5,
        default_player_impact=0.4,
        default_player_impact_sd=0.4,
        role_impacts={
            "QB": (4.0, 2.0),
            "star": (1.5, 1.0),
            "starter": (0.5, 0.4),
            "K": (0.4, 0.3),
        },
        status_play_prob=_status(questionable=0.70, doubtful=0.08, probable=0.95),
        key_number_weights={3: 2.4, 7: 1.7, 10: 1.25, 6: 1.2, 4: 1.15, 14: 1.3},
        tie_mass_factor=0.08,
    ),
    "NBA": LeagueConfig(
        league="NBA",
        score_model="gaussian",
        obs_sd=12.0,
        prior_rating_sd=5.0,
        process_sd_per_day=0.12,
        home_adv_prior=2.4,
        season_carryover=0.75,
        offseason_days=90,
        margin_cap_sds=2.5,
        total_obs_sd=18.5,
        league_avg_total=228.0,
        elo_k=20.0,
        elo_home_adv=70.0,
        elo_points_per_100=3.6,
        back_to_back_penalty=1.5,
        rest_advantage_per_day=0.3,
        travel_penalty_per_1000mi=0.3,
        timezone_penalty_per_hour=0.25,
        altitude_bonus=0.8,
        default_player_impact=0.8,
        default_player_impact_sd=0.8,
        role_impacts={"star": (4.0, 1.5), "starter": (1.2, 0.8), "bench": (0.3, 0.3)},
        status_play_prob=_status(questionable=0.50, doubtful=0.10, probable=0.93),
    ),
    "NHL": LeagueConfig(
        league="NHL",
        score_model="count",
        obs_sd=2.35,
        prior_rating_sd=0.45,
        process_sd_per_day=0.012,
        home_adv_prior=0.18,
        season_carryover=0.70,
        offseason_days=80,
        margin_cap_sds=3.0,
        total_obs_sd=2.4,
        league_avg_total=6.1,
        elo_k=6.0,
        elo_home_adv=33.0,
        elo_points_per_100=0.55,
        count_dispersion=None,
        count_decay_per_day=0.004,
        count_ridge_sd=0.20,
        count_park_sd=0.05,
        max_score=15,
        extra_time_home_edge=0.02,
        back_to_back_penalty=0.15,
        rest_advantage_per_day=0.03,
        travel_penalty_per_1000mi=0.03,
        timezone_penalty_per_hour=0.02,
        altitude_bonus=0.05,
        default_player_impact=0.06,
        default_player_impact_sd=0.06,
        role_impacts={"G": (0.25, 0.15), "star": (0.18, 0.08), "starter": (0.06, 0.05)},
        status_play_prob=_status(**{"day-to-day": 0.6}),
    ),
    "MLB": LeagueConfig(
        league="MLB",
        score_model="count",
        obs_sd=4.4,
        prior_rating_sd=0.55,
        process_sd_per_day=0.012,
        home_adv_prior=0.15,
        season_carryover=0.65,
        offseason_days=100,
        margin_cap_sds=2.5,
        total_obs_sd=4.5,
        league_avg_total=8.8,
        elo_k=4.0,
        elo_home_adv=24.0,
        elo_points_per_100=1.1,
        count_dispersion=8.0,  # re-estimated from data by method of moments
        count_decay_per_day=0.006,
        count_ridge_sd=0.15,
        count_park_sd=0.08,
        max_score=25,
        extra_time_home_edge=0.02,
        travel_penalty_per_1000mi=0.02,
        timezone_penalty_per_hour=0.03,
        default_player_impact=0.08,
        default_player_impact_sd=0.06,
        role_impacts={"star": (0.20, 0.08), "starter": (0.08, 0.05), "SP": (0.0, 0.0)},
        status_play_prob=_status(),
    ),
}


def get_config(league: str) -> LeagueConfig:
    try:
        return LEAGUES[league.upper()]
    except KeyError as e:
        raise ValueError(f"Unsupported league {league!r}; choose from {sorted(LEAGUES)}") from e
