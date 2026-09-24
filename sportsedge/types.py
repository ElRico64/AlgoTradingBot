"""Core data types shared across the engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Game:
    """A scheduled game. `extras` carries sport-specific context such as
    probable pitchers, starting goalies or weather (see models/situational.py)."""

    game_id: str
    league: str
    start_time: datetime
    home: str
    away: str
    neutral: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameResult(Game):
    home_score: int = 0
    away_score: int = 0
    # Market prices observed for this game (typically closing). Used for
    # stacking the model with the market and for backtest ROI.
    odds: Optional["MarketOdds"] = None

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    @property
    def total(self) -> int:
        return self.home_score + self.away_score


@dataclass
class MarketOdds:
    """American odds for the three main markets. Any field may be missing."""

    home_ml: Optional[float] = None
    away_ml: Optional[float] = None
    # Spread is quoted from the home side: -3.5 means home favoured by 3.5.
    spread: Optional[float] = None
    home_spread_price: Optional[float] = -110.0
    away_spread_price: Optional[float] = -110.0
    total: Optional[float] = None
    over_price: Optional[float] = -110.0
    under_price: Optional[float] = -110.0
    source: str = "unknown"
    # Optional consensus no-vig probabilities (e.g. pooled across books). When
    # absent they are derived from the prices above by de-vigging.
    fair_home_prob: Optional[float] = None
    fair_home_cover_prob: Optional[float] = None
    fair_over_prob: Optional[float] = None

    def has_moneyline(self) -> bool:
        return self.home_ml is not None and self.away_ml is not None


@dataclass
class NewsEvent:
    """A structured piece of news that can move a game's probability."""

    league: str
    team: str
    player: Optional[str]
    event_type: str  # injury | rest | suspension | trade | lineup | weather | other
    status: str  # out | doubtful | questionable | probable | available | unknown
    published: Optional[datetime] = None
    source: str = "unknown"
    reliability: float = 0.9  # 0..1, how much to trust the report
    role: Optional[str] = None  # e.g. QB, G (goalie), SP, star, starter
    impact: Optional[float] = None  # margin units (pts/goals/runs) if known
    impact_sd: Optional[float] = None
    text: str = ""


@dataclass
class MarketProbability:
    market: str  # moneyline | spread | total
    side: str  # home | away | over | under
    line: Optional[float]
    probability: float  # calibrated probability the side wins (pushes excluded)
    lower: float  # lower credible bound on that probability
    upper: float
    price: Optional[float] = None  # American odds offered, if known
    fair_market_prob: Optional[float] = None  # de-vigged market probability
    push_probability: float = 0.0
    calibrated: bool = True  # False until an out-of-sample calibrator exists

    @property
    def edge(self) -> Optional[float]:
        if self.fair_market_prob is None:
            return None
        return self.probability - self.fair_market_prob

    @property
    def expected_value(self) -> Optional[float]:
        """Expected profit per 1 unit staked, pushes refunded."""
        if self.price is None:
            return None
        from .stats.odds import american_to_decimal

        d = american_to_decimal(self.price)
        p_win = self.probability * (1.0 - self.push_probability)
        p_loss = (1.0 - self.probability) * (1.0 - self.push_probability)
        return p_win * (d - 1.0) - p_loss


@dataclass
class Prediction:
    game: Game
    home_win_prob: float
    home_win_lower: float
    home_win_upper: float
    expected_margin: float
    expected_total: float
    markets: list[MarketProbability] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    adjustments: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Pick:
    game: Game
    market: MarketProbability
    stake_fraction: float
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        m = self.market
        g = self.game
        if m.market == "moneyline":
            team = g.home if m.side == "home" else g.away
            what = f"{team} ML"
        elif m.market == "spread":
            team = g.home if m.side == "home" else g.away
            line = m.line if m.side == "home" else -m.line
            what = f"{team} {line:+g}"
        else:
            what = f"{m.side.upper()} {m.line:g}"
        price = f" ({m.price:+.0f})" if m.price is not None else ""
        return (
            f"[{g.league}] {g.away} @ {g.home} — {what}{price}: "
            f"p={m.probability:.1%} (80% band {m.lower:.1%}–{m.upper:.1%})"
        )
