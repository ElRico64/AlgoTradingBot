"""Margin-of-victory Elo (a deliberately different, robust ensemble member).

Update:  R <- R + K * M * (s - E),  E = 1 / (1 + 10^(-(dR + HFA)/400))
with the autocorrelation-corrected MOV multiplier
         M = ln(|mov| + 1) * 2.2 / (0.001 * dR_winner + 2.2)
and between-season reversion toward the mean. Elo is less statistically
efficient than the Kalman filter but has very different failure modes, which
is exactly what makes it useful inside the stacked ensemble.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EloRatings:
    k: float
    home_adv: float
    carryover: float = 0.67
    offseason_days: int = 90
    mean: float = 1500.0
    ratings: dict[str, float] = field(default_factory=dict)
    last_time: Optional[datetime] = None

    def _get(self, team: str) -> float:
        return self.ratings.setdefault(team, self.mean)

    def advance(self, when: datetime) -> None:
        if self.last_time is not None and (when - self.last_time).days >= self.offseason_days:
            for t, r in self.ratings.items():
                self.ratings[t] = self.mean + self.carryover * (r - self.mean)
        self.last_time = when

    def diff(self, home: str, away: str, neutral: bool = False) -> float:
        return self._get(home) - self._get(away) + (0.0 if neutral else self.home_adv)

    def prob(self, home: str, away: str, neutral: bool = False) -> float:
        return 1.0 / (1.0 + 10 ** (-self.diff(home, away, neutral) / 400.0))

    def update(self, home: str, away: str, margin: float, neutral: bool = False,
               when: Optional[datetime] = None) -> None:
        if when is not None:
            self.advance(when)
        d = self.diff(home, away, neutral)
        e = 1.0 / (1.0 + 10 ** (-d / 400.0))
        s = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_d = d if margin > 0 else -d
        mult = math.log(abs(margin) + 1.0) * 2.2 / (0.001 * winner_d + 2.2) if margin != 0 else 1.0
        delta = self.k * mult * (s - e)
        self.ratings[home] = self._get(home) + delta
        self.ratings[away] = self._get(away) - delta
