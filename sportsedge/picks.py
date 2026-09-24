"""The pick gate.

A pick is published only if ALL of the following hold:

  1. Calibrated probability  p        >= min_probability  (default 0.70)
  2. Lower credible bound    p_10%    >= min_lower_bound  (default 0.62) — the
     model must be confident *in its own confidence*: parameter uncertainty and
     unresolved injury news widen the band and can veto a pick.
  3. The probability comes from a calibrator fit on out-of-sample data.
  4. Expected value of at least +1% per unit at the offered price and an edge
     over the de-vigged market of at least `min_edge` (default on).

Why (4) matters: a 70% favourite priced at -250 (break-even 71.4%) loses
money even if the 70% is exactly right. Win rate alone is not a business;
win rate *at a price* is. Turn it off with `require_positive_ev=False` if you
only want high-probability picks, but then report ROI honestly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .stats.kelly import stake_fraction
from .stats.odds import american_to_decimal, breakeven_probability
from .types import MarketProbability, Pick, Prediction

Z80 = 1.2816


@dataclass
class PickPolicy:
    min_probability: float = 0.70
    min_lower_bound: float = 0.62
    require_calibrated: bool = True
    require_positive_ev: bool = True
    min_edge: float = 0.015
    min_ev: float = 0.01  # +1% per unit: smaller edges are inside model noise and get ~0 stake
    markets: tuple[str, ...] = ("moneyline", "spread", "total")
    max_favorite_price: float = -600.0  # never lay more than this
    kelly_multiplier: float = 0.25
    max_stake: float = 0.03
    one_pick_per_game: bool = True

    def failures(self, m: MarketProbability) -> list[str]:
        """Every gate condition this market fails (empty list = it qualifies)."""
        out = []
        if m.market not in self.markets:
            out.append("market disabled")
        if m.probability < self.min_probability:
            out.append(f"probability {m.probability:.1%} below {self.min_probability:.0%}")
        if m.lower < self.min_lower_bound:
            out.append(f"too uncertain: lower bound {m.lower:.1%} below {self.min_lower_bound:.0%}")
        if self.require_calibrated and not m.calibrated:
            out.append("no out-of-sample calibration for this market yet")
        if m.price is not None and m.price < self.max_favorite_price:
            out.append(f"price {m.price:+.0f} shorter than {self.max_favorite_price:+.0f}")
        if self.require_positive_ev:
            if m.price is None or m.fair_market_prob is None:
                out.append("no market price available")
            else:
                ev = m.expected_value or 0.0
                if ev <= 0:
                    out.append(f"negative value: price {m.price:+.0f} needs "
                               f"{breakeven_probability(m.price):.1%}, EV {ev:+.1%}")
                elif ev < self.min_ev:
                    out.append(f"value too thin: EV {ev:+.1%} below {self.min_ev:+.0%}")
                elif (m.edge or 0) < self.min_edge:
                    out.append(f"edge {m.edge:+.1%} over the market below {self.min_edge:.1%}")
        return out

    def evaluate(self, pred: Prediction) -> list[Pick]:
        out = []
        for m in pred.markets:
            if self.failures(m):
                continue
            reasons = [f"calibrated P(win)={m.probability:.1%} ≥ {self.min_probability:.0%}",
                       f"80% credible band {m.lower:.1%}–{m.upper:.1%} (lower ≥ {self.min_lower_bound:.0%})"]
            if m.price is not None:
                be = breakeven_probability(m.price)
                reasons.append(f"price {m.price:+.0f} needs {be:.1%}; EV {m.expected_value:+.1%} per unit")
            if m.fair_market_prob is not None:
                reasons.append(f"no-vig market {m.fair_market_prob:.1%} → edge {m.edge:+.1%}")
            comps = ", ".join(f"{k} {v:.1%}" for k, v in pred.components.items())
            reasons.append(f"home-win components: {comps}")
            if pred.adjustments:
                top = sorted(pred.adjustments.items(), key=lambda kv: -abs(kv[1]))[:4]
                reasons.append("adjustments: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))
            reasons.extend(f"news: {n}" for n in pred.notes)
            stake = 0.0
            if m.price is not None:
                # Pushes are refunded, so Kelly applies to the no-push probability.
                p_sd = max(1e-4, (m.upper - m.lower) / (2 * Z80))
                stake = stake_fraction(m.probability, p_sd, american_to_decimal(m.price),
                                       self.kelly_multiplier, self.max_stake)
            out.append(Pick(pred.game, m, stake, reasons))
        out.sort(key=lambda p: -(p.market.expected_value or p.market.probability))
        return out[:1] if (self.one_pick_per_game and out) else out

    def select(self, preds: Iterable[Prediction]) -> list[Pick]:
        picks = [p for pred in preds for p in self.evaluate(pred)]
        picks.sort(key=lambda p: -(p.market.expected_value or 0))
        return picks
