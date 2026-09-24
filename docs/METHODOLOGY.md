# sportsedge — Methodology

This document specifies every statistical component, the equations behind it,
and how the components combine into a published pick. Each section names the
code that implements it.

---

## 1. Problem statement

For a game *g*, we want a **calibrated** probability for each betting outcome
(moneyline, spread, total). We publish a pick only when that probability is at
least **70%**, when we are confident in the probability itself, and (by
default) when the bet has positive expected value at the price offered.

"Calibrated" means that across all forecasts of *p*, the event happens a
fraction *p* of the time. Calibration is measured out-of-sample (§8). It is
never assumed.

---

## 2. Dynamic team strength: Kalman state-space model (`models/kalman.py`)

State vector: **x**<sub>t</sub> = [h, θ<sub>1</sub>, …, θ<sub>n</sub>]. Here *h* is
home advantage and θ<sub>i</sub> is team *i*'s strength in margin units
(points, goals or runs).

**Transition** (random walk, scaled by elapsed calendar time Δ):

  θ<sub>t+Δ</sub> = θ<sub>t</sub> + w,  w ~ N(0, q²Δ I)

**Season break** (gap ≥ `offseason_days`): the ratings regress toward the mean,
and the regression inflates the covariance:

  θ ← ρθ,  P ← ρ²P + (1−ρ²)τ²I

**Observation** for a game between home team *i* and away team *j*:

  y = H**x** + ε,  H = [𝟙<sub>not neutral</sub>, +e<sub>i</sub>, −e<sub>j</sub>],  ε ~ N(0, σ²)

**Update** (standard Kalman recursions, with a Huber-clipped innovation for
robustness to blowouts):

  S = HPHᵀ + σ²,  K = PHᵀ/S,  **x** ← **x** + K·clip(y − H**x**, ±c√S),  P ← P − KSKᵀ

**Posterior predictive** win probability for a Gaussian margin:

  P(home wins) = Φ( H**x̂** / √(σ² + HPHᵀ) )

The HPHᵀ term is *epistemic* uncertainty: how unsure the model is about the
strengths themselves. It is large early in a season or for teams with little
data, and it feeds the credible band in §7.

**Hyperparameter estimation.** σ and q are estimated by maximising the
prediction-error decomposition of the likelihood:

  log L(σ, q) = Σ<sub>t</sub> −½ [ log(2πS<sub>t</sub>) + v<sub>t</sub>²/S<sub>t</sub> ]

This is run with `sportsedge train --tune`. A second filter with
H = [1, +e<sub>i</sub>, +e<sub>j</sub>] models game **totals**.

---

## 3. Margin-of-victory Elo (`models/elo.py`)

This is a deliberately different ensemble member:

  R ← R + K·M·(s − E),  E = 1/(1 + 10<sup>−(ΔR + HFA)/400</sup>)

It uses the autocorrelation-corrected MOV multiplier
M = ln(|mov|+1)·2.2/(0.001·ΔR<sub>winner</sub> + 2.2), with regression to the
mean between seasons. The stacker (§6) decides how much weight Elo earns. In
backtests the weight is often near zero, and that is useful information too.

---

## 4. Score models for NHL and MLB (`models/count_models.py`, `models/distributions.py`)

  log λ<sub>home</sub> = μ + h + att<sub>home</sub> + def<sub>away</sub> + park<sub>venue</sub>
  log λ<sub>away</sub> = μ + att<sub>away</sub> + def<sub>home</sub> + park<sub>venue</sub>

* **NHL:** goals ~ Poisson(λ), with the **Dixon–Coles** low-score correction
  τ(x, y; λ, μ, ρ) applied to the cells (0,0), (0,1), (1,0) and (1,1). ρ is
  estimated by profile likelihood.
* **MLB:** runs ~ **negative binomial** NB2(λ, r) to capture overdispersion.
  r is estimated by weighted method of moments:
  r̂ = Σwλ² / Σw[(y−λ)² − λ].
* **Estimation:** the MAP estimate under Gaussian (ridge) priors, with
  Dixon–Coles time-decay weights w = e<sup>−ξ·age</sup>. It is solved by
  Fisher scoring (IRLS):
  β ← β + (XᵀWX + Λ)⁻¹[Xᵀw(y−λ)·r/(r+λ) − Λβ].
* **Park effects** are *estimated* with a tight prior rather than hard-coded.
* **Hierarchical prior by empirical Bayes:** the prior sd τ of the team
  effects is not hand-set. It is re-estimated by EM:
  τ² = mean(β̂² + diag Cov β̂). A prior that is too loose under-shrinks and
  makes extreme matchups overconfident. In backtests this was a measurable
  source of miscalibration on puck lines, and the EM step removed it.
* **Uncertainty:** a Laplace approximation, Cov(β) ≈ (XᵀWX + Λ)⁻¹. From it
  we draw (log λ<sub>h</sub>, log λ<sub>a</sub>) jointly.
* **Regulation ties** go to OT or extra innings. The home team wins that
  period with probability 0.5 + edge + ½(λ<sub>h</sub>−λ<sub>a</sub>)/(λ<sub>h</sub>+λ<sub>a</sub>),
  by one goal or run. That is what makes puck lines, run lines and totals
  settle correctly.

**NFL/NBA final margins** use a Normal distribution discretised to integers.
For the NFL it is re-weighted at key numbers (3, 7, 10, 6, 4, 14), because
roughly 15% of NFL games land on exactly 3. The weights are configurable and
should be estimated from historical margins. Regulation ties are resolved
through OT.

---

## 5. Situational and news adjustments

**Situational** (`models/situational.py`) covers rest differential,
back-to-backs, travel distance (haversine), time-zone shifts, altitude, MLB
starting pitchers, NHL starting goalies, and weather. Pitcher and goalie
quality are shrunk by empirical Bayes:

  x̂ = (n·x<sub>obs</sub> + k·x<sub>league</sub>)/(n + k)

where k is the stabilisation sample (about 80 IP for FIP, about 2000 shots for
save %).

**News** (`news/`) enters as a **mixture model**. Each uncertain player *j*
plays with probability q<sub>j</sub>:

  P(win) = Σ<sub>scenarios s</sub> P(s)·P(win | s),
  q = 1 − r·(1 − q<sub>status</sub>),  r = reliability·e<sup>−max(0, age−24h)/96h</sup>

* q<sub>status</sub> comes from league priors (for example, NFL "questionable"
  ≈ 0.70 and NBA "questionable" ≈ 0.50).
* The player's value is itself uncertain: impact ~ N(μ, sd²), truncated at 0.
  It comes from a user-supplied table (RAPM, EPM, WAR, QB EPA and so on), from
  role defaults, or from the LLM's importance tier.
* The consequence is that a "questionable" star **widens** the probability
  band. That can push a pick below the confidence gate, which is the intended
  behaviour.

**LLM news extraction** (`news/llm.py`, optional) uses Claude structured
outputs to turn messy prose into schema-validated
`{team, player, status, role, importance, reliability}` records. The model
only extracts facts; all of the probability math stays in the transparent
pipeline above.

---

## 6. Ensemble + market: stacked log-linear pooling (`stats/calibration.py`)

  logit p = b₀ + Σ<sub>k</sub> w<sub>k</sub>·logit p<sub>k</sub> + w<sub>m</sub>·logit p<sub>market</sub>

The weights come from L2-penalised logistic regression, solved by Newton–IRLS.
It is fit **only on out-of-sample (walk-forward) forecasts**. This single
step:

* learns how much each model deserves,
* learns how much to trust the market, and
* calibrates the result.

The market's no-vig probability comes from **Shin's (1993) method**:

  p<sub>i</sub> = [√(z² + 4(1−z)π<sub>i</sub>²/Π) − z] / (2(1−z)),  Σp<sub>i</sub> = 1

Shin's method corrects the favourite–longshot bias better than proportional
normalisation. With multiple books, the de-vigged probabilities are pooled in
log-odds space, with sharp books (Pinnacle, Circa) weighted up.

Spread and total probabilities are stacked with their own market prices in the
same way. They use **Beta-calibration features** [ln p, −ln(1−p)] instead of a
single logit, so the two tails can be corrected independently. Both choices
came out of backtest diagnostics:

* Without market stacking, the selected 70%+ picks were **overconfident**.
  This is the winner's curse: selecting on model/market disagreement selects
  model error.
* With a single-slope logit stacker, NHL puck-line picks were still
  overconfident in the tail (claimed 74%, hit 62%, z = −2.6). After the
  hierarchical-prior fix (§4) and Beta features, the same backtest gives
  claimed 73.3% against 72.5% hit (z = −0.2).

Platt scaling, Beta calibration (Kull et al., 2017) and isotonic regression
(PAV) are also available.

---

## 7. Uncertainty: Monte Carlo over everything (`engine.py`)

For N draws (2000 by default), we sample:

* team strengths from N(**x̂**, P),
* count-model rates from the Laplace posterior, and
* each uncertain player's availability and impact.

Every draw is pushed through every component and then through the stacker.
The 10th–90th percentile of the resulting probabilities is the **80% credible
band**. Its lower end is a second, independent gate.

---

## 8. The pick gate (`picks.py`)

A pick is published only if **all** of these hold:

| # | Condition | Default |
|---|---|---|
| 1 | calibrated probability p ≥ | **0.70** |
| 2 | lower 80% credible bound ≥ | 0.62 |
| 3 | probability produced by an out-of-sample-fitted calibrator | required |
| 4 | edge over no-vig market ≥ and EV > 0 at the offered price | 1.5%, on |
| 5 | price not worse than | −600 |

**Why condition 4 exists.** A 72% favourite at −300 has a break-even of 75%,
so it loses money *even if the 72% is exactly right*. High confidence is
mostly found on heavy favourites, and heavy favourites are priced
accordingly. You can disable the condition with `--no-ev`, but then publish
ROI alongside hit rate.

**Stake sizing** uses fractional Kelly with a parameter-uncertainty shrinkage
(Baker & McHale, 2013):

  f = ¼ · f<sub>Kelly</sub> · e²/(e² + d²s²),  e = pd − 1

Here s is the posterior sd of p. There is also an exact simultaneous-Kelly
optimiser for concurrent bets (`stats/kelly.py`).

---

## 9. Validation protocol (`backtest.py`)

The backtest walks forward one day at a time:

1. Refit the calibrators on forecasts made strictly *before* today.
2. Predict today's games.
3. Settle the picks.
4. Only then learn from today's results.

Reported:

* **Forecast quality:** log loss and Brier score against the market (the only
  benchmark that matters), plus the Murphy decomposition.
* **Calibration:** ECE, a reliability table with Wilson intervals, and a
  Poisson-binomial z-test of pick hits against claimed probabilities.
* **Pick record:** hit rate with a 95% Wilson interval, and an exact one-sided
  binomial test of H₀: rate ≤ 70%.
* **Money:** flat-stake ROI with a bootstrap 95% CI, plus a fractional-Kelly
  bankroll path with maximum drawdown.

**Synthetic checks** (`sportsedge demo`, leagues with known truth):
walk-forward ECE is 0.007–0.036, and gated picks win at their claimed rate.
For example, MLB went 61–25 (70.9%) against 72.5% claimed, and NHL went
50–19 (72.5%) against 73.3% claimed, both with |z| < 1. Flat ROI is about
0%, as it should be when the model roughly *matches* a market it cannot
see through. What a synthetic run does *not* show: real-world edge. The simulated
market is noisier than real markets by construction.

---

## 10. Known limitations

* Default coefficients (rest, travel, weather, key numbers, status priors)
  are literature-informed priors. They should be re-estimated on real data.
* Team ratings already partly price in long-term injuries. The engine
  discounts IR and season-long absences, but a player-level model (lineup-
  adjusted ratings) would do better.
* ESPN's API is unofficial. The Odds API needs a paid key for volume. Check
  each provider's terms before commercial use.
* Closing lines are very efficient. Expect few qualifying picks per week. The
  honest measure of skill is **closing-line value** and out-of-sample
  log-loss against the market, not a hit rate.
