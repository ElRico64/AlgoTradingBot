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

## 5b. Pattern engine: regime detection + residual learning (`models/patterns.py`)

The structural models (§2–§5) encode what we know about sports. The pattern
engine looks for what they miss. It has three parts.

**Regime detection: a sticky hidden Markov model** (`models/regimes.py`).

* **Input.** Each team's standardised surprise z<sub>t</sub> = (margin −
  E[margin]) / sd, from the Kalman model and from the team's point of view.
  Before it updates a team, the surprise is taken net of the opponent's
  expected form (credit attribution).
* **Model.** A 3-state chain {cold, normal, hot} with Gaussian emissions.
* **Fitting.** Pooled Baum–Welch EM with scaled forward–backward recursions,
  plus a **sticky Dirichlet prior** on the transitions (Fox et al., 2011;
  MAP-EM).
* **Why the prior.** Without it, EM on noisy surprises collapsed into a
  memoryless mixture (self-transition ≈ 0.55) whose "form" signal was noise.
  With it, regimes persist (≈ 0.92) and the signal's correlation with the
  next result doubled.
* **Output.** Before each game, the filter gives P(cold/normal/hot) and the
  expected surprise.

**Residual learning in two stages.** Both stages start from the structural
logit, so they only learn corrections.

* **Features (27, all pre-game).** Structural logit and its uncertainty;
  rest and back-to-backs; travel and time zones; regime form and P(hot) −
  P(cold); fast EWMA form; home and road splits; head-to-head EWMA; streaks;
  volatility; season phase; games played; pace; neutral site.
* **Stage 1 (main effects).** Ridge logistic regression with a fixed offset.
* **Stage 2 (interactions).** Histogram gradient-boosted trees
  (`models/boosting.py`): second-order Newton leaves −G/(H+λ) and gain
  G<sub>L</sub>²/(H<sub>L</sub>+λ) + G<sub>R</sub>²/(H<sub>R</sub>+λ) −
  G²/(H+λ), the XGBoost/LightGBM algorithm written in NumPy.
* **Model selection.** The ridge penalty and the number of trees are chosen
  by **expanding-window time-series cross-validation** (3 folds, never
  shuffled).
* **Leak found and fixed.** An earlier version refit stage 1 on the
  validation block and then used that block to stop stage 2 early, so the
  trees always looked useless.

**Earn-your-place gate.** The correction is applied only when the
cross-validated held-out log-loss gain on the league's own history is at
least 0.0015. Otherwise the engine switches itself off. Its forecasts enter
the stacker (§6) as one more component, with a non-negative weight fit on
out-of-sample games only.

**Validation** (`sportsedge validate-patterns`). Synthetic leagues are
simulated with and without *planted* patterns (sticky hot/cold regimes of
±0.35 σ and a no-rest-after-long-travel penalty of 0.3 σ). Each is
backtested walk-forward with the engine on and off. Reported numbers are
out-of-sample log loss (lower is better):

| League | Planted patterns | Engine off | Engine on | Change |
|---|---|---|---|---|
| NBA | none | 0.6223 | 0.6226 | +0.0003 (noise level) |
| NBA | yes | 0.5959 | 0.5960 | +0.0001 (no gain) |
| NHL | none | 0.6639 | 0.6638 | −0.0001 |
| NHL | yes | 0.6555 | 0.6540 | **−0.0015** (finds part of the pattern) |

The values are model-only log loss over about 1,800–2,000 walk-forward games
per row (2 simulated seasons, seed 5). With the market in the blend, every
row is within ±0.0002. The simulated market already knows the truth, so there
is nothing left for the engine to add.

**What these numbers say.** An oracle that knew the planted effects
exactly would improve NBA log loss by about 0.020. Most of that is
unrecoverable: a team's true regime is revealed only slowly and noisily by
its results. With about two seasons of history, the learnable part is small,
at best about 0.001–0.003. The gate keeps the engine from adding noise when
it has nothing to offer. On real data, run
`sportsedge backtest` with the engine on and off (`EngineSettings(use_patterns=...)`)
before relying on it.

---

## 6. Ensemble + market: stacked log-linear pooling (`stats/calibration.py`)

  logit p = b₀ + Σ<sub>k</sub> w<sub>k</sub>·logit p<sub>k</sub> + w<sub>m</sub>·logit p<sub>market</sub>

The weights come from L2-penalised logistic regression fit **only on
out-of-sample (walk-forward) forecasts**. The weights are constrained to be
**non-negative** (Breiman, 1996, "stacked regressions"), and the fit uses a
bounded L-BFGS-B solver.

The constraint is not optional. The components and the market are highly
collinear: Kalman and Elo logits correlate at about 0.95. Without the
constraint, a backtest gave the Kalman model a weight of −0.40, and news
flows through the components, so a star's injury made the *injured* team
more likely to win. With non-negative weights, bad news for a team can only
lower its probability. A test enforces this.

With the constraint in place, this single step:

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

**News and the market.** When a price is available, the blend leans mostly on
the market. Market weights of 0.7–0.9 are typical, because closing lines are
very accurate. Sportsbooks re-price within minutes of major news, and the
board re-pulls odds on every refresh. So a late scratch reaches the forecast
mostly through the moved line; adding the full injury effect on top of a line
that already moved would double-count it. When no price is available, the
news mixture model (§5) drives the forecast directly.

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

Condition 4 means at least +1% expected value per unit. Smaller edges sit
inside model noise, and their uncertainty-shrunk Kelly stake is about 0.

**Two published tiers.** The rules above define **Best bets**. The **70%+
picks** tier drops condition 4 (and 5). It covers moneylines and totals only:
run and puck lines at +1.5 are 70%+ almost every night, and their price
already reflects it. Each 70%+ pick carries a *value line*, the worst
American price at which it still has EV ≥ +1%:

  d<sub>min</sub> = (1 + 0.01)/p,  e.g. p = 0.75 → −288

Both tiers are graded separately on the record.

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
