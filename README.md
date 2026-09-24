# sportsedge

**A calibrated, uncertainty-aware prediction engine for MLB, NHL, NBA and NFL
that publishes a pick only when it is at least 70% confident, and only when
that confidence is worth betting at the offered price.**

Most "AI picks" products output a single number and a win-rate claim.
sportsedge is built around three questions that sharper bettors and serious
investors actually ask:

1. **Is the probability calibrated?** Do the 70% picks win 70% of the time,
   measured out-of-sample, with confidence intervals?
2. **How sure is the model of its own probability?** Every forecast carries an
   80% credible band from Monte Carlo over the model's parameter uncertainty
   and unresolved injury news. A pick must clear the gate at the *lower* end
   of the band, not just at the point estimate.
3. **Does it beat the market, not just pick favourites?** The market's no-vig
   price is a model input (Shin de-vigging, sharp-book-weighted consensus).
   Picks need a positive edge and positive expected value at a price you can
   actually get.

## What's inside

| Layer | Technique | Code |
|---|---|---|
| Team strength | Kalman-filter state-space ratings with full posterior covariance, season regression, robust (Huber) updates, MLE-tuned noise | `models/kalman.py` |
| Scoring (NHL/MLB) | Time-decayed Poisson (Dixon–Coles ρ) / negative-binomial attack–defence–park model, MAP via IRLS, Laplace posterior | `models/count_models.py` |
| Scoring (NFL/NBA) | Discretised-Normal margins with NFL key-number re-weighting, OT tie resolution, totals filter | `models/distributions.py` |
| Diversity member | Margin-of-victory Elo with autocorrelation correction | `models/elo.py` |
| Context | Rest, back-to-backs, travel, time zones, altitude; MLB starter & NHL goalie via empirical-Bayes shrinkage; weather | `models/situational.py` |
| Live news | ESPN injury reports + headlines + any RSS; rule-based NLP or **Claude structured-output extraction**; Bernoulli availability mixtures with reliability and recency decay | `news/` |
| Market | American/decimal odds, multiplicative / additive / power / **Shin** de-vig, log-odds consensus across books, line shopping | `stats/odds.py`, `data/oddsapi.py` |
| Ensemble + calibration | Stacked log-linear opinion pool (penalised logistic regression on out-of-sample forecasts, market included); Platt / Beta / isotonic available | `stats/calibration.py` |
| Uncertainty | Monte Carlo over rating posteriors × rate posteriors × player availability → 80% credible band | `engine.py` |
| Decision | 70% gate + lower-bound gate + EV/edge gate; fractional Kelly with parameter-uncertainty shrinkage; exact simultaneous Kelly | `picks.py`, `stats/kelly.py` |
| Validation | Walk-forward backtest (no look-ahead); log loss/Brier vs market, ECE, reliability, Wilson CIs, exact binomial test vs 70%, bootstrap ROI CI, Kelly drawdown | `backtest.py` |

The full equations are in **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Quick start

```bash
pip install -e .            # numpy, scipy, requests
pip install -e ".[llm]"     # optional: Claude news analysis
pip install pytest && pytest -q

# 1. Verify the statistics end-to-end on synthetic leagues (no network)
sportsedge demo --league NBA NHL

# 2. Build history (results + posted odds) from ESPN
sportsedge fetch-history --league NBA --start 2023-10-24 --end 2025-04-13 --out data/nba.csv

# 3. Honest walk-forward backtest
sportsedge backtest --league NBA --history data/nba.csv --json reports/nba.json

# 4. Train (optionally MLE-tune the Kalman noise) and save state
sportsedge train --league NBA --history data/nba.csv --state nba.pkl --tune

# 5. The daily dashboard (what the GitHub workflow runs every morning)
sportsedge daily --site-dir site        # then open site/index.html
sportsedge demo-site --site-dir site    # preview with synthetic data, no keys needed

# 6. Today's picks in the terminal: live slate + multi-book odds + live news (+ Claude)
export ODDS_API_KEY=...        # https://the-odds-api.com
export ANTHROPIC_API_KEY=...   # optional, for --llm
sportsedge picks --league NBA --state nba.pkl --llm --impacts data/player_impacts.csv
```

History CSV columns: `date, home, away, home_score, away_score` are required.
Optional columns: `home_ml, away_ml, spread` (home line), `home_spread_price,
away_spread_price, total, over_price, under_price, neutral` and `extras`
(JSON, e.g. `{"home_pitcher": {"name": "...", "fip": 3.1, "ip": 120}}`).
Any source works: Retrosheet, nflverse, NHL API, basketball-reference or
licensed feeds, provided team codes are consistent.

A player-impact CSV (`league,team,player,impact,impact_sd,role`) lets you
plug in your own player-value model. For example, points of margin lost when
an NBA star sits (from RAPM/EPM), or a QB's value over the backup.

## The daily dashboard

`sportsedge daily` runs the whole pipeline and writes a self-contained web
page to `site/index.html`. The page has four parts:

* **Today's picks:** a ticket for each qualifying bet. Each ticket shows a
  gate strip with the model's probability and 80% band, the no-vig market,
  the price's break-even and the 70% line, plus edge, EV, stake and the reasons.
* **The board:** every game on the slate. You can search it, filter by market
  (moneyline, spread or total), sort it and expand any row. Expanding a row
  shows every market and exactly which gate condition held it back.
* **Track record:** every published pick graded at its listed price, with
  record, hit rate against claimed probability, units and ROI, a cumulative
  units chart and a 95% interval on the true hit rate.
* **Controls:** league tabs, US/decimal odds and light/dark themes.

`sportsedge demo-site` builds the same page from synthetic leagues with
fictional teams, so you can see the page before any API keys are set up.

### Running it every day (GitHub Actions + GitHub Pages)

`.github/workflows/daily-picks.yml` runs every morning. It:

1. downloads yesterday's results,
2. grades the picks in `data/ledger.json`,
3. retrains,
4. fetches today's slate, odds and news, and
5. commits the updated history and ledger, then publishes the page to
   GitHub Pages.

One-time setup:

1. Merge this branch into your default branch. Scheduled workflows only run
   there.
2. **Settings → Secrets and variables → Actions**. Add `ODDS_API_KEY`, and
   optionally `ANTHROPIC_API_KEY`.
3. **Settings → Pages → Source: GitHub Actions.** Pages on a private
   repository needs a paid GitHub plan. A page published from a public
   repository is public.
4. **Actions → Daily picks → Run workflow.** The first run downloads about
   two seasons per league from ESPN and takes a while.

### API keys

| Service | What it's for | Where to get it | Cost |
|---|---|---|---|
| ESPN site API | schedules, results, injuries, headlines | no key needed | free, unofficial |
| The Odds API | odds from many sportsbooks, no-vig consensus, best price | [the-odds-api.com](https://the-odds-api.com) → sign up → key by email | free tier (500 credits/month) covers a daily 4-league run in the `us` region; paid plans for more |
| Anthropic API (optional) | Claude reads news and injury reports | [console.anthropic.com](https://console.anthropic.com) → API Keys | pay per use |

Each Odds API request costs about (markets × regions) credits. The default
region is `us`. Set `ODDS_API_REGIONS=us,eu` to add Pinnacle to the sharp
consensus, which costs twice the credits. Never commit keys to the repository.
Keep them in GitHub secrets or environment variables.

## Example pick output

```
[NBA] NY @ BOS — BOS ML (-180): p=74.2% (80% band 71.0%–77.1%)  stake 1.10% of bankroll
    - calibrated P(win)=74.2% ≥ 70%
    - 80% credible band 71.0%–77.1% (lower ≥ 62%)
    - price -180 needs 64.3%; EV +15.4% per unit
    - no-vig market 62.1% → edge +12.1%
    - home-win components: kalman 76.0%, elo 71.3%
    - adjustments: back_to_back +1.50, travel +0.30
    - news: Jalen Brunson (NY, questionable) P(plays)=55% impact≈4.00±1.50
```

This example is illustrative, not a real forecast. Every pick explains
itself: the probability, its uncertainty, the price it needs, the edge over
the market, and which models, adjustments and news items moved it.

## Read this before pitching investors

These points are what separate this project from the "90% win rate" tout
services, so I'd lead with them rather than hide them.

* **No performance numbers exist yet.** The system has not been backtested
  on real historical odds in this repository. The synthetic demo shows that
  the math is correct: calibration holds, and 70%+ picks win at their claimed
  rate. It says nothing about real-world edge. Run `sportsedge backtest` on
  several seasons of real closing odds and report the full output, including
  the confidence intervals.
* **70% confidence and profit are different things.** Most games the model
  rates at 70%+ are heavy favourites priced at −250 to −500, which need
  71–83% just to break even. With the EV gate on (the default), expect **few**
  picks: a handful per week per league on real markets, sometimes zero. That
  is the correct outcome, not a bug.
* **The market is the benchmark.** Closing lines are among the most accurate
  forecasts that exist. Credible evidence of skill is (a) out-of-sample
  log loss at or below the market's, and (b) positive closing-line value on
  published picks, sustained over hundreds of bets. A win rate on 30 picks
  proves nothing: the Wilson interval on 21/30 is 52–83%.
* **Small samples lie.** The backtest prints the exact binomial p-value of
  "hit rate > 70%". Show it.
* **Compliance.** Selling picks, taking investor money for a betting
  operation, and marketing performance claims are regulated in many
  jurisdictions (gambling law, consumer-protection and advertising rules, and
  securities law if investors share in betting returns). Talk to a lawyer
  before raising money. Also check the terms of ESPN, The Odds API and any
  other data provider for commercial use.

## Project layout

```
sportsedge/
  engine.py            prediction pipeline (components → MC → stacking)
  daily.py             daily run: history, grading, predictions, ledger
  site.py, web/        the dashboard page (self-contained HTML)
  demo_site.py         dashboard from synthetic leagues (fictional teams)
  picks.py             confidence / EV gate and stake sizing
  backtest.py          walk-forward evaluation and statistics
  config.py            per-league priors and hyperparameters
  models/              kalman, elo, count models, distributions, situational
  stats/               odds & de-vig, calibration & tests, Kelly
  news/                parser, impact mixture, feeds (ESPN/RSS), Claude analyzer
  data/                ESPN scoreboard, The Odds API, CSV I/O
  synthetic.py         ground-truth leagues for verification
tests/                 unit + end-to-end statistical tests
docs/METHODOLOGY.md    equations and validation protocol
```
