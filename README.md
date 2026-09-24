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

## The dashboard

`sportsedge daily` runs the pipeline and writes a self-contained web page to
`site/index.html`. It is safe to run every 30 minutes. The page has these
parts:

* **Top pick:** the day's strongest pick, with team colors, a confidence
  ring, the price and the worst price still worth taking, a countdown to the
  start, and the latest injury news for that game.
* **Today's picks, in two tiers:**
  * **Best bets:** at least 70% likely *and* worth betting at today's price
    (expected value ≥ +1% per unit). These need a posted price.
  * **70%+ picks:** every moneyline or total the model gives at least 70%.
    Favourites are often priced too short to profit, so each card shows
    "worth it at": the worst price at which the pick still pays over time.
* **All games:** every game on the day's slate (the whole week for the NFL).
  Each card shows win probabilities as a bar in the two teams' colors, the
  projected score, the total lean, the posted price, injury-update chips, and
  a live score or final once the game starts. Tapping a card shows every
  market, the model's reasoning and the news for that game.
* **Track record:** each tier's published picks, graded at the price they
  were published at: record, hit rate against the model's stated
  probability, units and ROI, a units chart, and a 95% range for the true
  hit rate.
* **News wire:** injury and lineup updates for the teams playing today.
* **Settings and live data:** league tabs, US or decimal odds, and light or
  dark theme. On the hosted site, the page pulls each new refresh
  automatically.

`sportsedge demo-site` builds the same page from synthetic leagues with
fictional teams and players, so you can see it before anything is set up.

### How a game day works

* **First run of the day:** downloads yesterday's results, retrains every
  model and caches them for the day.
* **Every 30 minutes after that:**
  * pulls the scoreboard (upcoming, live and final games), fresh odds and
    fresh injury news;
  * re-predicts every game that hasn't started, and grades finished picks.
* **At the start of a game:** its prediction locks.
* **Withdrawals:** if news before the start pushes a published pick below the
  70% bar, the pick is marked *withdrawn* and stays visible in the record. It
  is never silently deleted.
* **Published prices:** a published pick stays on the record at the price it
  was published at.

### Do you need an odds API key?

**No.** ESPN's free feed covers everything the model itself needs:

* every game each day for all four leagues;
* results and live scores;
* probable pitchers;
* injury reports and headlines;
* a posted moneyline, spread and total for most games.

Winners, over/unders and 70%+ picks all work without a key. The Odds API is
an upgrade:

| | ESPN only (no key) | + The Odds API |
|---|---|---|
| Win / total / spread probabilities | yes | yes |
| 70%+ picks with a "worth it at" price | yes | yes |
| Best bets (value at the price) | from one posted line, when ESPN has it | best price across many sportsbooks |
| Market consensus the model blends with | one book | sharp-weighted consensus of many books |

Posted prices matter for accuracy: sportsbook lines are among the strongest
predictors there are, and the model blends with them. Without any price the
board still works on the model alone.

### Running it automatically (GitHub Actions + GitHub Pages)

`.github/workflows/daily-picks.yml` refreshes the board every 30 minutes
from 10 am to about 1:30 am US Eastern. It caches the day's trained models
between runs, commits only the history and the pick ledger, and publishes the
page to GitHub Pages. GitHub may start scheduled runs a few minutes late at
busy times.

One-time setup:

1. Merge this branch into your default branch. Scheduled workflows only run
   there.
2. **Settings → Secrets and variables → Actions.** Optionally add
   `ODDS_API_KEY` and `ANTHROPIC_API_KEY`.
3. **Settings → Pages → Source: GitHub Actions.** Pages on a private
   repository needs a paid GitHub plan. A page published from a public
   repository is public.
4. **Actions → Daily picks → Run workflow.** The first run downloads about
   two seasons per league and takes a while.

Actions minutes: a refresh takes about 3 minutes including the Pages deploy.
A 30-minute schedule (32 runs a day) is about 2,900 minutes a month. Public
repositories get Actions for free. On a private repository that exceeds the
free plan's 2,000 minutes a month. Either use a paid plan, or change the cron
lines to hourly (`0 14-23 * * *` and `0 0-5 * * *`, about 1,500 minutes a
month).

### API keys

| Service | What it's for | Where to get it | Cost |
|---|---|---|---|
| ESPN site API | games, results, live scores, injuries, headlines, posted lines | no key needed | free, unofficial |
| The Odds API (optional) | prices from many sportsbooks, consensus, best price | [the-odds-api.com](https://the-odds-api.com) → sign up → key by email | a free tier exists; a 30-minute refresh of four leagues needs a paid plan. Check their pricing page |
| Anthropic API (optional) | Claude reads messy news and injury reports | [console.anthropic.com](https://console.anthropic.com) → API Keys | pay per use |

Each Odds API request costs about (markets × regions) credits, and the board
makes one request per league per refresh while games are upcoming. The
default region is `us`. Set `ODDS_API_REGIONS=us,eu` to add Pinnacle to the
consensus, which costs twice the credits. Keep keys in GitHub secrets or
environment variables, never in the repository.

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
