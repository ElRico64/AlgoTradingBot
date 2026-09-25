# Getting started: see Sportsedge run in about 10 minutes

Everything here is free. You need a computer (Windows, Mac or Linux) and an
internet connection.

---

## Part 1: Run it on your own computer

### Step 1: Install Python (one time, about 3 minutes)

1. Go to **https://www.python.org/downloads/** and click the yellow
   **Download Python 3.x** button. Any version 3.10 or newer works.
2. Run the installer.
   * **Windows:** on the first screen, tick **"Add python.exe to PATH"** at
     the bottom, then click **Install Now**. This checkbox matters.
   * **Mac:** click through the installer with the default options.
3. Check it worked. Open a terminal:
   * **Windows:** press the Windows key, type `cmd`, press Enter.
   * **Mac:** press Cmd+Space, type `Terminal`, press Enter.

   Then type the command below and press Enter. On Mac, use `python3`
   instead of `python`.

   ```
   python --version
   ```

   You should see `Python 3.12.x` or similar.

### Step 2: Download the code (1 minute)

1. Open your repository on GitHub: **https://github.com/ElRico64/AlgoTradingBot**
2. Use the branch dropdown (top left, usually says `main`) to pick
   **`claude/sports-betting-prediction-algo-7hrv8j`**. Once you've merged it,
   stay on `main`.
3. Click the green **Code** button, then **Download ZIP**.
4. Unzip it: on Windows right-click → **Extract All**, on Mac double-click.
   You get a folder such as `AlgoTradingBot-claude-...`.

If you use git instead:

```
git clone -b claude/sports-betting-prediction-algo-7hrv8j https://github.com/ElRico64/AlgoTradingBot.git
```

### Step 3: Start it (double-click)

Open the unzipped folder, then:

* **Windows:** double-click **`start-windows.bat`**.
  * If Windows shows "Windows protected your PC", click
    **More info → Run anyway**. It's the script from this repository.
* **Mac:** right-click **`start-mac.command`** → **Open** → **Open**.
  Right-click is needed the first time only.
* **Linux:** in a terminal inside the folder, run `./start-linux.sh`.

A window opens with a menu:

```
  1) Demo board (fictional teams, works offline)
  2) Live board for today
  3) Live board, refreshed every 30 minutes
```

**Type `1` and press Enter first.**

Before the live board, you can check that your network can reach the
sports data sources:

```
python3 run.py --check
```

* The first time, it installs three free Python packages (numpy, scipy,
  requests).
* It then simulates four leagues and runs the full model on them. This takes
  about 3–8 minutes depending on your computer, and progress messages keep
  you posted.
* Your browser then opens **http://localhost:8000**, which is the board.
* Keep the black window open while you look at the page. Press **Ctrl+C** in
  it to stop.

### Step 4: The live board

Start again and pick from the menu:

```
  1) Live board: update now, then show it
  2) Live board: show it and keep it updated every 30 minutes (while this window is open)
  3) Just open my board (no update)
  4) Background updates ON  (keeps picks and track record updated even when Terminal is closed)
  5) Background updates OFF
  6) Demo board (fictional teams)
  7) Check data sources
```

**The first live run is long, once.** It downloads about two seasons of
results and trains the models, which takes roughly 10–25 minutes. After
that:

* **Starting again the same day:** your last board opens right away. The
  update runs in the background, and the page refreshes itself when it's
  done.
* **A new day:** the saved models learn only the new results, which takes
  a minute or two.

**Your data is saved in `~/Sportsedge`**, the Sportsedge folder in your home
folder. That includes the game history, the trained models, the board, and
the pick ledger, which is your track record.

* It survives closing Terminal, restarting the Mac, and downloading new
  versions of the code.
* The first time a new version runs, it moves your data there from the old
  download folder automatically.
* Picks are graded as soon as their games finish, the next time an update
  runs. Picks from days you weren't running are graded on the next run.

**Background updates (option 4, Mac):**

* A small scheduled task updates the board every 30 minutes while your Mac
  is awake, even when Terminal is closed.
* To look at the board, run `python3 run.py --open`, or double-click
  `~/Sportsedge/site/index.html`. The file version reloads itself every 10
  minutes.
* After you download a new version of the code, choose option 4 again so
  the background job uses it.
* Option 5 turns the background updates off. Your data is kept.

**What you'll see:**

* Today's real games for MLB, NHL, NBA and NFL (the NFL shows the whole
  week), with ESPN's posted odds, live injury news and picks.
* No games for leagues in their off-season or preseason. In late September,
  for example, that means MLB and NFL only.

### Step 5 (optional): Add a free odds key

The board works without any key. For prices from many sportsbooks:

1. Sign up at **https://the-odds-api.com**. The free plan is enough, and the
   key arrives by email.
2. In the project folder, copy **`.env.example`** to a new file named
   **`.env`**, and paste your key after `ODDS_API_KEY=`.
3. Start the board again.

The board budgets its calls to stay inside the free 500 credits a month.
Leave `ANTHROPIC_API_KEY` empty: it costs money, and the built-in news reader
is free.

---

## Part 2: See the model prove itself

In a terminal inside the project folder (Mac: use `python3`):

```
python -m pip install -e .
python -m sportsedge demo --league NBA          # walk-forward backtest with calibration report
python -m sportsedge validate-patterns          # pattern engine on vs. off, planted patterns
python -m pip install pytest && python -m pytest -q   # 60+ automated checks
```

To backtest on real history:

```
python -m sportsedge fetch-history --league NBA --start 2023-10-24 --end 2025-04-13 --out nba.csv
python -m sportsedge backtest --league NBA --history nba.csv
```

---

## Part 3: Put it online, updated automatically, for $0

GitHub runs the model for you on a schedule and hosts the page.

### Step 1: Merge the branch

On GitHub, open a pull request from `claude/sports-betting-prediction-algo-7hrv8j`
into `main` and merge it. Scheduled jobs only run from `main`.

### Step 2: Allow the job to save its results

**Settings → Actions → General → Workflow permissions:** choose
**Read and write permissions** → **Save**.

### Step 3: Choose where the page lives (both options are free)

**Public repository (simplest).** Anyone can see the code and the page.

* **Settings → Pages → Build and deployment → Source: GitHub Actions**.
* Your board will be at `https://elrico64.github.io/AlgoTradingBot/`.
* Actions and Pages are free and unlimited for public repositories.

**Private repository.** The code stays private.

* GitHub Pages needs a paid plan for private repositories, so the job also
  pushes the finished page to a branch named **`site`**. Host that branch
  for free with Cloudflare Pages:
  1. Create a free account at **https://pages.cloudflare.com**.
  2. **Create a project → Connect to Git →** pick this repository.
  3. Production branch: **`site`**. Build command: *(leave empty)*. Build
     output directory: **`/`**.
  4. Save. The board is at `https://<your-project>.pages.dev` and updates
     each time the job runs.
* The job uses about 600–800 of the 2,000 free Actions minutes a month.

### Step 4: Optional keys

**Settings → Secrets and variables → Actions → New repository secret:**

| Name | Value |
|---|---|
| `ODDS_API_KEY` | your free key from the-odds-api.com |

Leave `ANTHROPIC_API_KEY` unset to stay at $0.

### Step 5: First run

**Actions → Daily picks → Run workflow.**

* The first run takes 15–30 minutes (history download and training).
* After that, the job runs 10 times a day around game times: 10:15am to
  11:15pm Eastern, plus a 1:45am grading run.
* Each run takes about 2 minutes.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized (Windows) | Reinstall Python and tick **Add python.exe to PATH**, or try `py` instead of `python`. |
| Mac says the file can't be opened | Right-click `start-mac.command` → **Open** → **Open**. |
| Browser shows nothing | Open **http://localhost:8000** yourself. If port 8000 is busy, the window prints the address it used. |
| "No games on the board" | That league is in its off-season or preseason today. |
| First live run is slow | It's downloading two seasons of history once. Later runs take about a minute. |
| `403 Forbidden` or "ESPN refused" | ESPN is blocking requests from your network. Run `python3 run.py --check` to see which sources work. Usual causes: a VPN, iCloud Private Relay (System Settings → your name → iCloud → Private Relay), a school or work network, or too many rapid requests (wait about 15 minutes). MLB and NHL switch to their official feeds automatically; NBA and NFL need ESPN. |
| A league shows "unavailable" | Its data source didn't answer. The next refresh retries automatically. |
| GitHub job fails on "git push" | Step 2 of Part 3: give workflows read-and-write permission. |

---

## What it costs

| Item | Cost |
|---|---|
| Running on your computer | $0 |
| ESPN data (games, scores, injuries, news, posted odds) | $0, no key needed |
| The Odds API | $0 on the free tier (budgeted automatically) |
| GitHub Actions + Pages (public repo) | $0 |
| GitHub Actions (private repo) + Cloudflare Pages | $0 |
| Claude news reader | Paid. Optional; leave the key empty |
