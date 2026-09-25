#!/usr/bin/env python3
"""Sportsedge launcher: see the board on your own computer, for free.

    python3 run.py                  menu
    python3 run.py --live           update today's board, then show it
    python3 run.py --watch          show the board and keep it updated every 30 minutes
    python3 run.py --open           just open your board (no update)
    python3 run.py --background on  keep updating every 30 minutes even when Terminal is closed (Mac)
    python3 run.py --background off stop the background updates
    python3 run.py --demo           demo board (fictional teams)
    python3 run.py --check          test which data sources this computer can reach

Your data (game history, trained models, the pick ledger that holds your
track record, and the board itself) lives in ~/Sportsedge, outside the code
folder, so it survives closing Terminal, restarting the Mac and downloading
new versions of the code.
"""
from __future__ import annotations

import argparse
import functools
import glob
import http.server
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.abspath(os.path.expanduser(os.environ.get("SPORTSEDGE_HOME", "~/Sportsedge")))
DATA = os.path.join(HOME, "data")
SITE = os.path.join(HOME, "site")
DEMO_SITE = os.path.join(HOME, "demo-site")
LOGS = os.path.join(HOME, "logs")
LOCK = os.path.join(HOME, ".update.lock")
PORT = 8000
LABEL = "com.sportsedge.refresh"
PLIST = os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")


def say(msg: str) -> None:
    print(f"[sportsedge] {msg}", flush=True)


def ensure_python() -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 or newer is needed; this is {sys.version.split()[0]}. "
                 "Install it from https://www.python.org/downloads/ and run this again.")


def ensure_packages(python: str = sys.executable) -> None:
    missing = []
    for mod in ("numpy", "scipy", "requests"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        say(f"Installing {', '.join(missing)} (one time only)...")
        subprocess.check_call([python, "-m", "pip", "install", "--quiet", *missing])
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def load_env_file() -> None:
    """Read optional keys from a .env file (KEY=VALUE per line). Never committed."""
    for path in (os.path.join(HERE, ".env"), os.path.join(HOME, ".env")):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --------------------------------------------------------------- your data
def migrate_old_data() -> None:
    """Earlier versions kept data inside the code folder. Copy the most recent
    such folder (this one or a sibling download) into ~/Sportsedge, once."""
    if os.path.isdir(os.path.join(DATA, "history")):
        return
    parent = os.path.dirname(HERE)
    candidates = [HERE] + [d for d in glob.glob(os.path.join(parent, "*")) if os.path.isfile(os.path.join(d, "run.py"))]
    best, best_time = None, -1.0
    for d in candidates:
        hist = os.path.join(d, "data", "history")
        if os.path.isdir(hist):
            t = max((os.path.getmtime(os.path.join(hist, f)) for f in os.listdir(hist)), default=0.0)
            if t > best_time:
                best, best_time = d, t
    if best is None:
        return
    say(f"Moving your saved games, models and track record to {HOME} (one time)...")
    shutil.copytree(os.path.join(best, "data"), DATA, dirs_exist_ok=True)
    old_site = os.path.join(best, "site")
    if os.path.isfile(os.path.join(old_site, "index.html")):
        shutil.copytree(old_site, SITE, dirs_exist_ok=True)


class UpdateLock:
    """Only one update at a time (the background job and this window share the data)."""

    def __init__(self, wait_seconds: float = 0):
        self.wait = wait_seconds
        self.held = False

    def __enter__(self):
        deadline = time.time() + self.wait
        while True:
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.held = True
                return self
            except FileExistsError:
                if self._stale():
                    os.remove(LOCK)
                    continue
                if time.time() >= deadline:
                    return self
                if not getattr(self, "_told", False):
                    self._told = True
                    say("The background job is updating right now; waiting for it to finish...")
                time.sleep(5)

    def _stale(self) -> bool:
        try:
            pid = int(open(LOCK).read().strip() or 0)
            os.kill(pid, 0)
            return time.time() - os.path.getmtime(LOCK) > 3 * 3600
        except (ValueError, ProcessLookupError, FileNotFoundError):
            return True
        except PermissionError:
            return False

    def __exit__(self, *exc):
        if self.held:
            try:
                os.remove(LOCK)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------- serving
def serve(directory: str, port: int = PORT) -> str:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    handler.log_message = lambda *a, **k: None

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    for p in range(port, port + 20):
        try:
            httpd = Server(("127.0.0.1", p), handler)
        except OSError:
            continue
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return f"http://localhost:{p}/"
    raise SystemExit("Could not find a free port for the local web page.")


def show(directory: str, browser: bool) -> str:
    url = serve(directory)
    say(f"Your board: {url}   (keep this window open while you look at it; Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    return url


def has_board(directory: str = SITE) -> bool:
    return os.path.isfile(os.path.join(directory, "index.html"))


# ---------------------------------------------------------------- building
def build_demo() -> None:
    from sportsedge.demo_site import build_demo as _build

    say("Simulating four leagues with fictional teams and running the full model (a few minutes)...")
    t = time.time()
    data = _build(DEMO_SITE, progress=say)
    say(f"Demo board ready in {time.time() - t:.0f}s: {len(data['games'])} games.")


def build_live(leagues: list[str], quiet: bool = False) -> bool:
    from sportsedge.daily import run_daily

    with UpdateLock(wait_seconds=0 if quiet else 1800) as lock:
        if not lock.held:
            say("Another update is already running (probably the background job); skipping this one.")
            return False
        first = not os.path.isdir(os.path.join(DATA, "history"))
        if first:
            say("First live run: downloading about two seasons of results and training.")
            say("This takes roughly 10-25 minutes once. After that, updates take a minute or two.")
        use_llm = False
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                use_llm = True
            except ImportError:
                say("ANTHROPIC_API_KEY is set but the 'anthropic' package is missing (pip install anthropic).")
        t = time.time()
        data = run_daily(leagues, data_dir=DATA, site_dir=SITE, odds_key=os.environ.get("ODDS_API_KEY"),
                         use_llm=use_llm, progress=None if quiet else say)
        for lg in data["leagues"]:
            extra = f" ({lg['message'][:100]})" if lg["message"] else ""
            say(f"  {lg['league']}: {lg['games']} games, {lg['picks']} picks{extra}")
        say(f"Board updated in {time.time() - t:.0f}s.")
        return True


# ------------------------------------------------------- background (macOS)
PROTECTED = [os.path.expanduser(p) for p in ("~/Desktop", "~/Documents", "~/Downloads")]


def _in_protected(path: str) -> bool:
    return any(os.path.abspath(path).startswith(p + os.sep) for p in PROTECTED)


def background(on: bool) -> None:
    if sys.platform != "darwin":
        say("Background updates are set up automatically on macOS only.")
        say(f"On Linux, add this line with `crontab -e`:  */30 * * * * {sys.executable} {os.path.join(HERE, 'run.py')} --refresh-once")
        say("On Windows, create a Task Scheduler task that runs:  py run.py --refresh-once  every 30 minutes.")
        return
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", PLIST], capture_output=True)
    if not on:
        if os.path.exists(PLIST):
            os.remove(PLIST)
        say("Background updates are OFF. Your saved data and track record are kept.")
        return
    # macOS blocks background jobs from reading Desktop/Documents/Downloads, so the job
    # runs a copy of the code (and, if needed, its own Python) from ~/Sportsedge.
    app = os.path.join(HOME, "app")
    say(f"Copying the program to {app} ...")
    if os.path.isdir(app):
        shutil.rmtree(app)
    os.makedirs(app)
    shutil.copy2(os.path.join(HERE, "run.py"), app)
    shutil.copytree(os.path.join(HERE, "sportsedge"), os.path.join(app, "sportsedge"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(HERE, ".env")):
        shutil.copy2(os.path.join(HERE, ".env"), os.path.join(HOME, ".env"))
    python = sys.executable
    if _in_protected(python) or _in_protected(os.path.realpath(python)):
        venv = os.path.join(HOME, "venv")
        say("Creating a private Python for the background job (one time)...")
        subprocess.check_call([sys.executable, "-m", "venv", venv])
        python = os.path.join(venv, "bin", "python")
        subprocess.check_call([python, "-m", "pip", "install", "--quiet", "numpy", "scipy", "requests"])
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(os.path.dirname(PLIST), exist_ok=True)
    log = os.path.join(LOGS, "background.log")
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{python}</string><string>{os.path.join(app, 'run.py')}</string><string>--refresh-once</string></array>
  <key>WorkingDirectory</key><string>{app}</string>
  <key>EnvironmentVariables</key><dict><key>SPORTSEDGE_HOME</key><string>{HOME}</string></dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""
    with open(PLIST, "w") as f:
        f.write(plist)
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", PLIST], capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(["launchctl", "load", "-w", PLIST], capture_output=True, text=True)
    if r.returncode != 0:
        say(f"Could not start the background job: {r.stderr.strip()}")
        return
    say("Background updates are ON: every 30 minutes while your Mac is awake, even with Terminal closed.")
    say(f"Open your board any time with:  python3 run.py --open   (or open {os.path.join(SITE, 'index.html')})")
    say(f"Activity log: {log}")
    say("After downloading a new version of the code, run  python3 run.py --background on  again.")


# ------------------------------------------------------------------- main
MENU = """
Sportsedge
  1) Live board: update now, then show it
  2) Live board: show it and keep it updated every 30 minutes (while this window is open)
  3) Just open my board (no update)
  4) Background updates ON  (keeps picks and track record updated even when Terminal is closed)
  5) Background updates OFF
  6) Demo board (fictional teams)
  7) Check data sources
"""


def main() -> None:
    ensure_python()
    ap = argparse.ArgumentParser(description="See the Sportsedge board on your computer.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true", help="update today's board, then show it")
    g.add_argument("--watch", action="store_true", help="show the board and keep it updated")
    g.add_argument("--open", action="store_true", help="open your board without updating")
    g.add_argument("--background", choices=["on", "off"], help="background updates (macOS)")
    g.add_argument("--refresh-once", action="store_true", help=argparse.SUPPRESS)  # used by the background job
    g.add_argument("--demo", action="store_true", help="demo board with fictional teams")
    g.add_argument("--check", action="store_true", help="test which data sources this computer can reach")
    ap.add_argument("--every", type=int, default=30, help="minutes between updates with --watch")
    ap.add_argument("--league", nargs="+", default=["MLB", "NHL", "NBA", "NFL"])
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not any((a.live, a.watch, a.open, a.background, a.refresh_once, a.demo, a.check)):
        print(MENU)
        choice = input("Choose 1-7 and press Enter: ").strip()
        mapping = {"1": "live", "2": "watch", "3": "open", "6": "demo", "7": "check"}
        if choice in mapping:
            setattr(a, mapping[choice], True)
        elif choice in ("4", "5"):
            a.background = "on" if choice == "4" else "off"
        else:
            sys.exit("Nothing chosen.")
    os.makedirs(HOME, exist_ok=True)
    ensure_packages()
    load_env_file()
    migrate_old_data()

    if a.check:
        from sportsedge.data.diagnose import check

        check()
        return
    if a.background:
        background(a.background == "on")
        return
    if a.refresh_once:
        say(time.strftime("background update %Y-%m-%d %H:%M"))
        build_live(a.league, quiet=True)
        return
    if a.demo:
        build_demo()
        show(DEMO_SITE, not a.no_browser)
        _wait_forever()
        return
    if a.open:
        if not has_board():
            sys.exit("No board yet. Run  python3 run.py --live  first.")
        show(SITE, not a.no_browser)
        _wait_forever()
        return

    # --live / --watch: show the last board right away (if any), then update it.
    try:
        shown = False
        if has_board():
            show(SITE, not a.no_browser)
            shown = True
            say("Showing your last board while it updates; the page refreshes by itself when the update is done.")
        build_live(a.league)
        if not shown and has_board():
            show(SITE, not a.no_browser)
        while True:
            if a.watch:
                time.sleep(a.every * 60)
                build_live(a.league)
            else:
                time.sleep(3600)
    except KeyboardInterrupt:
        say("Stopped. Your data and track record are saved in " + HOME)


def _wait_forever() -> None:
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        say("Stopped.")


if __name__ == "__main__":
    main()
