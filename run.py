#!/usr/bin/env python3
"""Sportsedge launcher: see the board on your own computer, for free.

    python run.py            menu
    python run.py --demo     demo board (fictional teams, no sports data needed)
    python run.py --live     today's real games, odds and news
    python run.py --watch    live board that refreshes every 30 minutes

The page opens in your browser at http://localhost:8000. Press Ctrl+C in this
window to stop.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
DATA = os.path.join(HERE, "data")
PORT = 8000


def say(msg: str) -> None:
    print(f"[sportsedge] {msg}", flush=True)


def ensure_python() -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10 or newer is needed; this is {sys.version.split()[0]}. "
                 "Install it from https://www.python.org/downloads/ and run this again.")


def load_env_file() -> None:
    """Read optional keys from a local .env file (KEY=VALUE per line). Never committed."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ensure_packages() -> None:
    missing = []
    for mod in ("numpy", "scipy", "requests"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        say(f"Installing {', '.join(missing)} (one time only)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    sys.path.insert(0, HERE)


def serve(port: int) -> str:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=SITE)
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


def build_demo() -> None:
    from sportsedge.demo_site import build_demo as _build

    say("Simulating four leagues with fictional teams and running the full model.")
    say("This takes a few minutes the first time...")
    t = time.time()
    data = _build(SITE, progress=say)
    say(f"Demo board ready in {time.time() - t:.0f}s: {len(data['games'])} games, "
        f"{sum(1 for e in data['ledger'] if e['status'] == 'pending')} picks today.")


def build_live(leagues: list[str], first: bool) -> None:
    from sportsedge.daily import run_daily

    if first and not os.path.exists(os.path.join(DATA, "history")):
        say("First live run: downloading about two seasons of results from ESPN and training.")
        say("This takes roughly 5-15 minutes once; later refreshes take about a minute.")
    use_llm = False
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            use_llm = True
        except ImportError:
            say("ANTHROPIC_API_KEY is set but the 'anthropic' package is missing; "
                "run: pip install anthropic   (using the free news reader for now)")
    if first:
        say("Odds: " + ("The Odds API (budgeted to the free tier)" if os.environ.get("ODDS_API_KEY")
                        else "ESPN posted lines (no key needed)"))
    t = time.time()
    data = run_daily(leagues, data_dir=DATA, site_dir=SITE, odds_key=os.environ.get("ODDS_API_KEY"),
                     use_llm=use_llm, progress=say)
    for lg in data["leagues"]:
        extra = f" ({lg['message']})" if lg["message"] else ""
        say(f"  {lg['league']}: {lg['games']} games, {lg['picks']} picks{extra}")
    say(f"Board updated in {time.time() - t:.0f}s.")


def main() -> None:
    ensure_python()
    ap = argparse.ArgumentParser(description="See the Sportsedge board on your computer.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true", help="demo board with fictional teams")
    g.add_argument("--live", action="store_true", help="today's real games")
    g.add_argument("--watch", action="store_true", help="live board, refreshed every --every minutes")
    ap.add_argument("--every", type=int, default=30, help="minutes between refreshes with --watch")
    ap.add_argument("--league", nargs="+", default=["MLB", "NHL", "NBA", "NFL"])
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not (a.demo or a.live or a.watch):
        print("\nSportsedge\n  1) Demo board (fictional teams, works offline)\n"
              "  2) Live board for today\n  3) Live board, refreshed every 30 minutes\n")
        choice = input("Choose 1, 2 or 3 and press Enter: ").strip()
        a.demo, a.live, a.watch = choice == "1", choice == "2", choice == "3"
        if not (a.demo or a.live or a.watch):
            sys.exit("Nothing chosen.")
    ensure_packages()
    load_env_file()
    os.makedirs(SITE, exist_ok=True)
    if a.demo:
        build_demo()
    else:
        build_live(a.league, first=True)
    url = serve(PORT)
    say(f"Open {url} in your browser (Ctrl+C here to stop).")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        while True:
            if a.watch:
                time.sleep(a.every * 60)
                build_live(a.league, first=False)
                say("The open page picks up the new data by itself within 2 minutes.")
            else:
                time.sleep(3600)
    except KeyboardInterrupt:
        say("Stopped.")


if __name__ == "__main__":
    main()
