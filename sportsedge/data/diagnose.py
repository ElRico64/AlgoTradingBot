"""Connectivity check: which data sources can this computer reach?"""
from __future__ import annotations

import os
from datetime import date

import requests

from .http import PROFILES

PROFILE_NAMES = ["browser + espn.com referer", "browser", "plain"]


def _try(url: str, params=None) -> list[tuple[str, str]]:
    out = []
    for name, headers in zip(PROFILE_NAMES, PROFILES):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=12)
            out.append((name, f"HTTP {r.status_code}"))
        except requests.RequestException as e:
            out.append((name, type(e).__name__))
    return out


def check(print_fn=print) -> bool:
    today = date.today()
    tests = [
        ("ESPN scores (main)", f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today:%Y%m%d}", None),
        ("ESPN scores (backup host)", f"https://site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today:%Y%m%d}", None),
        ("ESPN injuries", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries", None),
        ("MLB official (statsapi.mlb.com)", "https://statsapi.mlb.com/api/v1/schedule", {"sportId": 1, "date": today.isoformat()}),
        ("NHL official (api-web.nhle.com)", f"https://api-web.nhle.com/v1/score/{today.isoformat()}", None),
    ]
    key = os.environ.get("ODDS_API_KEY")
    if key:
        tests.append(("The Odds API (free call)", "https://api.the-odds-api.com/v4/sports", {"apiKey": key}))
    ok_any = {}
    print_fn("Checking data sources from this computer...\n")
    for label, url, params in tests:
        res = _try(url, params)
        good = [n for n, s in res if s == "HTTP 200"]
        ok_any[label] = bool(good)
        verdict = f"OK (works with: {', '.join(good)})" if good else "FAILED"
        print_fn(f"  {label:34s} {verdict}")
        if not good:
            print_fn("      " + "; ".join(f"{n}: {s}" for n, s in res))
    espn_ok = ok_any["ESPN scores (main)"] or ok_any["ESPN scores (backup host)"]
    print_fn("")
    official = [lg for lg, label in (("MLB", "MLB official (statsapi.mlb.com)"), ("NHL", "NHL official (api-web.nhle.com)"))
                if ok_any[label]]
    if espn_ok:
        print_fn("ESPN is reachable: all four leagues, odds and injury news will work.")
    else:
        print_fn("ESPN is refusing this network, so NBA and NFL can't load and there are no posted odds.")
        if official:
            print_fn(f"{' and '.join(official)} will use the official league feed instead (predictions and 70%+ picks work).")
        print_fn("Common causes: a VPN, iCloud Private Relay (System Settings > Apple ID > iCloud > Private Relay),")
        print_fn("a school/work network, or ESPN rate-limiting after many rapid requests (wait ~15 minutes).")
        print_fn("Fix the cause, then run  python3 run.py --check  again.")
    if not espn_ok and not official:
        print_fn("\nNo sports data source is reachable from here: check your internet connection.")
    return espn_ok
