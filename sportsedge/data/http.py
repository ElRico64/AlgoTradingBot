"""A polite, resilient JSON client for the public sports feeds.

Why this exists: ESPN's CDN answers "403 Forbidden" to requests that look
like bots (a non-browser User-Agent, or bursts of parallel requests). This
client:

  * sends ordinary browser headers (and tries a plain profile if those fail);
  * tries ESPN's alternate public host when the main one refuses;
  * paces requests per host and retries 403/429/5xx with exponential backoff;
  * opens a circuit breaker after repeated refusals, so a blocked source fails
    fast with one clear message instead of hundreds of log lines, and the
    caller can switch to a backup source.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
PROFILES = [
    {"User-Agent": BROWSER_UA, "Accept": "application/json, text/plain, */*",
     "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.espn.com/", "Origin": "https://www.espn.com"},
    {"User-Agent": BROWSER_UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"},
    {},  # the HTTP library's own defaults
]
ALT_HOSTS = {"site.api.espn.com": ["site.web.api.espn.com"]}
RETRY_STATUS = {403, 408, 429, 500, 502, 503, 504}
MIN_INTERVAL = 0.45  # seconds between requests to the same host (~2 per second)
BREAKER_AFTER = 8  # consecutive refused requests before a host is paused
BREAKER_SECONDS = 600


class _HostState:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_time = 0.0
        self.consecutive_failures = 0
        self.open_until = 0.0
        self.profile = 0
        self.last_status: Optional[int] = None
        self.announced = False


_states: dict[str, _HostState] = {}
_states_lock = threading.Lock()
_local = threading.local()


def _state(host: str) -> _HostState:
    with _states_lock:
        return _states.setdefault(host, _HostState())


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


def _pace(st: _HostState) -> None:
    with st.lock:
        now = time.monotonic()
        wait = st.next_time - now
        st.next_time = max(now, st.next_time) + MIN_INTERVAL + random.uniform(0, 0.15)
    if wait > 0:
        time.sleep(wait)


def is_blocked(host: str) -> bool:
    return _state(host).open_until > time.monotonic()


def status(host: str) -> Optional[int]:
    return _state(host).last_status


def reset() -> None:
    with _states_lock:
        _states.clear()


def _candidates(url: str) -> list[str]:
    host = urlparse(url).netloc
    return [url] + [url.replace(host, alt, 1) for alt in ALT_HOSTS.get(host, [])]


def get_json(url: str, params: Optional[dict] = None, timeout: float = 15.0, retries: int = 3) -> Optional[dict]:
    """GET a JSON document. Returns None on 404 or when every attempt fails."""
    for attempt in range(retries):
        if all(is_blocked(urlparse(u).netloc) for u in _candidates(url)):
            return None  # every host for this request is paused: fail fast
        for u in _candidates(url):
            host = urlparse(u).netloc
            st = _state(host)
            if st.open_until > time.monotonic():
                continue
            order = [st.profile] + [i for i in range(len(PROFILES)) if i != st.profile]
            for pi in order:
                _pace(st)
                try:
                    r = _session().get(u, params=params, headers=PROFILES[pi], timeout=timeout)
                except requests.RequestException as e:
                    st.last_status = None
                    log.debug("network error %s: %s", u, e)
                    break  # try the next host / attempt
                st.last_status = r.status_code
                if r.status_code == 200:
                    st.consecutive_failures, st.profile = 0, pi
                    try:
                        return r.json()
                    except ValueError:
                        return None
                if r.status_code == 404:
                    return None
                if r.status_code not in RETRY_STATUS:
                    log.debug("HTTP %s for %s", r.status_code, u)
                    break
                st.consecutive_failures += 1
                if st.consecutive_failures >= BREAKER_AFTER:
                    st.open_until = time.monotonic() + BREAKER_SECONDS
                    if not st.announced:
                        st.announced = True
                        log.warning("%s refused %d requests in a row (HTTP %s); pausing it for %d minutes "
                                    "and using backup sources where available.", host, st.consecutive_failures,
                                    r.status_code, BREAKER_SECONDS // 60)
                    break
        if attempt < retries - 1:
            time.sleep(min(8.0, 1.5 * 2 ** attempt) + random.uniform(0, 0.5))
    return None
