"""Render the dashboard: one self-contained HTML page with the day's data embedded."""
from __future__ import annotations

import json
import os
from importlib import resources

PLACEHOLDER = "__SPORTSEDGE_DATA__"
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        # iPhone "Add to Home Screen": app icon, full-screen launch, title under the icon
        '<link rel="apple-touch-icon" href="apple-touch-icon.png">\n'
        '<link rel="icon" type="image/png" href="apple-touch-icon.png">\n'
        '<link rel="manifest" href="manifest.webmanifest">\n'
        '<meta name="apple-mobile-web-app-capable" content="yes">\n'
        '<meta name="mobile-web-app-capable" content="yes">\n'
        '<meta name="apple-mobile-web-app-status-bar-style" content="default">\n'
        '<meta name="apple-mobile-web-app-title" content="Sportsedge">\n'
        '<meta name="theme-color" content="#EDF0F4" media="(prefers-color-scheme: light)">\n'
        '<meta name="theme-color" content="#090D14" media="(prefers-color-scheme: dark)">\n'
        '</head>\n<body>\n')
MANIFEST = {"name": "Sportsedge", "short_name": "Sportsedge", "start_url": "./", "scope": "./",
            "display": "standalone", "background_color": "#EDF0F4", "theme_color": "#0E1420",
            "icons": [{"src": "apple-touch-icon.png", "sizes": "180x180", "type": "image/png"},
                      {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"}]}


def render_fragment(data: dict) -> str:
    """The page body (title, styles, markup, script) with data embedded."""
    template = resources.files("sportsedge").joinpath("web/dashboard.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), default=str).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)


def write_site(data: dict, site_dir: str) -> str:
    os.makedirs(site_dir, exist_ok=True)
    path = os.path.join(site_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEAD + render_fragment(data) + "\n</body>\n</html>\n")
    with open(os.path.join(site_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)
    with open(os.path.join(site_dir, ".nojekyll"), "w") as f:
        f.write("")
    with open(os.path.join(site_dir, "manifest.webmanifest"), "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, indent=1)
    for name in ("apple-touch-icon.png", "icon-512.png"):
        with open(os.path.join(site_dir, name), "wb") as f:
            f.write(resources.files("sportsedge").joinpath("web/" + name).read_bytes())
    return path
