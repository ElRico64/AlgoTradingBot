"""Render the dashboard: one self-contained HTML page with the day's data embedded."""
from __future__ import annotations

import json
import os
from importlib import resources

PLACEHOLDER = "__SPORTSEDGE_DATA__"
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '</head>\n<body>\n')


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
    return path
