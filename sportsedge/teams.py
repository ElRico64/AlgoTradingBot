"""Team registry: ESPN-style abbreviations, names (for news parsing) and
approximate home-venue coordinates (for travel / time-zone features).

Unknown teams are tolerated everywhere: they simply get no travel features.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    abbr: str
    name: str  # full name, e.g. "Boston Celtics"
    nickname: str  # e.g. "Celtics"
    lat: float
    lon: float
    altitude: bool = False


def _t(abbr, name, lat, lon, altitude=False):
    return Team(abbr, name, name.split()[-1] if not name.endswith("Sox") and not name.endswith("Jays")
                else " ".join(name.split()[-2:]), lat, lon, altitude)


MLB = [
    _t("ARI", "Arizona Diamondbacks", 33.45, -112.07),
    _t("ATL", "Atlanta Braves", 33.89, -84.47),
    _t("BAL", "Baltimore Orioles", 39.28, -76.62),
    _t("BOS", "Boston Red Sox", 42.35, -71.10),
    _t("CHC", "Chicago Cubs", 41.95, -87.66),
    _t("CHW", "Chicago White Sox", 41.83, -87.63),
    _t("CIN", "Cincinnati Reds", 39.10, -84.51),
    _t("CLE", "Cleveland Guardians", 41.50, -81.69),
    _t("COL", "Colorado Rockies", 39.76, -104.99, True),
    _t("DET", "Detroit Tigers", 42.34, -83.05),
    _t("HOU", "Houston Astros", 29.76, -95.36),
    _t("KC", "Kansas City Royals", 39.05, -94.48),
    _t("LAA", "Los Angeles Angels", 33.80, -117.88),
    _t("LAD", "Los Angeles Dodgers", 34.07, -118.24),
    _t("MIA", "Miami Marlins", 25.78, -80.22),
    _t("MIL", "Milwaukee Brewers", 43.03, -87.97),
    _t("MIN", "Minnesota Twins", 44.98, -93.28),
    _t("NYM", "New York Mets", 40.76, -73.85),
    _t("NYY", "New York Yankees", 40.83, -73.93),
    _t("ATH", "Athletics Athletics", 38.58, -121.51),
    _t("PHI", "Philadelphia Phillies", 39.91, -75.17),
    _t("PIT", "Pittsburgh Pirates", 40.45, -80.01),
    _t("SD", "San Diego Padres", 32.71, -117.16),
    _t("SF", "San Francisco Giants", 37.78, -122.39),
    _t("SEA", "Seattle Mariners", 47.59, -122.33),
    _t("STL", "St. Louis Cardinals", 38.62, -90.19),
    _t("TB", "Tampa Bay Rays", 27.77, -82.65),
    _t("TEX", "Texas Rangers", 32.75, -97.08),
    _t("TOR", "Toronto Blue Jays", 43.64, -79.39),
    _t("WSH", "Washington Nationals", 38.87, -77.01),
]

NHL = [
    _t("ANA", "Anaheim Ducks", 33.81, -117.88),
    _t("BOS", "Boston Bruins", 42.37, -71.06),
    _t("BUF", "Buffalo Sabres", 42.88, -78.88),
    _t("CGY", "Calgary Flames", 51.04, -114.05),
    _t("CAR", "Carolina Hurricanes", 35.80, -78.72),
    _t("CHI", "Chicago Blackhawks", 41.88, -87.67),
    _t("COL", "Colorado Avalanche", 39.75, -105.01, True),
    _t("CBJ", "Columbus Blue Jackets", 39.97, -83.01),
    _t("DAL", "Dallas Stars", 32.79, -96.81),
    _t("DET", "Detroit Red Wings", 42.34, -83.06),
    _t("EDM", "Edmonton Oilers", 53.55, -113.50),
    _t("FLA", "Florida Panthers", 26.16, -80.33),
    _t("LA", "Los Angeles Kings", 34.04, -118.27),
    _t("MIN", "Minnesota Wild", 44.94, -93.10),
    _t("MTL", "Montreal Canadiens", 45.50, -73.57),
    _t("NSH", "Nashville Predators", 36.16, -86.78),
    _t("NJ", "New Jersey Devils", 40.73, -74.17),
    _t("NYI", "New York Islanders", 40.72, -73.73),
    _t("NYR", "New York Rangers", 40.75, -73.99),
    _t("OTT", "Ottawa Senators", 45.30, -75.93),
    _t("PHI", "Philadelphia Flyers", 39.90, -75.17),
    _t("PIT", "Pittsburgh Penguins", 40.44, -79.99),
    _t("SJ", "San Jose Sharks", 37.33, -121.90),
    _t("SEA", "Seattle Kraken", 47.62, -122.35),
    _t("STL", "St. Louis Blues", 38.63, -90.20),
    _t("TB", "Tampa Bay Lightning", 27.94, -82.45),
    _t("TOR", "Toronto Maple Leafs", 43.64, -79.38),
    _t("UTAH", "Utah Mammoth", 40.77, -111.90, True),
    _t("VAN", "Vancouver Canucks", 49.28, -123.11),
    _t("VGK", "Vegas Golden Knights", 36.10, -115.18),
    _t("WSH", "Washington Capitals", 38.90, -77.02),
    _t("WPG", "Winnipeg Jets", 49.89, -97.14),
]

NBA = [
    _t("ATL", "Atlanta Hawks", 33.76, -84.40),
    _t("BOS", "Boston Celtics", 42.37, -71.06),
    _t("BKN", "Brooklyn Nets", 40.68, -73.98),
    _t("CHA", "Charlotte Hornets", 35.23, -80.84),
    _t("CHI", "Chicago Bulls", 41.88, -87.67),
    _t("CLE", "Cleveland Cavaliers", 41.50, -81.69),
    _t("DAL", "Dallas Mavericks", 32.79, -96.81),
    _t("DEN", "Denver Nuggets", 39.75, -105.01, True),
    _t("DET", "Detroit Pistons", 42.34, -83.06),
    _t("GS", "Golden State Warriors", 37.77, -122.39),
    _t("HOU", "Houston Rockets", 29.75, -95.36),
    _t("IND", "Indiana Pacers", 39.76, -86.16),
    _t("LAC", "LA Clippers", 33.94, -118.34),
    _t("LAL", "Los Angeles Lakers", 34.04, -118.27),
    _t("MEM", "Memphis Grizzlies", 35.14, -90.05),
    _t("MIA", "Miami Heat", 25.78, -80.19),
    _t("MIL", "Milwaukee Bucks", 43.05, -87.92),
    _t("MIN", "Minnesota Timberwolves", 44.98, -93.28),
    _t("NO", "New Orleans Pelicans", 29.95, -90.08),
    _t("NY", "New York Knicks", 40.75, -73.99),
    _t("OKC", "Oklahoma City Thunder", 35.46, -97.52),
    _t("ORL", "Orlando Magic", 28.54, -81.38),
    _t("PHI", "Philadelphia 76ers", 39.90, -75.17),
    _t("PHX", "Phoenix Suns", 33.45, -112.07),
    _t("POR", "Portland Trail Blazers", 45.53, -122.67),
    _t("SAC", "Sacramento Kings", 38.58, -121.50),
    _t("SA", "San Antonio Spurs", 29.43, -98.44),
    _t("TOR", "Toronto Raptors", 43.64, -79.38),
    _t("UTAH", "Utah Jazz", 40.77, -111.90, True),
    _t("WSH", "Washington Wizards", 38.90, -77.02),
]

NFL = [
    _t("ARI", "Arizona Cardinals", 33.53, -112.26),
    _t("ATL", "Atlanta Falcons", 33.76, -84.40),
    _t("BAL", "Baltimore Ravens", 39.28, -76.62),
    _t("BUF", "Buffalo Bills", 42.77, -78.79),
    _t("CAR", "Carolina Panthers", 35.23, -80.85),
    _t("CHI", "Chicago Bears", 41.86, -87.62),
    _t("CIN", "Cincinnati Bengals", 39.10, -84.52),
    _t("CLE", "Cleveland Browns", 41.51, -81.70),
    _t("DAL", "Dallas Cowboys", 32.75, -97.09),
    _t("DEN", "Denver Broncos", 39.74, -105.02, True),
    _t("DET", "Detroit Lions", 42.34, -83.05),
    _t("GB", "Green Bay Packers", 44.50, -88.06),
    _t("HOU", "Houston Texans", 29.68, -95.41),
    _t("IND", "Indianapolis Colts", 39.76, -86.16),
    _t("JAX", "Jacksonville Jaguars", 30.32, -81.64),
    _t("KC", "Kansas City Chiefs", 39.05, -94.48),
    _t("LV", "Las Vegas Raiders", 36.09, -115.18),
    _t("LAC", "Los Angeles Chargers", 33.95, -118.34),
    _t("LAR", "Los Angeles Rams", 33.95, -118.34),
    _t("MIA", "Miami Dolphins", 25.96, -80.24),
    _t("MIN", "Minnesota Vikings", 44.97, -93.26),
    _t("NE", "New England Patriots", 42.09, -71.26),
    _t("NO", "New Orleans Saints", 29.95, -90.08),
    _t("NYG", "New York Giants", 40.81, -74.07),
    _t("NYJ", "New York Jets", 40.81, -74.07),
    _t("PHI", "Philadelphia Eagles", 39.90, -75.17),
    _t("PIT", "Pittsburgh Steelers", 40.45, -80.02),
    _t("SF", "San Francisco 49ers", 37.40, -121.97),
    _t("SEA", "Seattle Seahawks", 47.60, -122.33),
    _t("TB", "Tampa Bay Buccaneers", 27.98, -82.50),
    _t("TEN", "Tennessee Titans", 36.17, -86.77),
    _t("WSH", "Washington Commanders", 38.91, -76.86),
]

REGISTRY: dict[str, dict[str, Team]] = {
    "MLB": {t.abbr: t for t in MLB},
    "NHL": {t.abbr: t for t in NHL},
    "NBA": {t.abbr: t for t in NBA},
    "NFL": {t.abbr: t for t in NFL},
}


def get_team(league: str, abbr: str) -> Team | None:
    return REGISTRY.get(league.upper(), {}).get(abbr)


def resolve_team(league: str, text: str) -> str | None:
    """Map a free-text team reference (abbr, full name, nickname, city) to an abbr."""
    if not text:
        return None
    teams = REGISTRY.get(league.upper(), {})
    s = text.strip()
    if s.upper() in teams:
        return s.upper()
    low = s.lower()
    for t in teams.values():
        if low == t.name.lower() or low == t.nickname.lower():
            return t.abbr
    for t in teams.values():
        for alias in (t.name, t.nickname):
            if re.search(rf"\b{re.escape(alias.lower())}\b", low):
                return t.abbr
    return None


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
