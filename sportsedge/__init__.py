"""sportsedge — calibrated, uncertainty-aware game prediction for MLB, NHL, NBA and NFL."""
from .backtest import walk_forward
from .engine import EngineSettings, LeagueEngine
from .picks import PickPolicy

__all__ = ["LeagueEngine", "EngineSettings", "PickPolicy", "walk_forward"]
from .version import VERSION as __version__  # noqa: E402
