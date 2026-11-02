"""API-Football (v3) client and team-context builders."""

from wm.football.client import APIFootballClient, APIFootballError
from wm.football.context import build_team_context

__all__ = ["APIFootballClient", "APIFootballError", "build_team_context"]
