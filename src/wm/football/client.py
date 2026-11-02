"""Minimal API-Football v3 client (https://www.api-football.com/documentation-v3)."""

from __future__ import annotations

from typing import Any

import requests

from wm.teams import TEAM_NAME_ALIASES, normalize

BASE_URL = "https://v3.football.api-sports.io"

# Leagues whose standings double as an international ranking (Friendlies, World Cup).
RANKING_LEAGUES = (30, 1)


class APIFootballError(RuntimeError):
    pass


class APIFootballClient:
    def __init__(self, api_key: str, *, timeout: float = 40, session: requests.Session | None = None) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["x-apisports-key"] = api_key

    def get(self, path: str, **params: Any) -> list[dict[str, Any]]:
        response = self.session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("errors"):
            raise APIFootballError(f"API-Football error: {data['errors']}")
        return data.get("response", [])

    def find_team(self, name: str) -> dict[str, Any]:
        """Best match for ``name``: exact, then national team, then substring."""
        for query in [name, *TEAM_NAME_ALIASES.get(name, [])]:
            results = self.get("/teams", search=query)
            if not results:
                continue
            target = normalize(query)
            team_name = lambda item: normalize(item.get("team", {}).get("name", ""))  # noqa: E731
            for candidates in (
                [r for r in results if team_name(r) == target],
                [r for r in results if r.get("team", {}).get("national") is True],
                [r for r in results if target in team_name(r)],
            ):
                if candidates:
                    return candidates[0]
        raise APIFootballError(f"Team not found: {name}")

    def recent_fixtures(self, team_id: int, season: int, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent fixtures, walking back up to four seasons to fill ``limit``."""
        collected: dict[Any, dict[str, Any]] = {}
        for year in range(season, season - 4, -1):
            for item in self.get("/fixtures", team=team_id, last=20, season=year):
                collected.setdefault(item.get("fixture", {}).get("id"), item)
            if len(collected) >= limit:
                break
        fixtures = sorted(collected.values(), key=lambda i: i.get("fixture", {}).get("date", ""), reverse=True)
        return fixtures[:limit]

    def injuries(self, team_id: int, season: int) -> list[dict[str, Any]]:
        return self.get("/injuries", team=team_id, season=season)

    def players(self, team_id: int, season: int) -> list[dict[str, Any]]:
        return self.get("/players", team=team_id, season=season)

    def ranking_entry(self, team_id: int, season: int) -> tuple[str, dict[str, Any]] | None:
        """(league name, standings row) for the team, if any ranking league lists it."""
        for league_id in RANKING_LEAGUES:
            try:
                payload = self.get("/standings", league=league_id, season=season)
            except (requests.RequestException, APIFootballError):
                continue
            for entry in payload:
                league = entry.get("league", {})
                for group in league.get("standings") or []:
                    for row in group:
                        if row.get("team", {}).get("id") == team_id:
                            return league.get("name", "Unknown league"), row
        return None
