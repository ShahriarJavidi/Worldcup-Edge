"""Render API-Football data as a plain-text team dossier for the knowledge base."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from wm.football.client import APIFootballClient
from wm.teams import group_of


def build_team_context(client: APIFootballClient, team_name: str, season: int | None = None) -> tuple[str, str]:
    """Return ``(canonical team name, dossier text)``."""
    season = season or datetime.now().year
    team = client.find_team(team_name)
    team_id = team["team"]["id"]
    label = team["team"].get("name") or team_name

    fixtures = client.recent_fixtures(team_id, season)
    group = group_of(team_name)
    sections = [
        f"World Cup context for {label}" + (f" (Group {group})" if group else ""),
        "Team profile:\n" + profile(team),
        recent_form(fixtures, label),
        ranking(client.ranking_entry(team_id, season)),
        "Recent fixtures:\n" + fixture_list(fixtures),
        "Key players:\n" + key_players(client.players(team_id, season)),
        "Recent injury / absence notes:\n" + injury_list(client.injuries(team_id, season)),
    ]
    return label, "\n\n".join(sections)


def profile(team: dict[str, Any]) -> str:
    info, venue = team.get("team", {}), team.get("venue", {})
    return "\n".join(
        [
            f"Founded: {info.get('founded') or 'unknown'}",
            f"Country: {info.get('country') or 'unknown'}",
            f"National team: {'yes' if info.get('national') else 'no'}",
            f"Venue: {venue.get('name') or 'unknown'} ({venue.get('city') or 'unknown'})",
            f"Capacity: {venue.get('capacity') or 'unknown'}",
        ]
    )


def result_for(fixture: dict[str, Any], team_name: str) -> str:
    """W / D / L from ``team_name``'s perspective, or P if not yet played."""
    teams, goals = fixture.get("teams", {}), fixture.get("goals", {})
    home, away = goals.get("home"), goals.get("away")
    if home is None or away is None:
        return "P"
    if home == away:
        return "D"
    is_home = teams.get("home", {}).get("name") == team_name
    return "W" if (home > away) == is_home else "L"


def recent_form(fixtures: list[dict[str, Any]], team_name: str) -> str:
    results = [result_for(f, team_name) for f in fixtures]
    if not results:
        return "Recent form: no results available"
    return f"Recent form (last {len(results)} matches, newest first): {' '.join(results)}"


def ranking(entry: tuple[str, dict[str, Any]] | None) -> str:
    if entry is None:
        return "International ranking: not available from API-Football standings."
    league, row = entry
    return (
        f"International ranking: #{row.get('rank', 'n/a')} in {league} "
        f"(Group {row.get('group') or 'N/A'}), {row.get('points', 'n/a')} pts, "
        f"{row.get('all', {}).get('played', 'n/a')} played, GD {row.get('goalsDiff', 'n/a')}, "
        f"form {row.get('form') or 'n/a'}"
    )


def fixture_list(fixtures: list[dict[str, Any]]) -> str:
    lines = []
    for item in fixtures:
        teams, goals = item.get("teams", {}), item.get("goals", {})
        date = item.get("fixture", {}).get("date", "")[:10]
        home = teams.get("home", {}).get("name", "Home")
        away = teams.get("away", {}).get("name", "Away")
        lines.append(f"{date}: {home} {goals.get('home', '?')} - {goals.get('away', '?')} {away}")
    return "\n".join(lines) or "No recent fixtures found."


def key_players(players: list[dict[str, Any]], limit: int = 8) -> str:
    lines = []
    for item in players[:limit]:
        player = item.get("player", {})
        stats = (item.get("statistics") or [{}])[0]
        games, goals, cards = stats.get("games", {}), stats.get("goals", {}), stats.get("cards", {})
        lines.append(
            f"- {player.get('name', 'Unknown player')}: {games.get('position') or 'Unknown'}, "
            f"age {player.get('age', 'n/a')}, {games.get('appearences') or 0} apps, "
            f"{goals.get('total') or 0} goals, {cards.get('yellow') or 0} yellow cards"
        )
    return "\n".join(lines) or "No player data available."


def injury_list(injuries: list[dict[str, Any]], limit: int = 10) -> str:
    lines = [
        f"{i.get('team', {}).get('name', 'Unknown team')}: "
        f"{i.get('player', {}).get('name', 'Unknown player')} - {i.get('reason') or 'unknown'}"
        for i in injuries[:limit]
    ]
    return "\n".join(lines) or "No recent injury reports found."
