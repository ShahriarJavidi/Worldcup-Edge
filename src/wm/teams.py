"""Static World Cup 2026 data: groups, hosts and team-name aliases."""

from __future__ import annotations

WORLD_CUP_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Qatar", "Switzerland", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d’Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Names API-Football uses where they differ from the official tournament names.
TEAM_NAME_ALIASES: dict[str, list[str]] = {
    "Czechia": ["Czech Republic"],
    "Côte d’Ivoire": ["Ivory Coast"],
    "Bosnia and Herzegovina": ["Bosnia", "Bosnia & Herzegovina"],
    "DR Congo": ["Congo DR"],
    "Türkiye": ["Turkey"],
    "Cabo Verde": ["Cape Verde Islands", "Cape Verde"],
    "United States": ["USA"],
}

HOST_NATIONS = {"canada", "mexico", "unitedstates", "usa"}


def normalize(name: str) -> str:
    """Case- and punctuation-insensitive key for comparing team names."""
    return "".join(ch.lower() for ch in name if ch.isalnum())


def is_host(team: str) -> bool:
    return normalize(team) in HOST_NATIONS


def group_of(team: str) -> str | None:
    key = normalize(team)
    for letter, members in WORLD_CUP_GROUPS.items():
        if key in {normalize(m) for m in members}:
            return letter
    return None


def teams_in_groups(letters: list[str]) -> list[str]:
    teams: list[str] = []
    for letter in letters:
        teams.extend(WORLD_CUP_GROUPS.get(letter.upper(), []))
    return teams


def all_teams() -> list[str]:
    return teams_in_groups(list(WORLD_CUP_GROUPS))


def host_context(team_a: str, team_b: str) -> str:
    """One sentence telling the model who (if anyone) has home advantage."""
    a_host, b_host = is_host(team_a), is_host(team_b)
    if a_host and b_host:
        return (
            "Both teams are 2026 host nations (USA, Mexico, Canada), so treat venue advantage as shared or uncertain."
        )
    if a_host or b_host:
        host = team_a if a_host else team_b
        return f"{host} is a 2026 host nation and may receive home advantage."
    return "Neutral venue. Do not infer home advantage from team order: the first team is not automatically the host."
