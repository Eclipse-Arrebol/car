"""Traffic-profile settings for regional scenario simulation."""

from __future__ import annotations

from dataclasses import dataclass


TRAFFIC_PROFILE_CHOICES = ("base", "old_city", "new_city", "suburb")


@dataclass(frozen=True)
class TrafficProfile:
    name: str
    ue_scale_multiplier: float
    daily_profile: str
    spawn_style: str


_PROFILES = {
    "base": TrafficProfile(
        name="base",
        ue_scale_multiplier=1.0,
        daily_profile="base",
        spawn_style="uniform",
    ),
    "old_city": TrafficProfile(
        name="old_city",
        ue_scale_multiplier=1.3,
        daily_profile="old_city",
        spawn_style="central",
    ),
    "new_city": TrafficProfile(
        name="new_city",
        ue_scale_multiplier=1.0,
        daily_profile="new_city",
        spawn_style="hotspot",
    ),
    "suburb": TrafficProfile(
        name="suburb",
        ue_scale_multiplier=0.7,
        daily_profile="suburb",
        spawn_style="peripheral",
    ),
}


def normalize_traffic_profile(name: str | None) -> str:
    profile = (name or "base").strip().lower()
    if profile not in _PROFILES:
        choices = ", ".join(TRAFFIC_PROFILE_CHOICES)
        raise ValueError(f"Unknown traffic_profile={name!r}; expected one of: {choices}")
    return profile


def get_traffic_profile(name: str | None) -> TrafficProfile:
    return _PROFILES[normalize_traffic_profile(name)]
