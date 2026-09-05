"""Profile loader — resolves use-case profiles with default merging.

Usage:
    profile = load_profile("customer_support")
    # Returns default values overridden by customer_support-specific settings
"""

from pathlib import Path
from typing import Any

import yaml

_PROFILES_PATH = Path(__file__).parent / "profiles.yaml"
_cache: dict[str, dict] | None = None


def _load_all_profiles() -> dict[str, dict]:
    """Load and cache all profiles from profiles.yaml."""
    global _cache
    if _cache is not None:
        return _cache

    if _PROFILES_PATH.exists():
        with open(_PROFILES_PATH, "r") as f:
            _cache = yaml.safe_load(f) or {}
    else:
        _cache = {}
    return _cache


def load_profile(use_case: str = "default") -> dict[str, Any]:
    """Load a use-case profile, merged with defaults.

    Args:
        use_case: The profile name (e.g., "customer_support").
                  Falls back to "default" if not found.

    Returns:
        A dict of profile settings with defaults filled in.
    """
    profiles = _load_all_profiles()
    defaults = profiles.get("default", {})

    if use_case == "default" or use_case not in profiles:
        return dict(defaults)

    # Merge: default values overridden by use-case-specific values
    merged = dict(defaults)
    merged.update(profiles[use_case])
    return merged


def list_profiles() -> list[str]:
    """Return the names of all available profiles."""
    return list(_load_all_profiles().keys())


def clear_cache() -> None:
    """Clear the profile cache (useful for testing)."""
    global _cache
    _cache = None
