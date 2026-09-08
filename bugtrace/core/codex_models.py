"""Codex model auto-discovery — keep the active model set in sync with the
ChatGPT backend.

The official Codex CLI ships with hardcoded model names that drift as OpenAI
adds/renames models, and the preset in ``bugtrace/data/providers/codex.json``
has the same problem. When the ``codex`` provider is active, BugTraceAI asks
the ChatGPT backend for the models it actually serves and remaps the preset's
slots:

- ``heavy`` slots (analysis/vision/lonewolf, where reasoning quality matters)
  get the primary (non-mini/nano) model
- ``light`` slots (mutation/skeptical/reporting/researcher, the high-volume
  ones) get the smallest model served (mini/nano), if any
- ``PRIMARY_MODELS`` (the model-shifting pool) becomes a mix of both

Discovery is a pure refresh: the static preset remains the fallback whenever
the backend is unreachable, and any slot the user has pinned manually keeps
winning only if the backend still serves that exact slug (a slug the backend
dropped is replaced by the newest available equivalent).
"""

from typing import Dict, List, Optional

from bugtrace.utils.logger import get_logger

logger = get_logger("core.codex_models")

# Slug markers that identify the small/fast tier of a model family. Anything
# without one of these is treated as the capable/primary tier.
LIGHT_MARKERS = ("-mini", "-nano", "-small", "-lite", "-flash", "-haiku")

# Slots that want the most capable model available.
HEAVY_SLOT_FIELDS = [
    "DEFAULT_MODEL",
    "CODE_MODEL",
    "ANALYSIS_MODEL",
    "ANALYSIS_PENTESTER_MODEL",
    "ANALYSIS_BUG_BOUNTY_MODEL",
    "ANALYSIS_AUDITOR_MODEL",
    "ANALYSIS_RED_TEAM_MODEL",
    "VISION_MODEL",
    "VALIDATION_VISION_MODEL",
    "LONEWOLF_MODEL",
]

# Slots that are called at high volume and can use the small/fast tier.
LIGHT_SLOT_FIELDS = [
    "ANALYSIS_RESEARCHER_MODEL",
    "WAF_DETECTION_MODELS",
    "MUTATION_MODEL",
    "SKEPTICAL_MODEL",
    "REPORTING_MODEL",
]

DEFAULT_MAX_PRIMARY = 3


def is_light_slug(slug: str) -> bool:
    """True for the small/fast tier of a model family (mini, nano, lite...)."""
    low = slug.lower()
    return any(marker in low for marker in LIGHT_MARKERS)


def plan_codex_assignments(
    slugs: List[str],
    preferred_heavy: str = "",
    preferred_light: str = "",
    max_primary: int = DEFAULT_MAX_PRIMARY,
) -> Dict[str, str]:
    """Map discovered model slugs onto BugTraceAI model-slot fields.

    Returns {setting_field: value} to apply, or {} when there is nothing to
    work with. Slot assignments respect the user's pinned slug when the
    backend still serves it; otherwise they fall forward to the newest
    available model of the right tier.
    """
    clean = [s.strip() for s in (slugs or []) if s and isinstance(s, str) and s.strip()]
    if not clean:
        return {}

    def pick(preferred: str, want_light: bool) -> str:
        # A user pin wins whenever the backend still serves it — even if the
        # pin is an unusual tier for the slot.
        if preferred and preferred in clean:
            return preferred
        # Otherwise fall forward within the slot's tier (first in server order).
        tiered = [s for s in clean if is_light_slug(s) == want_light] or clean
        return tiered[0]

    best_heavy = pick(preferred_heavy, want_light=False)
    best_light = pick(preferred_light, want_light=True)

    assignments: Dict[str, str] = {}
    for field in HEAVY_SLOT_FIELDS:
        assignments[field] = best_heavy
    for field in LIGHT_SLOT_FIELDS:
        assignments[field] = best_light

    # Model-shifting pool: best heavy first, then the light tier, then any
    # remaining slugs (deduped, capped) so resilience never costs latency.
    pool: List[str] = []
    for s in [best_heavy, best_light] + clean:
        if s not in pool:
            pool.append(s)
        if len(pool) >= max(1, max_primary):
            break
    assignments["PRIMARY_MODELS"] = ",".join(pool)
    return assignments


def apply_assignments(assignments: Dict[str, str], target) -> bool:
    """Apply slot assignments onto a settings-like object via object.__setattr__.

    Also refreshes the runtime ``_provider_config`` model map so a later
    provider hot-switch doesn't resurrect stale model names. Returns True when
    at least one field was applied.
    """
    if not assignments:
        return False
    applied = 0
    for field, value in assignments.items():
        if not hasattr(target, field):
            continue
        object.__setattr__(target, field, value)
        applied += 1
    # Keep the runtime preset copy consistent (this is the dict reconfigure
    # re-applies on a provider hot-switch).
    provider_cfg = getattr(target, "_provider_config", None)
    if isinstance(provider_cfg, dict):
        models = provider_cfg.setdefault("models", {})
        if isinstance(models, dict):
            models.update(assignments)
    if applied:
        logger.info(f"[Codex] Applied {applied} auto-discovered model assignment(s)")
    return applied > 0
