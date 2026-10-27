"""D4 prohibited items (spec 4.4): OWLv2 item boxes -> Gate 2 signals.

The boxes come from Gate 1's detector pass (app.gates.detection), so this adds no model
time. A box named after a harmless look-alike (toy gun, kitchen knife) maps to None and
never counts.
"""
from app.core.config import settings


def item_signals(found: dict[str, float]) -> set[str]:
    """Signals for the item queries whose best box is sure enough."""
    cfg = settings.gate2.items
    return {
        cfg.queries[query]
        for query, score in found.items()
        if score >= cfg.min_score and cfg.queries.get(query)
    }
