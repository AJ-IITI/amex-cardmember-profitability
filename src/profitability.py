"""Interpretable Premier Card profitability score from the documented 2026 File5 experiment.

This is a relative score, not observed issuer profit. The coefficients are
competition hypotheses, not actual American Express pricing or credit-loss rates.
"""
from __future__ import annotations

FEATURES = tuple(f"f{i}" for i in range(1, 24))
ACTIVE_FEATURES = (
    "f1","f2","f3","f5","f6","f7","f8","f9","f10",
    "f11","f12","f13","f14","f15","f17","f19","f21",
)

def to_number(value, *, default=0.0):
    if value is None or (isinstance(value, str) and not value.strip()):
        return float(default)
    return float(value)

def score_customer(row: dict, *, risk_median: float) -> float:
    """Compute an issuer-profitability proxy without using the customer identifier."""
    v = {f: to_number(row.get(f)) for f in ACTIVE_FEATURES}
    v["f11"] = to_number(row.get("f11"), default=risk_median)
    category = sum(v[f] for f in ("f6","f7","f8","f9","f10"))
    return (
        0.025*v["f6"] + 0.025*v["f7"] + 0.020*v["f8"]
        - 0.025*v["f9"] + 0.010*v["f10"] + 0.015*v["f5"]
        + 0.300*v["f1"]
        - 0.700*v["f11"]*(v["f1"] + category/12.0)
        - 0.025*v["f11"]*v["f17"]
        - v["f3"]*(0.200*v["f1"] + 200.0)
        - 45.0*v["f13"] - 0.75*v["f14"] - 12.0*v["f15"]
        - 200.0*v["f2"] - 0.0005*v["f21"]
        + 20.0*v["f19"] + 0.050*v["f12"]
    )
