def compute_score_tier(rank: int) -> str:
    if rank <= 5000:
        return "high"
    elif rank <= 30000:
        return "medium"
    else:
        return "low"


def compute_risk_window(rank: int, score_tier: str) -> dict:
    tiers = {
        "high":   {"冲": (0.85, 0.97), "稳": (0.97, 1.15), "保": (1.15, 1.50)},
        "medium": {"冲": (0.80, 0.95), "稳": (0.95, 1.20), "保": (1.20, 1.70)},
        "low":    {"冲": (0.75, 0.93), "稳": (0.93, 1.25), "保": (1.25, 2.00)},
    }
    t = tiers[score_tier]
    return {k: [int(rank * lo), int(rank * hi)] for k, (lo, hi) in t.items()}
