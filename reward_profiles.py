"""Per-client (per-city) reward profiles for the heterogeneous federated study.

Activated ONLY when env var HETERO_REWARD=1, so every prior experiment / model
that ran without it keeps the original uniform reward (0.4 trip, 0.4 queue,
0.2 fee). When active, each city optimizes a strongly different trade-off, so the
optimal station-selection policy genuinely differs per client:

  old_city : queue-averse  (congested city)
  new_city : fee-averse    (price-sensitive city)
  suburb   : trip-averse   (rural, long distances)

The reward is looked up by env.grid_variant inside HindsightTrainer (training
signal) and inside eval run_mode (evaluation), so no CLI threading is needed.
"""

import os

# (w_trip, w_queue, w_fee), summing to 1.0
DEFAULT_W = (0.4, 0.4, 0.2)

_HETERO = {
    "old_city": (0.15, 0.70, 0.15),  # queue-averse
    "new_city": (0.15, 0.15, 0.70),  # fee-averse
    "suburb":   (0.70, 0.15, 0.15),  # trip-averse
}


def hetero_enabled():
    return os.environ.get("HETERO_REWARD", "0") == "1"


def weights_for(grid_variant):
    """Reward weights for a city. Falls back to DEFAULT_W unless HETERO_REWARD=1."""
    if hetero_enabled():
        return _HETERO.get(grid_variant, DEFAULT_W)
    return DEFAULT_W
