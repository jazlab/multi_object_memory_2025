"""Constants."""

import os
from pathlib import Path

################################################################################
####  Paths
################################################################################

_CURRENT_DIR = Path(os.path.realpath(__file__)).parent
SPIKES_PER_TRIAL_DIR = (
    _CURRENT_DIR / "../cache/phys_processing/spikes_to_trials/spikes_per_trial"
)
BEHAVIOR_DIR_TRIANGLE = (
    _CURRENT_DIR / "../cache/behavior_processing/triangle.csv"
)
BEHAVIOR_DIR_RING = _CURRENT_DIR / "../cache/behavior_processing/ring.csv"
SELECTIVITY_DIR = _CURRENT_DIR / "../cache/phys_processing/selectivity"

################################################################################
####  Data variables
################################################################################

SPIKE_COUNT_TIMEFRAME_PER_PHASE = {
    "stimulus": (200, 1200),
    "delay": (1200, 2200),
}
SPIKE_COUNT_BIN_SIZE_MS = 10

################################################################################
####  Task variables
################################################################################

IDENTITIES = ["a", "b", "c"]
