"""Compute and cache triangle dataset mean firing rate data.

This will compute the mean firing rate for each unit during stimulus and delay
phases of the triangle task, conditionalized on object position,
save them to `../../../cache/figures/supp_fig_stimulus/orthogonality`.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_BEHAVIOR_CACHE_PATH_TRIANGLE = Path(
    "../../../cache/behavior_processing/triangle.csv"
)
_FIRING_RATES_DATA_DIR = Path(
    "../../../cache/phys_processing/spikes_to_trials/spikes_per_trial"
)
_WRITE_DIR = Path("../../../cache/figures/supp_fig_stimulus/orthogonality")
_PROBE_TO_BRAIN_AREA = {
    "s0": "DMFC",
    "vprobe0": "FEF",
    "vprobe1": "FEF",
}
LEAD_IN_SECONDS = 0.2
BIN_SIZE_SECONDS = 0.01
WRITE_BIN_SIZE_SECONDS = 0.05
BINS_PER_WRITE_BIN = int(WRITE_BIN_SIZE_SECONDS / BIN_SIZE_SECONDS)


def _get_condition(row: pd.Series) -> str:
    """Get condition string for a row in the behavior dataframe.

    Args:
        row: A row from the behavior dataframe.

    Returns:
        condition: A string representing the condition for the row.
    """
    if row.on_triangle and row.num_objects == 1:
        condition = f"{int(row.object_0_location)}_{row.object_0_id}"
    else:
        condition = np.nan
    return condition


def _average_with_attrition(frs, max_seconds=2.0):
    """Average firing rates with attrition for missing data.

    Args:
        frs: A list of firing rate lists.
        max_seconds: Maximum number of seconds to consider.

    Returns:
        mean_fr: A list of mean firing rates.
    """
    max_bins = int(max_seconds / WRITE_BIN_SIZE_SECONDS)
    binned_frs = np.array(
        [fr[:max_bins] + [np.nan] * (max_bins - len(fr)) for fr in frs]
    )
    mean_fr = np.nanmean(binned_frs, axis=0).tolist()
    return mean_fr


def _compute_firing_rates(
    unit_trials,
    unit_spike_counts,
    df_behavior,
    condition_on_identity=False,
):
    """Compute mean firing rates for a unit conditionalized on object position.

    Args:
        unit_trials: A list of trial indices for the unit.
        unit_spike_counts: A list of spike counts for the unit.
        df_behavior: The behavior dataframe for the session.
        condition_on_identity: Whether to condition on object identity as well
            as position.

    Returns:
        firing_rates: A list of mean firing rates per condition.
    """
    # Filter unit data
    keep_trials = df_behavior.trial_num.tolist()
    keep_indices = [
        i
        for i, trial_idx in enumerate(unit_trials)
        if trial_idx in keep_trials
    ]
    unit_trials = [unit_trials[i] for i in keep_indices]
    unit_spike_counts = [unit_spike_counts[i] for i in keep_indices]

    # Get condition per trial
    trial_nums = df_behavior["trial_num"].values
    if condition_on_identity:
        conditions = df_behavior["condition"].values
    else:
        conditions = df_behavior["condition"].apply(lambda x: str(x)[0]).values
    trial_to_condition = {
        trial_num: condition
        for trial_num, condition in zip(trial_nums, conditions)
    }

    # Get stimulus and delay duration per trial
    trial_to_duration = {
        row.trial_num: row.time_cue_onset - row.time_stimulus_onset
        for _, row in df_behavior.iterrows()
    }

    fr_per_condition = {condition: [] for condition in np.unique(conditions)}
    for trial, spike_counts in zip(unit_trials, unit_spike_counts):
        condition = trial_to_condition[trial]
        trial_duration = trial_to_duration[trial]

        # Compute start and stop bins
        start_bin = int(LEAD_IN_SECONDS / BIN_SIZE_SECONDS)
        n_bins = int(trial_duration / BIN_SIZE_SECONDS)
        n_bins = (n_bins // BINS_PER_WRITE_BIN) * BINS_PER_WRITE_BIN

        # Compute spike counts in window,
        spike_counts = spike_counts[start_bin : start_bin + n_bins]

        # Bin spike counts by BINS_PER_WRITE_BIN
        binned_spike_counts = []
        for i in range(0, len(spike_counts), BINS_PER_WRITE_BIN):
            n_spikes = np.sum(spike_counts[i : i + BINS_PER_WRITE_BIN])
            binned_spike_counts.append(float(n_spikes))

        fr_per_condition[condition].append(binned_spike_counts)

    # Compute mean firing rates per condition
    mean_frs = {
        condition: _average_with_attrition(frs)
        for condition, frs in fr_per_condition.items()
    }

    return mean_frs


def main():
    """Generate and save delay phase noise correlations."""
    # Load behavior data
    triangle_behavior = pd.read_csv(_BEHAVIOR_CACHE_PATH_TRIANGLE)

    # Set up data structures to hold firing rates
    firing_rates = []
    unit_df = {
        "subject": [],
        "session": [],
        "unit": [],
        "brain_area": [],
        "quality": [],
    }

    # Iterate through sessions
    for subject_dir in sorted(_FIRING_RATES_DATA_DIR.iterdir()):
        subject = subject_dir.name
        if subject.startswith("."):
            continue
        print(f"\nProcessing {subject}\n")
        for session_dir in sorted(subject_dir.iterdir())[:1]:
            session = session_dir.name
            if session.startswith("."):
                continue
            print(f"\nProcessing {session}\n")

            # Load behavior data for this session
            session_behavior = triangle_behavior[
                (triangle_behavior["subject"] == subject)
                & (triangle_behavior["session"] == session)
            ]
            if len(session_behavior) == 0:
                continue

            # Add conditions to behavior data
            session_behavior["condition"] = session_behavior.apply(
                _get_condition, axis=1
            )
            session_behavior = session_behavior[
                ~session_behavior["condition"].isna()
            ]

            # Load firing rates for this session
            for probe_dir in sorted(session_dir.iterdir()):
                probe = probe_dir.name
                for quality_dir in sorted(probe_dir.iterdir()):
                    quality = quality_dir.name
                    for unit_trials_path in quality_dir.iterdir():
                        if not "trials" in unit_trials_path.name:
                            continue
                        unit = int(unit_trials_path.stem.split("_")[0])
                        spike_counts_suffix = unit_trials_path.stem.replace(
                            "trials", "spike_counts"
                        )
                        unit_spike_counts_path = (
                            quality_dir / f"{spike_counts_suffix}.pkl"
                        )
                        unit_trials = pickle.load(open(unit_trials_path, "rb"))
                        unit_spike_counts = pickle.load(
                            open(unit_spike_counts_path, "rb")
                        )

                        unit_firing_rates = _compute_firing_rates(
                            unit_trials,
                            unit_spike_counts,
                            session_behavior,
                            condition_on_identity=False,
                        )
                        firing_rates.append(unit_firing_rates)

                        unit_df["subject"].append(subject)
                        unit_df["session"].append(session)
                        unit_df["unit"].append(unit)
                        unit_df["brain_area"].append(
                            _PROBE_TO_BRAIN_AREA[probe]
                        )
                        unit_df["quality"].append(quality)

    # Save firing rates and unit data
    firing_rates = {
        condition: np.array([unit_frs[condition] for unit_frs in firing_rates])
        for condition in unit_firing_rates.keys()
    }
    unit_df = pd.DataFrame(unit_df)
    _WRITE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_WRITE_DIR / "firing_rates.npy", firing_rates)
    unit_df.to_csv(_WRITE_DIR / "unit_data.csv", index=False)


if __name__ == "__main__":
    main()
