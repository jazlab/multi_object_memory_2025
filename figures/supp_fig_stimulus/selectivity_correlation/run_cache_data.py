"""Compute and cache triangle dataset mean firing rate data.

This will compute the mean firing rate for each unit during stimulus and delay
phases of the triangle task, conditionalized on object position,
save them to `../../../cache/figures/supp_fig_stimulus/orthogonality`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_BEHAVIOR_CACHE_PATH_TRIANGLE = Path(
    "../../../cache/behavior_processing/triangle.csv"
)
_SELECTIVITY_DATA_DIR = Path("../../../cache/phys_processing/selectivity")
_FIRING_RATES_DATA_DIR = Path(
    "../../../cache/phys_processing/spikes_to_trials"
)
_WRITE_DIR = Path(
    "../../../cache/figures/supp_fig_stimulus/selectivity_correlation"
)
_PROBE_TO_BRAIN_AREA = {
    "s0": "DMFC",
    "vprobe0": "FEF",
    "vprobe1": "FEF",
}


def _get_condition(row: pd.Series) -> str:
    """Get condition integer for a row in the behavior dataframe.

    Args:
        row: A row from the behavior dataframe.

    Returns:
        condition: An integer representing the condition for the row.
    """
    if row.on_triangle and row.num_objects == 1:
        condition = int(row.object_0_location)
    else:
        condition = np.nan
    return condition


def _compute_selectivity(fr_per_condition):
    """Compute selectivity metric for a unit."""
    values = np.array(
        [
            fr_per_condition[0],
            fr_per_condition[1],
            fr_per_condition[2],
        ]
    )
    values /= np.mean(values)
    return values[0], values[1], values[2]


def _compute_selectivities(
    df_behavior,
    df_delay_firing_rates,
    df_stimulus_firing_rates,
    df_selectivity,
):
    """Compute and save selectivities for a session."""

    # Sanity check that "trial" is the same in both firing rate dataframes
    assert np.array_equal(
        df_delay_firing_rates["trial"], df_stimulus_firing_rates["trial"]
    )

    # Add condition column to df_behavior
    df_behavior["condition"] = df_behavior.apply(_get_condition, axis=1)
    trials = df_delay_firing_rates["trial"].unique()
    conditions = {
        t: df_behavior[df_behavior["trial_num"] == t]["condition"].values[0]
        for t in trials
    }

    # Add condition to firing rate dataframes
    delay_trials = df_delay_firing_rates["trial"].values
    delay_conditions = np.array([conditions[t] for t in delay_trials])
    df_delay_firing_rates = df_delay_firing_rates.assign(
        condition=delay_conditions
    )
    stimulus_trials = df_stimulus_firing_rates["trial"].values
    stimulus_conditions = np.array([conditions[t] for t in stimulus_trials])
    df_stimulus_firing_rates = df_stimulus_firing_rates.assign(
        condition=stimulus_conditions
    )

    # Filter to only keep valid conditions
    df_delay_firing_rates = df_delay_firing_rates[
        ~np.isnan(df_delay_firing_rates["condition"])
    ]
    df_stimulus_firing_rates = df_stimulus_firing_rates[
        ~np.isnan(df_stimulus_firing_rates["condition"])
    ]

    # Iterate through units, fitting receptive field to each
    units = df_delay_firing_rates["unit"].unique()
    df = {
        "unit": [],
        "brain_area": [],
        "delay_0": [],
        "delay_1": [],
        "delay_2": [],
        "stimulus_0": [],
        "stimulus_1": [],
        "stimulus_2": [],
    }
    for unit in units:
        unit_selectivity = df_selectivity[df_selectivity["unit"] == unit]
        if not unit_selectivity.significant.values[0]:
            continue

        df_unit_delay = df_delay_firing_rates[
            df_delay_firing_rates["unit"] == unit
        ]
        df_unit_stimulus = df_stimulus_firing_rates[
            df_stimulus_firing_rates["unit"] == unit
        ]
        assert np.array_equal(
            df_unit_delay["trial"], df_unit_stimulus["trial"]
        )

        # Get firing rates per condition
        delay_per_condition = {
            int(cond): np.mean(
                df_unit_delay[df_unit_delay["condition"] == cond][
                    "firing_rate"
                ].values
            )
            for cond in df_unit_delay["condition"].unique()
        }
        stimulus_per_condition = {
            int(cond): np.mean(
                df_unit_stimulus[df_unit_stimulus["condition"] == cond][
                    "firing_rate"
                ].values
            )
            for cond in df_unit_stimulus["condition"].unique()
        }

        # Compute selectivity metrics
        d_x, d_y, d_z = _compute_selectivity(delay_per_condition)
        s_x, s_y, s_z = _compute_selectivity(stimulus_per_condition)
        probe = unit_selectivity.probe.values[0]
        df["unit"].append(unit)
        df["brain_area"].append(_PROBE_TO_BRAIN_AREA[probe])
        df["delay_0"].append(d_x)
        df["delay_1"].append(d_y)
        df["delay_2"].append(d_z)
        df["stimulus_0"].append(s_x)
        df["stimulus_1"].append(s_y)
        df["stimulus_2"].append(s_z)

    # Save dataframeq
    df = pd.DataFrame(df)
    subject = df_behavior["subject"].values[0]
    session = df_behavior["session"].values[0]
    write_path = _WRITE_DIR / subject / session
    write_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(write_path / "units.csv", index=False)

    return


def main():
    """Generate and save delay phase noise correlations."""
    # Load behavior data
    triangle_behavior = pd.read_csv(_BEHAVIOR_CACHE_PATH_TRIANGLE)

    # Iterate through sessions
    delay_data_dir = _FIRING_RATES_DATA_DIR / "delay_phase_firing_rates"
    stimulus_data_dir = _FIRING_RATES_DATA_DIR / "stimulus_phase_firing_rates"
    for subject_dir in sorted(delay_data_dir.iterdir()):
        subject = subject_dir.name
        if subject.startswith("."):
            continue
        print(f"\nProcessing {subject}\n")
        for session_dir in sorted(subject_dir.iterdir()):
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

            # Load firing rates for this session
            df_delay_firing_rates = pd.read_csv(
                session_dir / "fr.csv",
                index_col=0,
            )
            df_stimulus_firing_rates = pd.read_csv(
                stimulus_data_dir / subject / session / "fr.csv",
                index_col=0,
            )
            df_selectivity = pd.read_csv(
                _SELECTIVITY_DATA_DIR / subject / session / "units.csv",
                index_col=0,
            )

            _compute_selectivities(
                session_behavior,
                df_delay_firing_rates,
                df_stimulus_firing_rates,
                df_selectivity,
            )


if __name__ == "__main__":
    main()
