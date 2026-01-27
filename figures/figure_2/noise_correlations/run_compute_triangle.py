"""Compute and cache delay phase noise correlations for the triangle dataset.

This will compute the noise correlations for all triangle dataset sessions and
save them to `../../../cache/figures/figure_2/noise_correlations`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_BEHAVIOR_CACHE_PATH_TRIANGLE = Path(
    "../../../cache/behavior_processing/triangle.csv"
)
_SELECTIVITY_DATA_DIR = Path("../../../cache/phys_processing/selectivity")
_FIRING_RATES_DATA_DIR = Path(
    "../../../cache/phys_processing/spikes_to_trials/delay_phase_firing_rates"
)
_WRITE_DIR = Path(
    "../../../cache/figures/figure_2/noise_correlations_triangle"
)


def _get_condition(row: pd.Series, condition_on_identity: bool = True) -> str:
    """Get condition string for a row in the behavior dataframe.

    Args:
        row: A row from the behavior dataframe.
        condition_on_identity: If True, conditionalize on identity of objects.
            If False, only conditionalize on position of objects.

    Returns:
        condition: A string representing the condition for the row.
    """
    if not row.on_triangle:
        condition = np.nan
    elif row.num_objects == 1:
        if condition_on_identity:
            condition = f"{int(row.object_0_location)}_{row.object_0_id}"
        else:
            condition = f"{int(row.object_0_location)}"
    elif row.num_objects == 2:
        loc_0 = int(row.object_0_location)
        loc_1 = int(row.object_1_location)
        if condition_on_identity:
            id_0 = row.object_0_id
            id_1 = row.object_1_id
            if loc_0 < loc_1:
                condition = f"{loc_0}_{id_0}_{loc_1}_{id_1}"
            else:
                condition = f"{loc_1}_{id_1}_{loc_0}_{id_0}"
        else:
            if loc_0 < loc_1:
                condition = f"{loc_0}_{loc_1}"
            else:
                condition = f"{loc_1}_{loc_0}"
    else:
        condition = np.nan
    return condition


def _get_combination_triples(condition_on_identity: bool = True):
    """Get all combinations of conditions for noise correlation computation.

    Args:
        condition_on_identity: If True, conditionalize on identity of objects.
            If False, only conditionalize on position of objects.

    Returns:
        condition_triples: List of tuples, each containing three conditions.
            Each tuple contains two single-object conditions and one combination
            condition.
    """
    condition_triples = []
    if condition_on_identity:
        for loc_0 in range(2):
            for loc_1 in range(loc_0 + 1, 3):
                for id_0 in ["a", "b", "c"]:
                    for id_1 in ["a", "b", "c"]:
                        if id_1 == id_0:
                            continue
                        new_conditions = [
                            f"{loc_0}_{id_0}",
                            f"{loc_1}_{id_1}",
                            f"{loc_0}_{id_0}_{loc_1}_{id_1}",
                        ]
                        condition_triples.append(new_conditions)
    else:
        for loc_0 in range(2):
            for loc_1 in range(loc_0 + 1, 3):
                new_conditions = [f"{loc_0}", f"{loc_1}", f"{loc_0}_{loc_1}"]
                condition_triples.append(new_conditions)

    return condition_triples


def _get_signal_correlation(firing_rates_0, firing_rates_1, n_bootstraps=10):
    """Compute signal correlation between two sets of firing rates."""
    fr_i_0, fr_j_0 = firing_rates_0
    fr_i_1, fr_j_1 = firing_rates_1
    n_trials_0 = len(fr_i_0)
    n_trials_1 = len(fr_i_1)

    signal_correlation = []
    for _ in range(n_bootstraps):
        # Shuffle trials
        shuffle_inds_0 = np.random.permutation(n_trials_0)
        shuffle_inds_1 = np.random.permutation(n_trials_1)
        fr_j_0_shuffled = fr_j_0[shuffle_inds_0]
        fr_j_1_shuffled = fr_j_1[shuffle_inds_1]
        fr_i = np.concatenate([fr_i_0, fr_i_1])
        fr_j = np.concatenate([fr_j_0_shuffled, fr_j_1_shuffled])
        signal_correlation.append(np.corrcoef(fr_i, fr_j)[0, 1])

    return np.mean(signal_correlation)


def _compute_correlations(
    df_firing_rates: pd.DataFrame,
    df_behavior: pd.DataFrame,
    write_path: Path,
    condition_on_identity: bool = True,
):
    """Compute and save noise correlations for a session.

    Args:
        df_firing_rates: DataFrame with firing rates for each unit and trial.
        df_behavior: DataFrame with behavior data for the session.
        write_path: Path to save the results.
        condition_on_identity: If True, conditionalize on identity of objects.
            If False, only conditionalize on position of objects.
    """
    # Add condition column to session_behavior
    df_behavior["condition"] = df_behavior.apply(
        _get_condition, axis=1, condition_on_identity=condition_on_identity
    )
    trials = df_firing_rates["trial"].unique()
    conditions = np.array(
        [
            df_behavior[df_behavior["trial_num"] == t]["condition"].values[0]
            for t in trials
        ]
    )

    # Make matrix of firing rates of shape [units, trials]
    units = df_firing_rates["unit"].unique()
    unit_to_index = {unit: i for i, unit in enumerate(units)}
    trial_to_index = {trial: i for i, trial in enumerate(trials)}
    firing_rate_mat = np.nan * np.ones((len(units), len(trials)))
    for i, row in df_firing_rates.iterrows():
        trial_index = trial_to_index[row.trial]
        unit_index = unit_to_index[row.unit]
        firing_rate_mat[unit_index, trial_index] = row[f"firing_rate"]

    # Remove firing rates with invalid conditions
    remove_trials = conditions == "nan"
    conditions = conditions[~remove_trials]
    firing_rate_mat = firing_rate_mat[:, ~remove_trials]

    # Get triples for correlation
    condition_triples = _get_combination_triples(
        condition_on_identity=condition_on_identity
    )

    # Compute noise correlations
    corr_data = {
        "unit_0": [],
        "unit_1": [],
        "condition_0": [],
        "condition_1": [],
        "condition_combo": [],
        "n_cond_0_trials": [],
        "n_cond_1_trials": [],
        "n_combo_trials": [],
        "noise_correlation": [],
        "signal_correlation": [],
    }
    for cond_count, (cond_0, cond_1, cond_combo) in enumerate(
        condition_triples
    ):
        print(
            f"Computing correlations for set {cond_count + 1} of {len(condition_triples)}"
        )
        num_units = len(units)
        for i in range(num_units):
            for j in range(i + 1, num_units):
                # Get signal correlation on single-object trials
                firing_rates_per_cond = {}
                for cond in [cond_0, cond_1, cond_combo]:
                    trials_cond = np.where(conditions == cond)[0]
                    fr_i = firing_rate_mat[i, trials_cond]
                    fr_j = firing_rate_mat[j, trials_cond]
                    keep_inds = ~np.isnan(fr_i) & ~np.isnan(fr_j)
                    fr_i = fr_i[keep_inds]
                    fr_j = fr_j[keep_inds]
                    firing_rates_per_cond[cond] = (fr_i, fr_j)

                skip = False
                for k, v in firing_rates_per_cond.items():
                    if len(v[0]) == 0:
                        skip = True
                if skip:
                    continue

                signal_correlation = _get_signal_correlation(
                    firing_rates_per_cond[cond_0],
                    firing_rates_per_cond[cond_1],
                )

                # Compute noise correlation
                noise_correlation = np.corrcoef(
                    firing_rates_per_cond[cond_combo][0],
                    firing_rates_per_cond[cond_combo][1],
                )[0, 1]

                # Update corr_data
                corr_data["unit_0"].append(units[i])
                corr_data["unit_1"].append(units[j])
                corr_data["condition_0"].append(cond_0)
                corr_data["condition_1"].append(cond_1)
                corr_data["condition_combo"].append(cond_combo)
                corr_data["n_cond_0_trials"].append(
                    len(firing_rates_per_cond[cond_0][0])
                )
                corr_data["n_cond_1_trials"].append(
                    len(firing_rates_per_cond[cond_1][0])
                )
                corr_data["n_combo_trials"].append(
                    len(firing_rates_per_cond[cond_combo][0])
                )
                corr_data["signal_correlation"].append(signal_correlation)
                corr_data["noise_correlation"].append(noise_correlation)
    df_corr = pd.DataFrame(corr_data)

    # Save conditions, units, firing_rate_mat, and noise_correlations
    np.save(write_path / "units.npy", units)
    np.save(write_path / "conditions.npy", conditions)
    np.save(write_path / "firing_rate_mat.npy", firing_rate_mat)
    df_corr.to_csv(write_path / "df_correlations.csv", index=False)

    return


def main():
    """Generate and save delay phase noise correlations."""
    # Load behavior data
    triangle_behavior = pd.read_csv(_BEHAVIOR_CACHE_PATH_TRIANGLE)

    # Iterate through sessions
    for subject_dir in sorted(_FIRING_RATES_DATA_DIR.iterdir()):
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
            df_firing_rates = pd.read_csv(session_dir / "fr.csv", index_col=0)

            # Load selectivity data
            selectivity_df = pd.read_csv(
                _SELECTIVITY_DATA_DIR / subject / session / "units.csv"
            )
            selectivity_df = selectivity_df[
                selectivity_df.mean_firing_rate > 1
            ]

            # Filter units to only keep selective ones
            df_firing_rates = df_firing_rates[
                df_firing_rates["unit"].isin(selectivity_df["unit"])
            ]

            write_path = _WRITE_DIR / subject / session
            write_path.mkdir(parents=True, exist_ok=True)
            if (write_path / "units.npy").exists():
                continue
            _compute_correlations(
                df_firing_rates, session_behavior, write_path
            )


if __name__ == "__main__":
    main()
