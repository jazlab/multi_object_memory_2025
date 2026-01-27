"""Compute and cache delay phase noise correlations for the ring dataset.

This will compute the noise correlations for all ring dataset sessions and
save them to `../../../cache/figures/figure_5/noise_correlations`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.append("../../../behavior_processing")
import load_ring_data

_BASE_NEURAL_DIR = Path(
    "../../../cache/phys_processing/spikes_to_trials/delay_phase_firing_rates"
)
_BASE_SELECTIVITY_DIR = Path("../../../cache/phys_processing/selectivity")
_BASE_WRITE_DIR = Path(
    "../../../cache/figures/figure_5/noise_correlations_ring"
)


def cache_noise_correlations(df_behavior):
    subject = df_behavior.subject.unique()
    if len(subject) != 1:
        raise ValueError("Expected exactly one subject in df_behavior.")
    subject = subject[0]
    session = df_behavior.session.unique()
    if len(session) != 1:
        raise ValueError("Expected exactly one session in df_behavior.")
    session = session[0]
    print(f"Processing {subject} - {session}")

    write_path = _BASE_WRITE_DIR / subject / session / "noise_correlations.csv"
    if write_path.exists():
        print(f"Skipping {subject} - {session} as results already exist.")
        return

    # Neural data
    neural_path = _BASE_NEURAL_DIR / subject / session / "fr.csv"
    if not neural_path.exists():
        print(f"Skipping {subject} - {session} as neural data does not exist.")
        return
    df_neural = pd.read_csv(neural_path, index_col=0)
    print(df_neural.columns)

    # Selectivity data
    df_units = pd.read_csv(
        _BASE_SELECTIVITY_DIR / subject / session / "units.csv", index_col=0
    )
    print(df_units.columns)

    # Compute embeddings
    embeddings = np.zeros((len(df_behavior), 6))
    id_to_start_index = {
        "a": 0,
        "b": 2,
        "c": 4,
    }
    for i, row in df_behavior.iterrows():
        id_0 = row["object_0_id"]
        x_0 = row["object_0_x"]
        y_0 = row["object_0_y"]
        id_1 = row["object_1_id"]
        x_1 = row["object_1_x"]
        y_1 = row["object_1_y"]

        start_index_0 = id_to_start_index[id_0]
        embeddings[i, start_index_0] = x_0
        embeddings[i, start_index_0 + 1] = y_0

        if id_1 in id_to_start_index:
            start_index_1 = id_to_start_index[id_1]
            embeddings[i, start_index_1] = x_1
            embeddings[i, start_index_1 + 1] = y_1

    # Add embeddings to behavior dataframe
    for i in range(6):
        df_behavior[f"embedding_{i}"] = embeddings[:, i]

    # Initialize results dictionary
    results_dict = {
        "unit_0": [],
        "unit_1": [],
        "signal_ccs": [],
        "noise_ccs": [],
        "num_signal_trials": [],
        "num_noise_trials": [],
    }

    for unit_index, unit_0 in enumerate(df_units.unit.unique()):
        print(
            f"Unit {unit_0} ({unit_index + 1}/{len(df_units.unit.unique())})..."
        )

        for unit_1 in df_units.unit.unique():
            if unit_0 == unit_1:
                continue  # Skip self-correlations

            df_neural_0 = df_neural[df_neural.unit == unit_0]
            df_neural_1 = df_neural[df_neural.unit == unit_1]

            # Keep only trials with both units
            df_neural_0 = df_neural_0[
                df_neural_0.trial.isin(df_neural_1.trial)
            ]
            df_neural_1 = df_neural_1[
                df_neural_1.trial.isin(df_neural_0.trial)
            ]

            ################################################################################
            #### Compute signal correlations
            ################################################################################

            # Get 1-object trials
            trial_nums = df_neural_0.trial.values
            tmp_df_behavior = df_behavior[
                df_behavior.trial_num.isin(trial_nums)
            ]
            keep_inds = tmp_df_behavior.num_objects == 1
            tmp_df_behavior = tmp_df_behavior[keep_inds]
            num_signal_trials = len(tmp_df_behavior)
            if len(tmp_df_behavior) < 2:
                print(
                    f"Skipping {unit_0} and {unit_1} due to no 1-object trials."
                )
                continue
            df_neural_0_keep = df_neural_0[
                df_neural_0.trial.isin(tmp_df_behavior.trial_num)
            ]
            df_neural_1_keep = df_neural_1[
                df_neural_1.trial.isin(tmp_df_behavior.trial_num)
            ]

            # For each trial, compute the nearest trial
            embeddings = tmp_df_behavior[
                [f"embedding_{i}" for i in range(6)]
            ].values
            embeddings_diffs = embeddings[:, None, :] - embeddings[None, :, :]
            embeddings_diffs = np.linalg.norm(embeddings_diffs, axis=-1)
            embeddings_diffs[
                np.arange(len(embeddings)), np.arange(len(embeddings))
            ] = np.inf
            nearest_trial_indices = np.argmin(embeddings_diffs, axis=1)

            # Get the firing rates for the nearest trials
            neural_0_signal = df_neural_0_keep.firing_rate.values
            neural_1_signal = df_neural_1_keep.firing_rate.values[
                nearest_trial_indices
            ]

            # Compute the correlation coefficient
            signal_corr_coef = pearsonr(neural_0_signal, neural_1_signal)[0]

            ################################################################################
            #### Compute noise correlations
            ################################################################################

            # Get 2-object trials
            trial_nums = df_neural_0.trial.values
            tmp_df_behavior = df_behavior[
                df_behavior.trial_num.isin(trial_nums)
            ]
            keep_inds = tmp_df_behavior.num_objects == 2
            tmp_df_behavior = tmp_df_behavior[keep_inds]
            num_noise_trials = len(tmp_df_behavior)
            df_neural_0_keep = df_neural_0[
                df_neural_0.trial.isin(tmp_df_behavior.trial_num)
            ]
            df_neural_1_keep = df_neural_1[
                df_neural_1.trial.isin(tmp_df_behavior.trial_num)
            ]

            # For each trial, compute the num_neighbors nearest trials
            num_neighbors = 10
            if len(tmp_df_behavior) < num_neighbors + 1:
                print(
                    f"Skipping {unit_0} and {unit_1} due to not enough 2-object trials."
                )
                continue
            embeddings = tmp_df_behavior[
                [f"embedding_{i}" for i in range(6)]
            ].values
            embeddings_diffs = embeddings[:, None, :] - embeddings[None, :, :]
            embeddings_diffs = np.linalg.norm(embeddings_diffs, axis=-1)
            embeddings_diffs[
                np.arange(len(embeddings)), np.arange(len(embeddings))
            ] = np.inf
            nearest_trial_indices = np.argsort(embeddings_diffs, axis=1)[
                :, :num_neighbors
            ]

            # Compute the residual firing rates for neural_0 and neural_1
            residuals_0 = []
            residuals_1 = []
            for i in range(len(tmp_df_behavior)):
                fr_0 = df_neural_0_keep.firing_rate.values[i]
                fr_1 = df_neural_1_keep.firing_rate.values[i]
                nearest_indices = nearest_trial_indices[i]
                nearest_fr_0 = df_neural_0_keep.firing_rate.values[
                    nearest_indices
                ]
                nearest_fr_1 = df_neural_1_keep.firing_rate.values[
                    nearest_indices
                ]
                residual_0 = fr_0 - np.mean(nearest_fr_0)
                residual_1 = fr_1 - np.mean(nearest_fr_1)
                residuals_0.append(residual_0)
                residuals_1.append(residual_1)
            residuals_0 = np.array(residuals_0)
            residuals_1 = np.array(residuals_1)

            # Compute the correlation coefficient
            noise_corr_coef = pearsonr(residuals_0, residuals_1)[0]

            # Append the results
            results_dict["unit_0"].append(unit_0)
            results_dict["unit_1"].append(unit_1)
            results_dict["signal_ccs"].append(signal_corr_coef)
            results_dict["noise_ccs"].append(noise_corr_coef)
            results_dict["num_signal_trials"].append(num_signal_trials)
            results_dict["num_noise_trials"].append(num_noise_trials)

    # Save the results
    results_df = pd.DataFrame(results_dict)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(write_path, index=False)


def main():
    """Generate and save noise correlations."""
    df_behavior = load_ring_data.load_data()
    for subject in df_behavior.subject.unique():
        df_subject = df_behavior[df_behavior.subject == subject]
        for session in df_subject.session.unique():
            df_session = df_behavior[
                (df_behavior.subject == subject)
                & (df_behavior.session == session)
            ].reset_index(drop=True)
            cache_noise_correlations(df_session)


if __name__ == "__main__":
    main()
