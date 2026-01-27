"""Run this script to process and cache behavior data.

Usage:
$ python3 run_cache_data.py

This writes dataframes "../cache/behavior/triangle.csv" and
"../cache/behavior/ring.csv" with data for the triangle and ring tasks.

Note: For all angular data columns (e.g. "response_theta", "object_0_theta",
...), the angle has zero at due right and increases counterclockwise in radians,
like polar coordinates for complex numbers.

This script requires the behavior dandi data to be downloaded and cached, which
you can do by navigating to `../` and running
`$ python download_dandi_data.py --modality=behavior`.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

_DATA_DIR = Path("../cache/dandi_data/behavior")
_WRITE_DIR = Path("../cache/behavior")
_DATA_COLUMNS = [
    "subject",
    "session",
    "task",
    "trial_num",
    "completed",
    "time_fixation_onset",
    "time_stimulus_onset",
    "time_delay_onset",
    "time_cue_onset",
    "time_response_onset",
    "time_feedback_onset",
    "time_iti_onset",
    "num_objects",
    "object_0_r",
    "object_0_theta",
    "object_0_id",
    "object_1_r",
    "object_1_theta",
    "object_1_id",
    "object_2_r",
    "object_2_theta",
    "object_2_id",
    "target_object_index",
    "response_time",
    "response_r",
    "response_theta",
]
_RADIUS = 0.35  # Radius of the stimuli
_TRIANGLE_THETAS = [
    0,  # right
    2 * np.pi / 3,  # top-left
    4 * np.pi / 3,  # bottom-left
]


def _add_triangle_locations(triangle_df: pd.DataFrame) -> None:
    """Add triangle locations to the triangle dataframe."""
    on_triangle = []
    object_locations = []
    for _, row in triangle_df.iterrows():
        # Check if the stimuli are on the triangle
        object_locs = []
        stimuli_on_triangle = True
        for i in range(3):
            r, theta = row[f"object_{i}_r"], row[f"object_{i}_theta"]

            # Check if the object exists
            if np.isnan(r) or np.isnan(theta):
                object_locs.append(np.nan)
                continue

            # Check if the object radius is close to the triangle radius
            if not np.isclose(r, _RADIUS, atol=0.01):
                stimuli_on_triangle = False
                object_locs.append(np.nan)
                continue

            # Check if the object theta is close to one of the triangle thetas
            close_thetas = [
                np.isclose(theta, t, atol=0.01) for t in _TRIANGLE_THETAS
            ]
            if not np.any(close_thetas):
                stimuli_on_triangle = False
                object_locs.append(np.nan)
                continue

            # Find nearest theta
            nearest_loc = np.argwhere(close_thetas)[0, 0]
            object_locs.append(nearest_loc)

        on_triangle.append(stimuli_on_triangle)
        object_locations.append(object_locs)

    triangle_df["on_triangle"] = on_triangle
    for i in range(3):
        triangle_df[f"object_{i}_location"] = [x[i] for x in object_locations]


def _cartesian_to_polar(x: float, y: float) -> tuple[float, float]:
    """Convert Cartesian coordinates to polar coordinates."""
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    theta = np.arctan2(y - 0.5, x - 0.5) % (2 * np.pi)
    return r, theta


def _str_to_bool(s: str) -> bool:
    """Convert a string to a boolean."""
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError(f"Cannot convert string {s} to boolean")


def _get_session_df(nwbfile: NWBHDF5IO) -> pd.DataFrame:
    """Get a dataframe of session data from an NWB file.

    Args:
        nwbfile: NWB file object.

    Returns:
        session_df: DataFrame of session data.
    """
    subject = nwbfile.subject.subject_id
    session = nwbfile.session_id
    trial_df = nwbfile.trials.to_dataframe()

    # Ensure all values of delay_object_blanks are False
    if trial_df.delay_object_blanks.any():
        raise ValueError("delay_object_blanks contains True values")

    # Convert list columns to lists
    trial_df.stimulus_object_target = trial_df.stimulus_object_target.apply(
        lambda x: [_str_to_bool(s.strip()) for s in x[1:-1].split(",")]
    )

    # Create session_df to append to df
    session_df = pd.DataFrame(
        columns=[x for x in _DATA_COLUMNS if x != "task"]
    )
    for trial_num, row in trial_df.iterrows():
        completed = not row.broke_fixation and np.isfinite(row.response_time)
        trial_data = dict(
            subject=subject,
            session=session,
            trial_num=trial_num,
            completed=completed,
            time_fixation_onset=row.phase_fixation_time,
            time_stimulus_onset=row.phase_stimulus_time,
            time_delay_onset=row.phase_delay_time,
            time_cue_onset=row.phase_cue_time,
            time_response_onset=row.phase_response_time,
            time_feedback_onset=row.phase_reveal_time,
            time_iti_onset=row.phase_iti_time,
            response_time=row.response_time,
        )

        # Add response data
        if row.response_position is None:
            trial_data["response_r"] = np.nan
            trial_data["response_theta"] = np.nan
        else:
            response_r, response_theta = _cartesian_to_polar(
                row.response_position[0], row.response_position[1]
            )
            trial_data["response_r"] = response_r
            trial_data["response_theta"] = response_theta
            trial_data["response_x"] = row.response_position[0]
            trial_data["response_y"] = row.response_position[1]

        # Add object data
        trial_data["target_object_index"] = np.argwhere(
            row.stimulus_object_target
        )[0, 0]
        obj_positions = ast.literal_eval(row.stimulus_object_positions)
        obj_identities = ast.literal_eval(row.stimulus_object_identities)
        trial_data["num_objects"] = len(obj_positions)
        for i in range(3):
            if i < len(obj_positions):
                obj_r, obj_theta = _cartesian_to_polar(
                    obj_positions[i][0], obj_positions[i][1]
                )
                trial_data[f"object_{i}_r"] = obj_r
                trial_data[f"object_{i}_theta"] = obj_theta
                trial_data[f"object_{i}_x"] = obj_positions[i][0]
                trial_data[f"object_{i}_y"] = obj_positions[i][1]
                trial_data[f"object_{i}_id"] = obj_identities[i]
            else:
                trial_data[f"object_{i}_r"] = np.nan
                trial_data[f"object_{i}_theta"] = np.nan
                trial_data[f"object_{i}_id"] = np.nan

        # Add trial data to session_df
        trial_data = pd.DataFrame(trial_data, index=[0])
        if len(session_df) == 0:
            session_df = trial_data
        else:
            session_df = pd.concat([session_df, trial_data], ignore_index=True)

    return session_df


def main():
    """Generate and save behavior data."""
    # Create an empty dataframe with the correct columns
    df_per_task = {
        "triangle": pd.DataFrame(columns=_DATA_COLUMNS),
        "ring": pd.DataFrame(columns=_DATA_COLUMNS),
    }

    # Append each session to the dataframe
    for subject_dir in sorted(_DATA_DIR.iterdir()):
        if subject_dir.name.startswith("."):
            continue
        for session_file in sorted(subject_dir.iterdir()):
            if session_file.name.startswith("."):
                continue
            with NWBHDF5IO(session_file, "r") as io:
                print(f"Processing {session_file}")
                nwbfile = io.read()
                session_df = _get_session_df(nwbfile=nwbfile)

                # Figure out which task the session is
                three_obj_trials = np.mean(session_df.num_objects == 3)
                if three_obj_trials > 0.2:
                    task = "triangle"
                elif three_obj_trials == 0:
                    task = "ring"
                else:
                    raise ValueError(
                        "Invalid fraction of three-object trials: "
                        f"{three_obj_trials}"
                    )

                # Append session_df to the correct dataframe
                if len(df_per_task[task]) == 0:
                    df_per_task[task] = session_df
                else:
                    df_per_task[task] = pd.concat(
                        [df_per_task[task], session_df], ignore_index=True
                    )

    # Add triangle locations to triangle dataframe
    _add_triangle_locations(df_per_task["triangle"])

    # Save the dataframe to a CSV file
    for task, df in df_per_task.items():
        write_path = _WRITE_DIR / f"{task}.csv"
        print(f"Writing {task} data to {write_path}")
        if write_path.exists():
            print(f"Overwriting existing file {write_path}")
        write_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(write_path, index=False)


if __name__ == "__main__":
    main()
