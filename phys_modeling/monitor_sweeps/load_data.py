"""Functions to load data for monitoring sweeps."""

import json

import numpy as np
import pandas as pd


def load_data(log_dir, **kwargs):
    """Load data from a directory.

    Args:
        log_dir (str): Directory to load data from.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Data loaded from the directory.
    """
    # Get results dataframe
    results_data_dict = dict(
        train_mean_log_likelihood=[],
        train_mean_squared_error=[],
        train_num_neuron_trials=[],
        train_num_neurons=[],
        train_num_trials=[],
        test_mean_log_likelihood=[],
        test_mean_squared_error=[],
        test_num_neuron_trials=[],
        test_num_neurons=[],
        test_num_trials=[],
        stop_early=[],
        stop_step=[],
    )
    progress_data_dict = dict(
        step=[],
        train_mean_log_likelihood=[],
        train_mean_squared_error=[],
        test_mean_log_likelihood=[],
        test_mean_squared_error=[],
    )
    for k in kwargs.keys():
        results_data_dict[k] = []
        progress_data_dict[k] = []
    if not (log_dir / "test_metrics.json").exists():
        return None, None, None
    train_metrics = json.load(open(log_dir / "train_metrics.json"))
    test_metrics = json.load(open(log_dir / "test_metrics.json"))

    # Update results dataframe
    best_step = np.argmax(
        [metric["mean_log_likelihood"] for metric in test_metrics]
    )
    best_train_metric = train_metrics[best_step]
    best_test_metric = test_metrics[best_step]
    for k in best_train_metric.keys():
        if k == "step":
            continue
        results_data_dict["train_" + k].append(best_train_metric[k])
        results_data_dict["test_" + k].append(best_test_metric[k])
    for k, v in kwargs.items():
        results_data_dict[k].append(v)
    results_data_dict["stop_early"].append(best_step < len(test_metrics) - 1)
    results_data_dict["stop_step"].append(best_step)

    # Update progress dataframe
    for train_metric, test_metric in zip(train_metrics, test_metrics):
        for k in ["mean_log_likelihood", "mean_squared_error"]:
            progress_data_dict["train_" + k].append(train_metric[k])
            progress_data_dict["test_" + k].append(test_metric[k])
        progress_data_dict["step"].append(train_metric["step"])
        for k, v in kwargs.items():
            progress_data_dict[k].append(v)

    results_df = pd.DataFrame(results_data_dict)
    progress_df = pd.DataFrame(progress_data_dict)

    # Get loss dataframe
    loss_data_dict = dict(
        step=[],
        loss=[],
    )
    for k in kwargs.keys():
        loss_data_dict[k] = []
    losses_path = log_dir / "training_losses.json"
    if not losses_path.exists():
        return None, None, None
    losses = json.load(open(losses_path))
    for step, loss in enumerate(losses):
        loss_data_dict["step"].append(step)
        loss_data_dict["loss"].append(loss)
        for k, v in kwargs.items():
            loss_data_dict[k].append(v)
    loss_df = pd.DataFrame(loss_data_dict)

    hash_number = np.random.randint(int(1e9))
    results_df["hash"] = len(results_df) * [hash_number]
    loss_df["hash"] = len(loss_df) * [hash_number]
    progress_df["hash"] = len(progress_df) * [hash_number]

    return results_df, loss_df, progress_df


def _load_sessions(results_dir, subdirectory_keys, **kwargs):
    """Load all datasets from a directory."""
    if results_dir.name.startswith("."):
        return [], [], []

    # If results_dir is not a directory, return nothing
    if not results_dir.is_dir():
        return [], [], []

    # If we are at a leaf directory, load data
    if (results_dir / "config.json").exists():
        results_df, loss_df, progress_df = load_data(results_dir, **kwargs)
        if results_df is None:
            return [], [], []
        return [results_df], [loss_df], [progress_df]

    # Recurse into subdirectories
    results_dfs = []
    loss_dfs = []
    progress_dfs = []
    for directory in sorted(results_dir.iterdir()):
        tmp_kwargs = kwargs.copy()
        tmp_kwargs[subdirectory_keys[0]] = directory.name
        results_df, loss_df, progress_df = _load_sessions(
            directory,
            subdirectory_keys=subdirectory_keys[1:],
            **tmp_kwargs,
        )
        results_dfs.extend(results_df)
        loss_dfs.extend(loss_df)
        progress_dfs.extend(progress_df)
    return results_dfs, loss_dfs, progress_dfs


def load_all_datasets(
    results_dir, subdirectory_keys, check_product_space=True
):
    """Load dataframes from a directory."""
    results_dfs, loss_dfs, progress_dfs = _load_sessions(
        results_dir, subdirectory_keys
    )
    results_df = pd.concat(results_dfs, ignore_index=True)
    loss_df = pd.concat(loss_dfs, ignore_index=True)
    progress_df = pd.concat(progress_dfs, ignore_index=True)

    # Combine subject and session columns
    results_df["subject_session"] = (
        results_df["subject"] + "_" + results_df["session"]
    )
    loss_df["subject_session"] = loss_df["subject"] + "_" + loss_df["session"]
    progress_df["subject_session"] = (
        progress_df["subject"] + "_" + progress_df["session"]
    )

    # Remove subject and session columns
    results_df.drop(columns=["subject", "session"], inplace=True)
    loss_df.drop(columns=["subject", "session"], inplace=True)
    progress_df.drop(columns=["subject", "session"], inplace=True)

    print("Results dataframe columns:")
    print(results_df.columns)
    print("Loss dataframe columns:")
    print(loss_df.columns)
    print("Progress dataframe columns:")
    print(progress_df.columns)

    if check_product_space:
        # Check which elements of the product space are rows in results_df
        product_space_columns = ["subject_session"] + list(
            subdirectory_keys[2:]
        )
        unique_values_per_column = [
            results_df[column].unique() for column in product_space_columns
        ]
        product_space = pd.MultiIndex.from_product(
            unique_values_per_column, names=product_space_columns
        )
        for row in product_space:
            tmp = results_df.loc[
                (results_df[product_space_columns] == row).all(axis=1)
            ]
            if len(tmp) == 0:
                print(row)

    return results_df, loss_df, progress_df
