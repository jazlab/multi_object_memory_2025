"""Functions for creating hyperparameter sweeps.

A hyperparameter spec is a list of dictionaries, each of which has:
    * 'node': A list of strings, specifying the path to a leaf in the config.
    * 'value': The new value for that leaf.

Thus a spec is exactly the 'overrides' argument to
./override_config.override_config(), so to run a set of experiments we just have
to create a sweep of specs, then pass each one to override_config in the main
run script.

This file has tools to create a sweep (a list of specs).
"""

import itertools
import json


def discrete(node, values):
    """Discrete sweep for a single node.

    This can then be fed into product() or zipper() to combine with sweeps over
    other nodes.

    Args:
        node: Iterable of strings. Must be a valid 'node' argument to
            override_config.override_config_node().
        values: Singleton or list or tuple of values to sweep for the node.

    Returns:
        Hyperparameter sweep (list of specs) over values.
    """
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [[{"node": node, "value": v}] for v in values]


def product(*sweeps):
    """Product of multiple sweeps.

    Args:
        sweeps: Iterable of sweeps.

    Returns:
        One big sweep of length N_0 * N_1 * ... * N_k where
            N_i = len(sweeps[i]).
    """
    return [list(itertools.chain(*x)) for x in itertools.product(*sweeps)]


def zipper(*sweeps):
    """Zip of sweeps, all of which must have the same length.

    Args:
        sweeps: Iterable of sweeps, each of the same length N.

    Returns:
        One sweep of length N combining all the specs in the sweeps.
    """
    # Check all sweeps have the same length, since python zip() doesn't check
    sweep_lengths = [len(sweep) for sweep in sweeps]
    if not all(length == sweep_lengths[0] for length in sweep_lengths):
        raise ValueError(
            "All sweeps fed into zipper() must have the same length, but they "
            "have lengths {}".format(sweep_lengths)
        )

    return [list(itertools.chain(*x)) for x in zip(*sweeps)]


def chain(*sweeps):
    """Chain sweeps, namely concatenate them.

    Args:
        sweeps: Iterable of sweeps.

    Returns:
        One sweep of length N_0 + N_1 + ... + N_k where N_i = len(sweeps[i]).
    """
    return list(itertools.chain(*sweeps))


def serialize_sweep_elements(sweep):
    """Convert a sweep to a list of json strings.

    Each element of the returned list can be passed to
    override_config.override_config_from_json().
    """
    return [json.dumps(spec) for spec in sweep]


def add_log_dir_sweep(sweep):
    """Add 'log_dir=...' sweep to a sweep of other parameters.

    The '...' in 'log_dir=...' is a contracted string of the hyparameters being
    swept over and their values.

    The 'log_dir=...' elements of the resulting sweep do not conform to the
    standard format of a sweep entry. Specifically, the elements are strings
    instead of dictionaries {'node': ..., 'value': ...}. Consequently, they must
    be filtered out of the sweep before being passed to
    override_config.override_config().

    Args:
        sweep: A sweep of some parameters. This should be a sweep of all of the
            parameters that are varying in the sweep (i.e. should not include
            parameters that are constant across the sweep) to ensure that the
            log directories are as short as possible while being unique per
            sweep element.

    Returns:
        sweep in which each spec has a 'log_dir=...' string appended to it.
    """
    log_dir_sweep = []
    # Remove these characters from log_dir
    characters_to_remove = [" ", "[", "]", "/", "'", '"']
    for s in sweep:
        log_dir = ""
        for var in s:
            log_dir += str(var["node"][-1])
            log_dir += "_"
            str_value = str(var["value"])
            for c in characters_to_remove:
                str_value = str_value.replace(c, "")
            log_dir += str_value
            log_dir += "_"
        log_dir = log_dir[:-1]  # Remove the last '_'
        log_dir_sweep.append(["log_dir=" + log_dir])

    sweep = zipper(sweep, log_dir_sweep)
    return sweep
