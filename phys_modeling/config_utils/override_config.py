"""Tools for overriding nodes of a configuration dictionary.

It is common to feed config overrides as a flag to a main run sctip, then use
that flag to modify the config with a line like this:
    config = override_config_from_json(config, FLAGS.config_overrides)
before building the config with
    build_from_config.build_from_config(config)

Typically you do not need config overrides for local runs, since you can just
modify the config directly. However, when running on openmind config overrides
become very important because they can be used to run modifications of the
config by just passing in flags to the run script.
"""

import json


def override_config_node(config, node, value):
    """Override a node of a config.

    Args:
        config: Base configuration dictionary to be modified.
        node: Iterable of keys of config corresponding to a path to a leaf node.
        value: New value for the leaf node.

    Returns:
        Dictionary, config with a new leaf value.
    """
    if not isinstance(node, (list, tuple)):
        raise ValueError("node must be an iterable, but is {}".format(node))

    if node[0] not in config:
        raise ValueError("node {} is not in config.".format(node))

    if len(node) == 1:
        config[node[0]] = value
    else:
        override_config_node(config[node[0]], node[1:], value)


def override_config(config, overrides):
    """Override nodes of the config based on override_dict.

    Args:
        config: Base configuration dictionary (possibly nested).
        overrides: Iterable of dictionaries, each with a 'node' and a 'value'
            element specifying how to modify config. For each such dictionary,
            'node' is a tuple of keys in config that corresponding to a path to
            a leaf in the config and 'value' is the new value for that leaf.

    Returns:
        Dictionary, config overridden according to override_dict.
    """
    if not overrides:
        return config

    # Remove non-dictionary elements. This is important for example to remove
    # 'log_dir=...' elements that can be added by sweep generators.
    overrides = [x for x in overrides if isinstance(x, dict)]

    for x in overrides:
        override_config_node(config, **x)
    return config


def override_config_from_json(config, json_overrides):
    """Override nodes of the config based on json_overrides.

    This is the same as override_config(), except it takes a json serialization
    of the overrides.
    """
    if not json_overrides:
        return config

    return override_config(config, json.loads(json_overrides))
