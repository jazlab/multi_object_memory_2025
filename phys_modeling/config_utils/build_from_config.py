"""Recursively instantiate config.

The entry point is build_from_config(), which is typically used to instantiate
all elements of a config dictionary. This is typically called in a run script
after the config has been loaded and config overrides applied.
"""

import importlib
import logging


def build_from_config(x):
    """Recursively instantiate config.

    The general idea is to structure all hyperparameters of an experiment into
    one big dictionary so that the hyperparameters can be easily swept over by
    altering leaves of the dictionary. Then in the main run script, the
    dictionary is converted into an object (or a bunch of objects) that are run
    in some way.

    This function does that conversion, i.e. it takes a dictionary and returns a
    constructed object. The constructor is the value of a 'constructor' key, and
    the args and kwargs are values of the 'args' and 'kwargs' keys. For example,
        {'constructor': MyClass, 'args': [1, 2], 'kwargs': {'a': 3, 'b': 4}}
    will be converted to
        MyClass(1, 2, a=3, b=4)
    The 'args' and 'kwargs' keys are optional.

    Furthermore, this function recurses through lists, tuples, and dictionaries,
    so the values of 'args' and 'kwargs' can themselves contain dictionaries
    with 'constructor', 'args', and 'kwargs' keys.

    In addition to instantiating dictionaries with a 'constructor' key, this
    function does special things to two other keys:
        * 'module'. If it encounters a dictionary with a 'module' key, it will
          look for a 'method' key and return the method imported from the
          module. This means that any function (including class names
          themselves) can be represented as a string in the config and can be
          swept over. For example, to sweep over activation function you can do:
            {'constructor': {'module': 'torch.nn', 'method': 'ReLU'}}
          then in a sweep alter the 'constructor' -> 'method' node of the
          config, e.g. to 'Sigmoid' or 'Tanh'.
        * 'choice'. Sometimes you may need to sweep over methods or classes with
          different args/kwargs signatures. In this case you can create a
          nested 'options' dictionary with all of the options you want to sweep
          over and specify which to use by the 'choice' value. For example:
            {'activation_fn': {
                'choice': 'relu',
                'options': {
                    'relu': torch.nn.ReLU,
                    'sigmoid': torch.nn.Sigmoid',
                    'elu': {
                        'constructor': torch.nn.ELU,
                        'kwargs': {'alpha': 1.}
                    },
                },
            }}
          In this example, since 'choice' has value 'relu', the options['relu']
          value will be used, but you could sweep over 'choice' to use the other
          options and even sweep over args/kwargs of those options, e.g.
          'elu' -> 'kwargs' -> 'alpha'.

    Args:
        x: Input of any type. If dict containing 'constructor', should contain
            only 'constructor' and optionally 'args' and 'kwargs' keys. If dict
            containing 'module', must also contain 'method' key. If dict
            containing 'choice', must also contain 'options' key.

    Returns:
        Module constructed from x.
    """
    if isinstance(x, list):
        return [build_from_config(i) for i in x]
    elif isinstance(x, tuple):
        return tuple([build_from_config(i) for i in x])
    elif isinstance(x, dict):
        if "constructor" in x:
            # Instantiate module on 'args' and 'kwargs'
            constructor = build_from_config(x["constructor"])

            if "args" in x:
                args = tuple([build_from_config(v) for v in x["args"]])
            else:
                args = ()
            if "kwargs" in x:
                kwargs = {
                    k: build_from_config(v) for k, v in x["kwargs"].items()
                }
            else:
                kwargs = {}
            logging.info("Constructor: {}.".format(constructor))
            return constructor(*args, **kwargs)
        elif "module" in x:
            module = importlib.import_module(x["module"])
            method = getattr(module, x["method"])
            return method
        elif "choice" in x:
            return build_from_config(x["options"][x["choice"]])
        else:
            return {k: build_from_config(v) for k, v in x.items()}
    else:  # x has a type that we won't worry about
        return x
