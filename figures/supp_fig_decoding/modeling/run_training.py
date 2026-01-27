"""Run model training."""

import importlib

_CONFIG = "config_1_triangle"


def main():
    """Main function to run model training."""
    config_module = importlib.import_module(_CONFIG)
    trainer = config_module.get_trainer()
    trainer()


if __name__ == "__main__":
    main()
