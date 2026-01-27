"""Config."""


def get_config():
    config = dict(
        constructor=dict(
            module="synthetic_datasets.generator",
            method="Generator",
        ),
        kwargs=dict(
            log_dir="../../cache/phys_modeling/training_logs/main_results/triangle_wsum_linear_positive/Perle/2022-05-26/2",
            random_seed=0,
        ),
    )
    return config
