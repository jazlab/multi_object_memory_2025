"""Config."""


def _get_dataset_config():
    config = dict(
        constructor=dict(
            module="dataset",
            method="Dataset",
        ),
        kwargs=dict(
            subject="Perle",
            session="2022-06-13",
            phase="delay",
            unit_filter=dict(
                constructor=dict(
                    module="unit_filters",
                    method="UnitFilter",
                ),
                kwargs=dict(
                    brain_areas=("DMFC", "FEF"),
                    qualities=("good", "mua"),
                ),
            ),
            trial_filter=dict(
                constructor=dict(
                    module="trial_filters",
                    method="RingTrialFilter",
                ),
            ),
            max_n_objects=2,
            test_fraction=0.2,
            random_seed=2,
        ),
    )
    return config


def _get_model_config():
    config = dict(
        constructor=dict(
            module="models.slot_partition",
            method="SlotPartition",
        ),
        kwargs=dict(
            dataset=_get_dataset_config(),
            batch_size=4096,
            condition_on_identity=True,
            time_noise_sigma=0.1,
            position_noise_sigma=0.05,
            n_slots=2,
            include_empty=True,
            slot_prioritization=False,
            hidden_sizes=(512, 1024, 512),
        ),
    )
    return config


def get_config():
    config = {
        "constructor": dict(
            module="trainer",
            method="Trainer",
        ),
        "kwargs": dict(
            model=_get_model_config(),
            training_steps=10000,
            start_optimizing_per_trial=1000,
            lr_embedding=0.003,
            lr_per_trial=0.003,
            stop_step=None,
        ),
    }
    return config
