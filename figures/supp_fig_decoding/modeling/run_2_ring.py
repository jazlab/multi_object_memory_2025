"""Run model training."""

from pathlib import Path

import dataset as dataset_lib
import labelers as labelers_lib
import models as models_lib
import torch
import trainer as trainer_lib

_PHASES = ["stimulus", "delay"]
_SUBJECT_SESSIONS = [
    ("Perle", "2022-06-09"),
    ("Perle", "2022-06-10"),
    ("Perle", "2022-06-11"),
]
_SHUFFLE_LABELS = [False, True]
_MODES = [
    dataset_lib.Mode.NEURAL.value,
    dataset_lib.Mode.ORTHOGONAL.value,
    dataset_lib.Mode.RECEPTIVE_FIELDS.value,
]
_FEATURES = [labelers_lib.Features.POSITION]
_RANDOM_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def run(
    mode,
    subject,
    session,
    phase,
    feature,
    shuffle_labels,
    random_seed,
):
    dataset = dataset_lib.Dataset(
        num_objects=2,
        subject=subject,
        session=session,
        phase=phase,
        shuffle_labels=shuffle_labels,
        random_seed=random_seed,
        labeler=labelers_lib.Ring(feature=feature),
        mode=mode,
    )
    model = models_lib.LinearModel2Objects(
        dataset=dataset,
        l1_weight=0.003,
        batch_size=512,
    )
    feature_str = f"feature={feature.name.lower()}"
    shuffle_labels_str = f"shuffle_labels={int(shuffle_labels)}"
    suffix = (
        f"{mode}/{feature_str}/{subject}_{session}/{phase}/{shuffle_labels_str}/"
        f"{random_seed}"
    )
    log_dir = Path(f"../../../cache/figures/supp_fig_decoding/2_ring/{suffix}")
    if log_dir.exists():
        print(f"Log dir {log_dir} exists; skipping...")
        return

    trainer = trainer_lib.Trainer(
        model=model,
        training_steps=10000,
        lr=0.001,
        optimizer=torch.optim.Adam,
        log_dir=log_dir,
    )
    trainer()


def main():
    """Main function to run model training."""
    for mode in _MODES:
        for subject, session in _SUBJECT_SESSIONS:
            for phase in _PHASES:
                for feature in _FEATURES:
                    for shuffle_labels in _SHUFFLE_LABELS:
                        for random_seed in _RANDOM_SEEDS:
                            run(
                                subject=subject,
                                session=session,
                                phase=phase,
                                feature=feature,
                                shuffle_labels=shuffle_labels,
                                random_seed=random_seed,
                                mode=mode,
                            )


if __name__ == "__main__":
    main()
