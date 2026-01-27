"""Run model training."""

from pathlib import Path

import dataset as dataset_lib
import labelers as labelers_lib
import models as models_lib
import torch
import trainer as trainer_lib

_PHASES = ["stimulus", "delay"]
_BRAIN_AREA_SUBJECT_SESSIONS = [
    (None, "Perle", "2022-06-01"),
    (None, "Perle", "2022-06-04"),
    (None, "Perle", "2022-05-31"),
    ("DMFC", "Perle", "2022-06-01"),
    ("DMFC", "Perle", "2022-06-04"),
    ("DMFC", "Perle", "2022-05-31"),
    ("FEF", "Perle", "2022-06-01"),
    ("FEF", "Perle", "2022-06-04"),
    ("FEF", "Perle", "2022-05-31"),
]
_SHUFFLE_LABELS = [False, True]
_FEATURES = [labelers_lib.Features.POSITION, labelers_lib.Features.IDENTITY]
_RANDOM_SEEDS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
]


def run(
    brain_area,
    subject,
    session,
    phase,
    feature,
    shuffle_labels,
    random_seed,
):
    dataset = dataset_lib.Dataset(
        num_objects=1,
        subject=subject,
        session=session,
        phase=phase,
        shuffle_labels=shuffle_labels,
        random_seed=random_seed,
        labeler=labelers_lib.Triangle(feature=feature),
        brain_area=brain_area,
    )
    model = models_lib.LinearModel1Object(
        dataset=dataset,
        l1_weight=0.003,
        batch_size=512,
        loss_fn=torch.nn.BCELoss,
    )
    feature_str = f"feature={feature.name.lower()}"
    shuffle_labels_str = f"shuffle_labels={int(shuffle_labels)}"
    suffix = (
        f"{feature_str}/{brain_area}/{subject}_{session}/{phase}/"
        f"{shuffle_labels_str}/{random_seed}"
    )
    log_dir = Path(
        f"../../../cache/figures/supp_fig_decoding/1_triangle/{suffix}"
    )
    if log_dir.exists():
        print(f"Log dir {log_dir} exists; skipping...")
        return

    trainer = trainer_lib.Trainer(
        model=model,
        training_steps=10000,
        lr=0.01,
        optimizer=torch.optim.SGD,
        log_dir=log_dir,
    )
    trainer()


def main():
    """Main function to run model training."""
    for brain_area, subject, session in _BRAIN_AREA_SUBJECT_SESSIONS:
        for phase in _PHASES:
            for feature in _FEATURES:
                for shuffle_labels in _SHUFFLE_LABELS:
                    for random_seed in _RANDOM_SEEDS:
                        run(
                            brain_area=brain_area,
                            subject=subject,
                            session=session,
                            phase=phase,
                            feature=feature,
                            shuffle_labels=shuffle_labels,
                            random_seed=random_seed,
                        )


if __name__ == "__main__":
    main()
