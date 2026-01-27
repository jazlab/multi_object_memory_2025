"""Config for 1-object decoding."""

import dataset as dataset_lib
import labelers as labelers_lib
import models as models_lib
import torch
import trainer as trainer_lib


def _get_dataset():
    dataset = dataset_lib.Dataset(
        num_objects=2,
        subject="Perle",
        session="2022-06-10",
        phase="delay",
        labeler=labelers_lib.Ring(feature=labelers_lib.Features.POSITION),
        mode=dataset_lib.Mode.NEURAL,
    )
    return dataset


def _get_model():
    model = models_lib.LinearModel2Objects(
        dataset=_get_dataset(),
        l1_weight=0.003,
        batch_size=512,
    )
    return model


def get_trainer():
    model = _get_model()
    trainer = trainer_lib.Trainer(
        model=model,
        training_steps=10000,
        lr=0.001,
        optimizer=torch.optim.Adam,
        log_dir="../../cache/figures/supp_fig_decoding/tmp_v0/2_objects_ring",
    )
    return trainer
