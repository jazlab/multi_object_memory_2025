"""Trainer class for training models."""

import json
import logging
from pathlib import Path

import torch


class Trainer:
    """Trainer class for training models."""

    def __init__(
        self,
        model,
        log_dir,
        training_steps,
        lr=0.01,
        optimizer=torch.optim.SGD,
        grad_clip=1,
        num_log_steps=10,
    ):
        """Trainer constructor."""
        self._model = model
        self._log_dir = Path(log_dir)
        self._training_steps = training_steps
        self._optimizer = optimizer
        self._grad_clip = grad_clip
        self._scalar_eval_every = self._training_steps // num_log_steps
        self._optimizer = optimizer(self._model.parameters(), lr=lr)

    def __call__(self):
        """Run the training loop.

        Args:
            log_dir: Directory to save training logs and model snapshots.
        """
        logging.info("\nBeginning the training")

        # Run training loop
        train_losses = []
        test_losses = []
        for step in range(self._training_steps):
            self._optimizer.zero_grad()
            train_loss_terms = self._model.loss_terms()
            train_loss = sum(train_loss_terms.values())
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._grad_clip,
            )
            self._optimizer.step()
            train_loss_terms = {
                k: float(v.detach()) for k, v in train_loss_terms.items()
            }
            train_losses.append(train_loss_terms)

            if step % self._scalar_eval_every == 0:
                test_loss_terms = self._model.loss_terms(train=False)
                test_loss_terms = {
                    k: float(v.detach()) for k, v in test_loss_terms.items()
                }
                print(
                    f"    Step {step} / {self._training_steps}; "
                    f"Train: {train_loss_terms}; "
                    f"Test: {test_loss_terms}"
                )
                test_loss_terms["step"] = step
                test_losses.append(test_loss_terms)

        # Log and exit
        print(f"Saving to {self._log_dir}")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        json.dump(
            train_losses, open(f"{self._log_dir}/training_losses.json", "w")
        )
        json.dump(test_losses, open(f"{self._log_dir}/test_losses.json", "w"))
        self._model.log(self._log_dir)
