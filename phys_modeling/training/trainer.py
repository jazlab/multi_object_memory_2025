"""Trainer class for training models."""

import json
import logging
from pathlib import Path

import numpy as np
import torch


def get_eval_stats(model: torch.nn.Module) -> dict:
    """Get evaluation statistics for the model.

    Args:
        model: The trained model to evaluate.

    Returns:
        A dictionary containing evaluation statistics.
    """
    # log_likelihood and target_finite both have shape
    # (num_trials, num_neurons, num_timesteps)
    log_likelihood, _, target_finite = model.goodness_of_fit_array(train=False)
    log_likelihood_finite = log_likelihood * target_finite

    # Now compute log_likelihood per trial per unit and number of trials per
    # unit, taking into consideration the trials where the unit was observed
    # (i.e., finite_per_trial > 0)
    ll_per_unit = []
    trials_per_unit = []
    ll_per_trial = np.sum(log_likelihood_finite, axis=2)
    finite_per_trial = target_finite[:, :, 0]
    n_trials, n_units = ll_per_trial.shape
    for unit_index in range(n_units):
        ll_per_unit_trial = []
        trials_per_unit_trial = []
        for trial_index in range(n_trials):
            if finite_per_trial[trial_index, unit_index] > 0:
                ll_per_unit_trial.append(ll_per_trial[trial_index, unit_index])
                trials_per_unit_trial.append(trial_index)
        ll_per_unit.append(ll_per_unit_trial)
        trials_per_unit.append(trials_per_unit_trial)

    # Compile and return the results
    eval_stats = {
        "ll_per_unit": ll_per_unit,
        "trials_per_unit": trials_per_unit,
    }

    return eval_stats


class Trainer:
    """Trainer class for training models."""

    def __init__(
        self,
        model: torch.nn.Module,
        training_steps: int,
        lr_embedding: float = 0.003,
        lr_per_trial: float = 0.003,
        start_optimizing_per_trial: int = 0,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        grad_clip: float = 1,
        num_log_steps: int = 100,
        stop_step: int = None,
    ):
        """Trainer constructor.

        Args:
            model: The model to train.
            training_steps: Number of training steps to run.
            lr_embedding: Learning rate for the embedding parameters.
            lr_per_trial: Learning rate for the per-trial parameters.
            optimizer: Optimizer class to use for training.
            grad_clip: Gradient clipping value.
            num_log_steps: Number of steps between logging training progress.
            stop_step: If provided, log at this specific step and stop training.
        """
        self._model = model
        self._training_steps = training_steps
        self._optimizer = optimizer
        self._grad_clip = grad_clip
        self._scalar_eval_every = self._training_steps // num_log_steps
        self._start_optimizing_per_trial = start_optimizing_per_trial
        self._stop_step = stop_step

        # Create optimizer
        self._optimizer_embedding = optimizer(
            self._model.embedding_parameters, lr=lr_embedding
        )
        self._optimizer_per_trial = optimizer(
            self._model.per_trial_parameters, lr=lr_per_trial
        )

    def __call__(self, log_dir: str):
        """Run the training loop.

        Args:
            log_dir: Directory to save training logs and model snapshots.
        """
        logging.info("\nBeginning the training")
        log_dir = Path(log_dir)

        # Run training loop
        losses = []
        train_metrics = []
        test_metrics = []
        stop_metric = []
        for step in range(self._training_steps):
            self._optimizer_embedding.zero_grad()
            self._optimizer_per_trial.zero_grad()
            loss = self._model.loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.embedding_parameters
                + self._model.per_trial_parameters,
                self._grad_clip,
            )
            self._optimizer_embedding.step()
            if step >= self._start_optimizing_per_trial:
                self._optimizer_per_trial.step()
            losses.append(float(loss.detach()))

            if step % self._scalar_eval_every == 0:
                step_train_metrics = self._model.goodness_of_fit(train=True)
                step_test_metrics = self._model.goodness_of_fit(train=False)
                step_train_metrics["step"] = step
                step_test_metrics["step"] = step
                train_metrics.append(step_train_metrics)
                test_metrics.append(step_test_metrics)
                logging.info(
                    f"    Step {step} / {self._training_steps}; "
                    f"Loss = {losses[-1]:.7f}; "
                    f"Train LL = {step_train_metrics['mean_log_likelihood']:.7f}; "
                    f"Test LL = {step_test_metrics['mean_log_likelihood']:.7f}"
                )

                # Detect stopping
                if step >= self._start_optimizing_per_trial:
                    test_ll = step_test_metrics["mean_log_likelihood"]
                    new_high = all([test_ll >= x for x in stop_metric])
                    stop_metric.append(test_ll)
                    if new_high:
                        # Save new snapshot
                        log_dir_best = log_dir / "best_step"
                        log_dir_best.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            self._model.state_dict(),
                            log_dir_best / "model_snapshot.pth",
                        )
                        json.dump(step, open(f"{log_dir_best}/step.json", "w"))
                        self._model.cache(write_dir=log_dir_best)

            if self._stop_step is not None and step == self._stop_step:
                log_dir_stop = log_dir / "stop_step"
                log_dir_stop.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self._model.state_dict(),
                    log_dir_stop / "model_snapshot.pth",
                )
                json.dump(step, open(f"{log_dir_stop}/step.json", "w"))
                self._model.cache(write_dir=log_dir_stop)
                eval_stats = get_eval_stats(self._model)
                json.dump(
                    eval_stats, open(f"{log_dir_stop}/eval_stats.json", "w")
                )
                logging.info(f"Stopping training at step {self._stop_step}")
                break

        # Save units dataframe and training losses
        self._model.dataset.df_units.to_csv(f"{log_dir}/df_units.csv")
        json.dump(losses, open(f"{log_dir}/training_losses.json", "w"))
        json.dump(train_metrics, open(f"{log_dir}/train_metrics.json", "w"))
        json.dump(test_metrics, open(f"{log_dir}/test_metrics.json", "w"))
