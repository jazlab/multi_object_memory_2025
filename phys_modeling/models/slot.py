"""Slot model."""

import itertools

import numpy as np
import torch
import trials_to_stimuli

from . import abstract_model, embedding_mlp


def _get_assignments(n_slots: int, n_objects: int):
    """Get all possible assignments for a given number of slots and objects.

    Args:
        n_slots: Number of slots.
        n_objects: Number of objects.

    Returns:
        assignments: Numpy array of shape [n_assignments, n_objects, n_slots].
            Each element is a boolean assignment of objects to slots.
    """
    if n_objects > n_slots:
        to_product = n_slots * [np.arange(n_objects)]
        obj_per_slot = itertools.product(*to_product)
        obj_per_slot = [a for a in obj_per_slot if len(set(a)) == len(a)]
        assignments = []
        for x in obj_per_slot:
            a = np.zeros((n_objects, n_slots))
            for slot_index, obj_index in enumerate(x):
                a[obj_index, slot_index] = 1
            assignments.append(a)
    else:
        slot_per_obj = itertools.permutations(np.arange(n_slots), n_objects)
        assignments = []
        for x in slot_per_obj:
            a = np.zeros((n_objects, n_slots))
            for obj_index, slot_index in enumerate(x):
                a[obj_index, slot_index] = 1
            assignments.append(a)

    return np.array(assignments)


def _get_assignment_per_trial(assignments: np.ndarray, n_objects: int):
    """Get assignment per trial.

    Args:
        assignments: Numpy array of shape [n_assignments, n_objects, n_slots].
            Each element is a boolean assignment of objects to slots.
        n_objects: Number of objects in the trial.

    Returns:
        a: Numpy array of shape [n_assignments, n_objects, n_slots].
            Each element is a boolean assignment of objects to slots, with
            the last slots set to 0 if there are fewer objects than slots.
    """
    a = np.copy(assignments)
    a[:, n_objects:] = 0

    # Remove assignments that throw away objects
    n_objects_kept = np.sum(a, axis=(1, 2))
    n_slots = assignments.shape[2]
    min_kept = min(n_objects, n_slots)
    a[n_objects_kept < min_kept, :, :] = 0
    return a


def _valid_assignments(
    assignments: np.ndarray,
    n_objects: int,
    slot_prioritization: bool,
) -> np.ndarray:
    """Check if assignments are valid.

    Args:
        assignments: Numpy array of shape [n_assignments, n_objects, n_slots].
            Each element is a boolean assignment of objects to slots.
        n_objects: Number of objects in the trial.
        slot_prioritization: Boolean indicating whether to prioritize slots.

    Returns:
        valid: Numpy array of shape [n_assignments,] indicating whether each
            assignment is valid. An assignment is valid if it has at least one
            object in each slot, and if slot_prioritization is True, it has at
            least one object in the first n_objects slots.
    """
    valid = np.ones(len(assignments))

    # Remove invalid assignments if slot_prioritization
    if slot_prioritization:
        for i, a in enumerate(assignments):
            occupied_slots = np.sum(a, axis=0)
            if np.any(occupied_slots[:n_objects] == 0):
                valid[i] = 0.0

    # Remove zero assignments
    for i, a in enumerate(assignments):
        if np.sum(a) == 0:
            valid[i] = 0

    # Remove duplicate assignments
    for i, a in enumerate(assignments[1:]):
        duplicate = any(
            [np.array_equal(a, other) for other in assignments[: i + 1]]
        )
        if duplicate:
            valid[i + 1] = 0.0

    return valid


class Slot(abstract_model.AbstractModel):

    def __init__(
        self,
        dataset,
        batch_size: int,
        condition_on_identity: bool = True,
        time_noise_sigma: float = 0.2,
        position_noise_sigma: float = 0.2,
        n_slots: int = 2,
        include_empty: bool = True,
        slot_prioritization: bool = True,
        **embedding_mlp_kwargs
    ):
        """Constructor.

        Args:
            dataset: Dataset object containing neural data and stimuli. Must be
                an instance of phys_modeling.training.dataset.Dataset.
            batch_size: Size of the training batch.
            condition_on_identity: Boolean indicating whether to condition on
                identity in the embedding MLP.
            time_noise_sigma: Standard deviation of noise for time input.
            position_noise_sigma: Standard deviation of noise for position
                input.
            n_slots: Number of slots to use in the model.
            include_empty: Whether to include an empty slot in the model.
            slot_prioritization: Whether to prioritize slots in the model.
            embedding_mlp_kwargs: Dictionary of keyword arguments for the
                embedding MLP. If None, defaults will be used.
        """
        super(Slot, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            time_noise_sigma=time_noise_sigma,
            position_noise_sigma=position_noise_sigma,
            **embedding_mlp_kwargs,
        )
        self._n_slots = n_slots
        self._include_empty = include_empty
        self._slot_prioritization = slot_prioritization

        # Get number of objects per trial [n_trials]
        n_objects_per_trial = np.sum(self.dataset.valid_objects, axis=1)

        # Get assignments per trial [n_trials, n_assignments, max_n_objects, n_slots]
        self._assignments = _get_assignments(
            self._n_slots, self.dataset.max_n_objects
        )
        self._n_assignments = len(self._assignments)
        self._assignments_per_trial = np.array(
            [
                _get_assignment_per_trial(self._assignments, n)
                for n in n_objects_per_trial
            ],
            dtype=np.float32,
        )
        self._assignments_per_trial_torch = torch.from_numpy(
            self._assignments_per_trial
        )

        # Get valid assignments [n_trials, n_assignments]
        self._valid_assignments = np.array(
            [
                _valid_assignments(a, n, slot_prioritization)
                for a, n in zip(
                    self._assignments_per_trial, n_objects_per_trial
                )
            ],
            dtype=np.float32,
        )
        self._valid_assignments_torch = torch.from_numpy(
            self._valid_assignments
        )

        # Get attention per assignment
        attn_init = torch.zeros(self.dataset.n_trials, self._n_assignments)
        self._attention_assignments = torch.nn.Parameter(
            attn_init, requires_grad=True
        )

        # Construct embedding MLP
        self.embedding_mlps = [
            embedding_mlp.EmbeddingMLP(
                n_units=self.dataset.n_units,
                condition_on_position=True,
                condition_on_identity=condition_on_identity,
                **embedding_mlp_kwargs,
            )
            for _ in range(self._n_slots)
        ]

    def forward(
        self, train, batch_size=None, noise: bool = True
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            train: Boolean indicating whether to use training data or test data.
            batch_size: Size of the batch to use for training. If None, uses the
                default batch size from the dataset.
            noise: Boolean indicating whether to add noise to time and position.

        Returns:
            pred_per_assignment: Tensor of shape
                [batch_size, n_assignments, n_units].
            neural: Tensor of shape [batch_size, n_units].
            neural_finite: Tensor of shape [batch_size, n_units].
        """
        # Get a batch of training data of shape
        # [batch_size, n_objects, n_features]
        data = self.dataset.get_batch(train=train, batch_size=batch_size)
        trial = data["trial"]
        time = data["time"]
        positions = data["positions"] - 0.5
        identities = data["identities"]
        neural = data["neural"]
        neural_finite = data["neural_finite"]

        # Add noise to time and position if needed
        if noise:
            time += torch.randn_like(time) * self._time_noise_sigma
            positions += (
                torch.randn_like(positions) * self._position_noise_sigma
            )

        # If self._include_empty, add an empty slot
        num_objects = self._dataset.max_n_objects
        if self._include_empty:
            empty_positions_shape = list(positions.shape)
            empty_positions_shape[-2] = 1
            empty_identities_shape = list(identities.shape)
            empty_identities_shape[-2] = 1
            positions = torch.cat(
                [positions, torch.zeros(empty_positions_shape)], dim=-2
            )
            identities = torch.cat(
                [identities, torch.zeros(empty_identities_shape)], dim=-2
            )
            num_objects += 1

        # [batch_size, n_objects, n_slots, n_units]
        reps = [1] * len(time.shape) + [num_objects]
        time_tiled = time.unsqueeze(-1).repeat(reps)
        embeddings = torch.stack(
            [
                network(
                    time=time_tiled,
                    position=positions,
                    identity=identities,
                )
                for network in self.embedding_mlps
            ],
            dim=-2,
        )

        # Get assignments [batch_size, n_assignments, n_objects, n_slots]
        assignments = self._assignments_per_trial_torch[trial]

        # Add empty slot if needed
        if self._include_empty:
            is_empty = torch.sum(assignments, axis=-2) < 1
            is_empty.unsqueeze_(-2)
            assignments = torch.cat([assignments, is_empty], dim=-2)

        # Make shape to view assignments and embeddings
        shape_until_objects = embeddings.shape[:-3]
        assignments_shape = shape_until_objects + (
            self._n_assignments,
            num_objects,
            self._n_slots,
            1,
        )
        embeddings_shape = shape_until_objects + (
            1,
            num_objects,
            self._n_slots,
            self.dataset.n_units,
        )

        # [batch_size, n_assignments, n_objects, n_slots, n_units]
        embedding_per_assignment = assignments.view(
            assignments_shape
        ) * embeddings.view(embeddings_shape)

        # [batch_size, n_assignments, n_units]
        pred_per_assignment = torch.sum(
            embedding_per_assignment, axis=(-3, -2)
        )

        # Add bias and apply nonlinearity
        bias = self.bias_mlp(time=time)  # [batch_size, n_units]
        pred_per_assignment = self._softplus(
            pred_per_assignment + bias.unsqueeze(-2)
        )

        # [batch_size, n_assignments]
        attention = self.get_attention_assignments()[trial]

        return pred_per_assignment, attention, neural, neural_finite

    def prediction_per_unit_numpy(self, train) -> np.ndarray:
        """Get predictions per unit as a numpy array.

        Returns:
            prediction: Numpy array of shape [n_trials, n_timesteps, n_units]
                containing the predicted firing rates for each unit across all
                trials and timesteps.
            neural: Numpy array of shape [n_trials, n_timesteps, n_units].
            neural_finite: Numpy array of shape [n_trials, n_timesteps, n_units].
        """
        # prediction has shape [n_trials, n_timesteps, n_assignments, n_units]
        # neural and neural_finite have shape [n_trials, n_timesteps, n_units]
        pred_per_assignment, _, neural, neural_finite = self.forward(
            train=train, batch_size=None, noise=False
        )

        # [n_trials, n_assignments]
        attention_assignments = self.get_attention_assignments()

        # [n_trials]
        max_attn_assignments = torch.argmax(attention_assignments, axis=1)

        # Index pred_per_assignment with max_attn_assignments
        # [n_trials, n_timesteps, n_units]
        prediction = pred_per_assignment[
            torch.arange(self.dataset.n_trials), :, max_attn_assignments
        ]

        # Detach and convert to numpy
        prediction = prediction.detach().cpu().numpy()
        neural = neural.detach().cpu().numpy()
        neural_finite = neural_finite.detach().cpu().numpy()

        return prediction, neural, neural_finite

    def loss(self) -> torch.Tensor:
        """Compute the loss for the model.

        Returns:
            A tensor containing the loss value.
        """
        # prediction has shape [batch_size, n_assignments, n_units]
        # attention has shape [batch_size, n_assignments]
        # neural and neural_finite have shape [batch_size, n_units]
        pred_per_assignment, attention, neural, neural_finite = self.forward(
            train=True, batch_size=self._batch_size, noise=True
        )

        # Get loss matrix of shape
        # [batch_size, n_assignments, n_units]
        negative_log_likelihood = pred_per_assignment - neural[
            :, None
        ] * torch.log(pred_per_assignment)
        loss_matrix = neural_finite[:, None] * negative_log_likelihood

        # Sum loss over units
        loss_matrix = torch.sum(loss_matrix, axis=2)

        # Weight loss matrix by attention and normalize
        loss = torch.sum(loss_matrix * attention) / torch.sum(neural_finite)

        return loss

    def _cache(self, write_dir: str):
        """Cache attention assignments and related data to disk.

        Args:
            write_dir: Directory to write the cached data.
        """
        write_dir.mkdir(parents=True, exist_ok=True)
        attention_assignments = self.get_attention_assignments()
        max_attn_assignments = torch.argmax(attention_assignments, axis=1)
        # [n_trials, n_objects, n_slots]
        assignments = self._assignments[max_attn_assignments.detach().numpy()]
        # [n_trials, n_slots, n_objects]
        assignments = np.transpose(assignments, (0, 2, 1))
        attention_path = write_dir / "attention.npy"
        np.save(attention_path, assignments)
        return

    def get_attention_assignments(self) -> torch.Tensor:
        """Get attention assignments.

        Returns:
            A tensor of shape [n_trials, n_assignments] containing the attention
            assignments for each trial and assignment.
        """
        exp_attn = torch.exp(self._attention_assignments)
        exp_attn_masked = exp_attn * self._valid_assignments_torch
        softmax_attn = exp_attn_masked / torch.sum(
            exp_attn_masked, axis=1, keepdim=True
        )
        return softmax_attn

    def _embedding_parameters(self) -> list[torch.Tensor]:
        """Get embedding parameters."""
        params = list(
            itertools.chain.from_iterable(
                [network.parameters() for network in self.embedding_mlps]
            )
        )
        return params

    @property
    def per_trial_parameters(self) -> list[torch.Tensor]:
        """Get per-trial parameters."""
        return [self._attention_assignments]
