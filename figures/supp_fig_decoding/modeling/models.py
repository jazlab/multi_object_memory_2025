"""Models."""

import abc

import numpy as np
import torch


class AbstractLinearModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for neural data modeling."""

    def __init__(self, dataset, batch_size, loss_fn=torch.nn.MSELoss):
        """Constructor."""
        # Set random seed
        np.random.seed(dataset.random_seed)
        torch.manual_seed(dataset.random_seed)

        # Initialize the parent class and register variables
        super(AbstractLinearModel, self).__init__()
        self._dataset = dataset
        self._batch_size = batch_size

        # Binary cross entropy loss for linear model
        self._epsilon = 1e-8
        self._loss_fn = loss_fn(reduction="none")

    @abc.abstractmethod
    def forward(self):
        """Forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def _loss_terms(self, data_batch):
        """Get the model's loss on a data batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, log_dir):
        """Log the model's parameters."""
        raise NotImplementedError

    def get_batch(self, train: bool = True):
        """Get a batch of data from the dataset."""
        if train:
            data_batch = self.dataset.get_batch(
                batch_size=self._batch_size,
                train=True,
            )
        else:
            data_batch = self.dataset.get_batch(
                batch_size=None,
                train=False,
            )
        return data_batch

    def loss_terms(self, train=True):
        """Compute the loss_terms for the model."""
        data_batch = self.get_batch(train=train)
        return self._loss_terms(data_batch)

    @property
    def dataset(self):
        """Get the dataset."""
        return self._dataset


class LinearModel1Object(AbstractLinearModel):
    """Linear model for 1-object decoding."""

    def __init__(
        self,
        dataset,
        batch_size=512,
        l1_weight=0.003,
        loss_fn=torch.nn.MSELoss,
    ):
        """Constructor."""
        super(LinearModel1Object, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            loss_fn=loss_fn,
        )
        self._l1_weight = l1_weight

        # Create model parameters
        self._linear = torch.nn.Linear(
            in_features=self._dataset.n_units,
            out_features=self._dataset.label_dim,
            bias=True,
        )

    def forward(self, x):
        """Forward pass of the model."""
        x = torch.sigmoid(self._linear(x))
        return x

    def _loss_terms(self, data_batch):
        """Get the model's loss on a data batch."""
        # Compute loss
        inputs = data_batch["neural"]
        targets = data_batch["labels"]
        outputs = self.forward(inputs)
        loss = torch.mean(self._loss_fn(outputs, targets))
        loss_terms = {"classification_loss": loss}

        # If needed, add L1 regularization loss
        if self._l1_weight > 0:
            l1_loss = 0.0
            for param in self._linear.parameters():
                l1_loss += torch.sum(torch.abs(param))
            l1_loss = self._l1_weight * l1_loss
            loss_terms["l1_loss"] = l1_loss

        return loss_terms

    def log(self, log_dir):
        """Log the model parameters."""
        # Log model linear weights
        linear_weights = self._linear.weight.detach().cpu().numpy()
        linear_bias = self._linear.bias.detach().cpu().numpy()
        np.save(log_dir / "linear_weights.npy", linear_weights)
        np.save(log_dir / "linear_bias.npy", linear_bias)

        # Save train and test targets and outputs
        test_data_batch = self.get_batch(train=False)
        test_inputs = test_data_batch["neural"]
        test_targets = test_data_batch["labels"]
        test_outputs = self.forward(test_inputs).detach().cpu().numpy()
        test_targets = test_targets.detach().cpu().numpy()
        np.save(log_dir / "test_outputs.npy", test_outputs)
        np.save(log_dir / "test_targets.npy", test_targets)


class LinearModel2Objects(AbstractLinearModel):
    """Linear model for 2-object decoding."""

    def __init__(
        self,
        dataset,
        batch_size=512,
        l1_weight=0.001,
        loss_fn=torch.nn.MSELoss,
    ):
        """Constructor."""
        super(LinearModel2Objects, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            loss_fn=loss_fn,
        )
        self._l1_weight = l1_weight

        # Create model parameters
        self._linear_0 = torch.nn.Linear(
            in_features=self._dataset.n_units,
            out_features=self._dataset.label_dim,
            bias=True,
        )
        self._linear_1 = torch.nn.Linear(
            in_features=self._dataset.n_units,
            out_features=self._dataset.label_dim,
            bias=True,
        )
        self._logits = torch.nn.Parameter(
            torch.zeros(self._dataset.n_train_trials)
        )

    def forward(self, x):
        """Forward pass of the model."""
        x_0 = torch.sigmoid(self._linear_0(x))
        x_1 = torch.sigmoid(self._linear_1(x))
        return x_0, x_1

    def _loss_terms(self, data_batch):
        """Get the model's loss on a data batch."""
        # Compute loss for each target and decoder
        inputs = data_batch["neural"]
        targets = data_batch["labels"]
        indices = data_batch["indices"]
        outputs_0, outputs_1 = self.forward(inputs)
        loss_0_0 = torch.mean(self._loss_fn(outputs_0, targets[:, 0]), dim=1)
        loss_1_0 = torch.mean(self._loss_fn(outputs_1, targets[:, 0]), dim=1)
        loss_0_1 = torch.mean(self._loss_fn(outputs_0, targets[:, 1]), dim=1)
        loss_1_1 = torch.mean(self._loss_fn(outputs_1, targets[:, 1]), dim=1)
        loss_cross = loss_1_0 + loss_0_1
        loss_same = loss_0_0 + loss_1_1

        # Compute loss based on assignments
        if indices is None:
            # Pick the minimum loss assignment if no indices are provided
            classification_loss = torch.mean(
                torch.minimum(loss_cross, loss_same)
            )
        else:
            # Use the logits to select the loss assignment
            prob_same = torch.sigmoid(self._logits[indices])
            prob_cross = 1.0 - prob_same
            classification_loss = torch.mean(
                prob_same * loss_same + prob_cross * loss_cross
            )
        loss_terms = {"classification_loss": classification_loss}

        # If needed, add L1 regularization loss
        if self._l1_weight > 0:
            l1_loss = 0.0
            for param in self._linear_0.parameters():
                l1_loss += torch.sum(torch.abs(param))
            for param in self._linear_1.parameters():
                l1_loss += torch.sum(torch.abs(param))
            l1_loss = self._l1_weight * l1_loss
            loss_terms["l1_loss"] = l1_loss

        return loss_terms

    def log(self, log_dir):
        """Log the model parameters."""
        # Log model linear weights
        linear_0_weights = self._linear_0.weight.detach().cpu().numpy()
        linear_0_bias = self._linear_0.bias.detach().cpu().numpy()
        np.save(log_dir / "linear_0_weights.npy", linear_0_weights)
        np.save(log_dir / "linear_0_bias.npy", linear_0_bias)
        linear_1_weights = self._linear_1.weight.detach().cpu().numpy()
        linear_1_bias = self._linear_1.bias.detach().cpu().numpy()
        np.save(log_dir / "linear_1_weights.npy", linear_1_weights)
        np.save(log_dir / "linear_1_bias.npy", linear_1_bias)

        # Log model logits
        logits = self._logits.detach().cpu().numpy()
        logits = torch.sigmoid(self._logits).detach().cpu().numpy()
        np.save(log_dir / "logits.npy", logits)

        # Save train and test targets and outputs
        test_data_batch = self.get_batch(train=False)
        test_inputs = test_data_batch["neural"]
        test_targets = test_data_batch["labels"]
        test_outputs_0, test_outputs_1 = self.forward(test_inputs)
        test_outputs_0 = test_outputs_0.detach().cpu().numpy()
        test_outputs_1 = test_outputs_1.detach().cpu().numpy()
        test_targets = test_targets.detach().cpu().numpy()
        np.save(log_dir / "test_outputs_0.npy", test_outputs_0)
        np.save(log_dir / "test_outputs_1.npy", test_outputs_1)
        np.save(log_dir / "test_targets.npy", test_targets)
