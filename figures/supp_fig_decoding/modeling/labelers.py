"""Labels for datasets."""

import enum

import numpy as np
import pandas as pd


class Features(enum.Enum):
    POSITION = "position"
    IDENTITY = "identity"


class Triangle:
    def __init__(self, feature):
        self.feature = feature

    def __call__(self, dataset, row):
        """Convert a behavior dataframe row to a label."""
        if dataset.task != "triangle":
            raise ValueError("Dataset task must be triangle.")
        labels = []
        for i_object in range(3):
            if self.feature == Features.POSITION:
                location = row[f"object_{i_object}_location"]
                if pd.isna(location):
                    continue
                label = [0, 0, 0]
                label[int(location)] = 1
            elif self.feature == Features.IDENTITY:
                identity = row[f"object_{i_object}_id"]
                if pd.isna(identity):
                    continue
                label = dataset.IDENTITY_ONEHOT[identity]
            labels.append(label)

        if len(labels) != dataset.num_objects:
            return None

        return np.squeeze(labels)


class TriangleComplement:

    def __init__(self, feature):
        self.feature = feature

    def __call__(self, dataset, row):
        """Convert a behavior dataframe row to a label."""
        if dataset.task != "triangle":
            raise ValueError("Dataset task must be triangle.")
        if dataset.num_objects != 2:
            raise ValueError("Dataset num_objects must be 2.")
        labels = []
        for i_object in range(3):
            if self.feature == Features.POSITION:
                location = row[f"object_{i_object}_location"]
                if pd.isna(location):
                    continue
                label = [0, 0, 0]
                label[int(location)] = 1
            elif self.feature == Features.IDENTITY:
                identity = row[f"object_{i_object}_id"]
                if pd.isna(identity):
                    continue
                label = dataset.IDENTITY_ONEHOT[identity]
            labels.append(label)

        if len(labels) != 2:
            return None

        # Now get the complement label
        labels = 1 - np.sum(np.array(labels), axis=0)

        return labels


class Ring:

    def __init__(self, feature):
        self.feature = feature

    def __call__(self, dataset, row):
        """Convert a behavior dataframe row to a label."""
        if dataset.task != "ring":
            raise ValueError("Dataset task must be ring.")
        labels = []
        for i_object in range(2):
            if self.feature == Features.POSITION:
                pos_x = row[f"object_{i_object}_x"]
                pos_y = row[f"object_{i_object}_y"]
                if pd.isna(pos_x) or pd.isna(pos_y):
                    continue
                label = [2 * pos_x - 1, 2 * pos_y - 1]
            elif self.feature == Features.IDENTITY:
                identity = row[f"object_{i_object}_id"]
                label = dataset.IDENTITY_ONEHOT[identity]
            labels.append(label)

        if len(labels) != dataset.num_objects:
            return None

        return np.squeeze(labels)
