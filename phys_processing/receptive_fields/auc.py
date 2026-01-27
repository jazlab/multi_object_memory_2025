"""ROC analysis."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def _compute_auc(labels):
    """Compute area under the ROC curve."""
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("Labels must be a 2D array with two columns.")

    # Compute ROC curve
    tp = np.cumsum(labels[:, 0])
    fp = np.cumsum(labels[:, 1])
    fn = np.sum(labels[:, 0]) - tp
    tn = np.sum(labels[:, 1]) - fp
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    # Compute area under false positive vs true positive curve
    auc = np.trapz(tpr, fpr)

    return auc


def compute_statistics(samples, n_bootstrap=5000, plot=False):
    # Set random seed for reproducibility
    np.random.seed(0)

    if len(samples) != 2:
        raise ValueError("Only two distributions are supported.")

    # Make array of values and labels
    values = np.concatenate(samples)
    labels = np.zeros((len(values), 2), dtype=int)
    labels[: len(samples[0]), 0] = 1
    labels[len(samples[0]) :, 1] = 1

    # Sort values and labels
    idx = np.argsort(values)
    values = values[idx]
    labels = labels[idx]

    # Compute AUC of the ROC curve
    auc = _compute_auc(labels)

    # Run bootstraps to compute confidence intervals
    bootstrap_auc = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(len(values), len(values), replace=True)
        bootstrap_auc[i] = _compute_auc(labels[idx])

    # Compute quantile of empirical AUC in auc distribution
    p_value = np.mean(bootstrap_auc >= auc)

    # Plot bootstrap distribution and empirical AUC if necessary
    if plot:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(bootstrap_auc, ax=axes[0])
        axes[0].axvline(auc, color="r")
        axes[0].set_xlabel("AUC")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"p-value: {p_value:.3f} | AUC: {auc:.3f}")

        # Plot histogram of distributions using seaborn
        df = pd.DataFrame(
            {
                "value": np.concatenate(samples),
                "label": np.concatenate(
                    [[i] * len(s) for i, s in enumerate(samples)]
                ),
            }
        )
        sns.histplot(df, x="value", hue="label", ax=axes[1])

    # Normalize
    p_value = 1 - p_value if p_value > 0.5 else p_value
    auc = 1 - auc if auc < 0.5 else auc

    return p_value, auc


def main():
    """Main function."""

    distributions = [
        dict(mu=0, scale=1, n=100),
        dict(mu=-0.5, scale=1, n=80),
    ]
    samples = [
        np.random.normal(d["mu"], d["scale"], d["n"]) for d in distributions
    ]
    compute_statistics(samples, plot=True)

    plt.show()


if __name__ == "__main__":
    main()
