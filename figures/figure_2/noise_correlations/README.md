# Noise correlation analyses

This directory contains tools for computing noise correlations in the triangle
dataset and notebooks for reproducing plots of noise correlations.

The noise correlation data is in this GitHub repo, so if you have cloned the
repo including the `cache/` directory, then you do not need to re-compute them.
However, if you would like to re-compute the noise correlations, run
`$ python run_compute_triangle.py`. This will compute the noise correlations for
all triangle dataset sessions and save them to
`../../../cache/figures/figure_2/noise_correlations_triangle`.

To generate plots of the relationship between noise and signal correlations for
all pairs of units, run the notebook `plot_noise_vs_signal_correlation.ipynb`.

To generate plots noise correlations for example units, run the notebook
`plot_example_units.ipynb`.
