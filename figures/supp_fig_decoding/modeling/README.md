# Decoding Models

This directory contains code for running decoding models.

## Getting Started

To train a decoding model, run
```
$ python run_training.py
```
Be sure to run this from a virtual environment with the dependencies in
`../requirements.txt` installed. Also be sure to have cached the spiking data
(see `../../phys_processing/spikes_to_trials/run_spikes_per_trial.py` for
details).

This will train a decoding model, by default on 1-object Triangle dataset
conditions. To decoding models on other datasets, change the `_CONFIG` flag at
the top of `run_training.py`.

## Running sweeps

The files `run_1_triangle.py`, `run_2_triangle_complement.py`,
`run_2_triangle.py`, and `run_2_ring.py` are scripts to run all the decoding
models shown in the paper. These cache the results to
`../../cache/figures/supp_fig_decoding/`, which the notebooks in `../analysis/`
use to generate plots. Note that we have provided this cache in our OSF dataset,
so you do not need to re-generate it to run the plotting notebooks.
