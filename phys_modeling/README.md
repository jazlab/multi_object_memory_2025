# Modeling

This directory contains model implementations and code for training models.

## Directory Structure

There are the following sub-directories:
- `training`: This contains the script to actually train a model, and associated
  tools
- `configs`: This contains config files specifying models to train.
- `config_utils`: This contains code for parsing and modifying configs.
- `models`: This contains the model implementations.
- `synthetic_datasets`: This contains code for generating synthetic datasets.
- `sweeps`: This contains scripts for launching cluster jobs to train many
  models.
- `monitor_sweeps`: This contains notebooks for checking the progress of sweeps.

## Getting Started

Before training models, first make sure you are working in a virtual environment
with the dependencies in `../requirements.txt` installed (see `../README` for
details).

Second, be sure to have cached the spiking data. To cache the spiking data in a
format used by the models, download it from our OSF dataset by going to
[https://osf.io/vyw49](https://osf.io/vyw49) and downloading
`data_for_modeling.zip`. Unzip the result to find a folder `spikes_per_trial`.
Move this folder to `../cache/phys_processing spikes_to_trials/`, next to the
existing cached folders there (which should include `delay_phase_firing_rates/`
and `stimulus_phase_firing_rates/`). Alternatively, if you would like to
generate the spikes per trial data yourself, you can run
`../phys_processing/spikes_to_trials/run_spikes_per_trial.py`.

To train a model navigate to `./training` and run
```
$ python run_training.py
```

This will train a model, by default the Gain model on the Triangle dataset. As
it trains, it will print progress to the terminal (train and test log
likelihoods). It takes about 20 minutes to train on a laptop. To run other
models or other datasets, change the `config` flag in `run_training.py`. See
`configs/` for available config files.
