# Repo_Autoencoder

In the folder 'autoencoder_bayer' a new folder called 'generated-data' needs to be added before starting the code.
When using tensorboard, a second folder called 'runs' will emerge here (automatically).
Also on the same stage as the 'autoencoder_bayer', 'model', github and readme a new folder called 'models' need to be added.

The code can be started in a conda environment with the following attributes:
python train.py --signal-type=vel -bs=128 -vt=2 -hz=100 --slice-time-windows=2s-overlap

If you just want to load certain datasets, have a look in 
'autoencoder_bayer\data\__init__.py'
and comment those you don't need out.
