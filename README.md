# Visualizing the hidden activity of artificial neural networks

## Mateusz Knapik, Jakub Czerski 2020

### Scenarios

To run exact scenario of analysis look at scenarios/scenarios.py and execute chosen one in main.py.
Keep in mind that scenarios with trained models need model saved to the file.

### Saved model and network history

The following format applies:

- for model: `model_<dataset_name>_<network_type>_<epochs>`
- for network history: `model_<dataset_name>_<network_type>_<epochs>_history.json`
- for network weights after each epoch during training: `model_<dataset_name>_<network_type>_<epochs>_<epoch>.hdf5`

### Saved transformed points (t-SNE)

Each t-SNE transformation saves transformed points (if and only if such a transformed points does not exist).
The following format applies: `points_<model_name>_<layer>_<epochs>.npy`

### SVHN dataset

These files should be placed in `data/` directory:

- <http://ufldl.stanford.edu/housenumbers/train_32x32.mat>
- <http://ufldl.stanford.edu/housenumbers/test_32x32.mat>

## Michał Kmicikiewicz, Paweł Kopel 2022

### Visualisations

[`vis_notebook.ipynb`](vis_notebook.ipynb) contains example "Inter-layer evolution", "Inter-epoch evolution",
"Neuron projections" and "Discriminative neuron map" visualisations of networks trained with MNIST and CIFAR datasets.
