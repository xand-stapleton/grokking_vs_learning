# Grokking vs Learning: Information-Geometric Analysis

This repository contains code for analyzing neural network learning dynamics and grokking phenomena across different tasks using information-geometric measures.

## Overview

This project investigates the interplay between generalisation (learning) and sudden sharp increases in test accuracy (grokking) in neural networks. It computes and analyses information-geometric quantities such as Fisher Information Matrices (FIMs) to characterise learning dynamics across multiple tasks:

- **Ising Spin Models**: Physics-inspired classification task on binary sequences with learned structure
- **Modular Arithmetic (ModAdd)**: Symbolic task on modular addition
- **MNIST**: Handwritten digit classification

## Project Structure

```
.
├── InformationGeometricAnalysis.py    # Main analysis script for computing information-geometric measures
├── ising/                              # Ising spin model experiments
│   ├── run_ising.py                   # Training script for Ising tasks
│   ├── data_objects.py                # Data loading and handling
│   ├── dynamic_plot.py                # Real-time plotting during training
│   ├── configs/                       # Configuration files for different training regimes
│   ├── datasets/                      # Dataset creation and utilities
│   ├── models/                        # Model architectures and training utilities
│   ├── plotting/                      # Analysis and visualization scripts
│   ├── tools/                         # Configuration parsing and training tools
│   └── helper_functions/              # Utility functions for data and training
│
├── mod_add/                            # Modular arithmetic experiments
│   ├── run_mod_add.py                 # Training script for ModAdd tasks
│   ├── examine_runs.py                # Script to analyze completed runs
│   ├── configs/                       # Configuration files
│   ├── datasets/                      # Dataset creation utilities
│   ├── models/                        # Model architectures
│   ├── plotting/                      # Visualization scripts
│   ├── tools/                         # Utilities and configuration parsing
│   └── helper_functions/              # Helper functions
│
├── mnist/                              # MNIST experiments
│   ├── run_script_config_seedavg.sh   # Shell script for batch training
│   ├── mnist_2.py                     # Training script
│   ├── mnist_config.yaml              # Configuration file
│   ├── grokfast.py                    # Grokfast algorithm implementation
│   ├── dynamic_plot.py                # Real-time visualization
│   └── cluster_run_average.py         # Cluster job management
│
└── pyproject.toml                      # Project configuration
```

## Installation

### Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (fast Python package installer)

To install `uv`, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

1.a. Install `uv` from [Astral's website](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1.b. Clone the repository:
```bash
git clone <repository-url>
cd grokking_vs_learning
```

2. Create a virtual environment and install dependencies using `uv`:
```bash
uv sync
```

## Usage

### Training Neural Networks

#### Ising Spin Model
```bash
cd ising
python run_ising.py --config configs/ising_config.yaml
```

Configuration options can be found in `configs/ising_config.yaml`. Key parameters:
- `epochs`: Number of training epochs
- `learning_rate`: Optimizer learning rate
- `weight_decay`: L2 regularization coefficient
- `train_size` / `test_size`: Dataset sizes
- `train_type`: Task type (e.g., "Ising")

#### Modular Arithmetic
```bash
cd mod_add
python run_mod_add.py --config configs/mod_config.yaml
```

#### MNIST
```bash
cd mnist
python mnist_2.py --config mnist_config.yaml
```

Or use the batch script:
```bash
bash run_script_config_seedavg.sh
```

### Information-Geometric Analysis

After training is complete, run the main analysis script to compute Fisher Information Matrices and other information-geometric quantities:

```bash
python InformationGeometricAnalysis.py
```

**Important Notes:**
- Update the `baseroot` variable in `InformationGeometricAnalysis.py` to point to your data directory
- Set `exp_root` to choose between tasks: `'ising'`, `'modadd'`, or `'mnist'`
- Adjust `epoch_plot_max` and `epoch_sample_stepsize` based on your needs
- The script expects data to be organized with subdirectories: `Learning/` and `Grokking/`

### Analysis and Visualisation

Each task directory includes plotting utilities:

```bash
# Ising pruning analysis
cd ising/plotting
python plot_pruning.py
python plot_pruning_per_layer.py

# ModAdd analysis
cd mod_add/plotting
python plot.py
python simple_plot.py
```

## Data

The repository includes code for generating synthetic datasets:
- **Ising Dataset**: Generated using the Ising model configuration
- **ModAdd Dataset**: Modular arithmetic examples
- **MNIST**: Uses standard MNIST dataset

## Configuration

Each task uses YAML configuration files in their respective `configs/` directories. Common parameters include:

- `seed_*`: Random seed ranges for reproducibility
- `learning_rate`: Learning rate for SGD optimizer
- `weight_decay`: L2 regularization strength
- `train_size` / `test_size`: Dataset sizes
- `epochs`: Maximum training epochs
- `hiddenlayers_input`: Network architecture (hidden layer sizes)
- `train_fraction`: Fraction of data used for training

## Key Features

- **Fisher Information Matrix Computation**: Efficient computation of FIMs for analyzing parameter space geometry
- **Multi-Task Support**: Unified framework for analyzing different learning tasks
- **Seed Averaging**: Support for averaging results across multiple random seeds
- **Interactive Visualisation**: Real-time plotting of training dynamics
- **Flexible Architecture**: Support for different network architectures and activation functions

## References


## License

See [LICENSE](LICENSE) for details.

## Notes

- The working directory must be set to the respective task folder (ising, mod_add, or mnist) when running training scripts
- Ensure filepaths are correctly configured for your system
- Data paths should be configured in both training scripts and the analysis script
- Computation of Fisher Information Matrices can be memory-intensive; adjust batch sizes accordingly for your hardware

## Contact

For questions or data requests, please contact the authors.
