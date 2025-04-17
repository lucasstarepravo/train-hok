# train-hok

train-hok is a PyTorch-based framework with Distributed Data Parallel for training surrogate models for the high-order kernel in high-order mesh-free methods. This repository supports training of: MLP (with regular and custom loss function) and ResMLP. 'hok' stands for high-order kernel.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Resuming Training](#resuming-training)
- [Requirements](#requirements)


## Features

- **Multiple Model Types:** Train surrogate models using MLP, or PINN.
- **Distributed Training:** Leverages PyTorch Distributed Data Parallel for efficient multi-GPU training.
- **Custom Architecture:** Define the model architecture by specifying the number of layers and neurons in each layer.
- **Flexible Parameter Settings:** Adjust high-level training parameters (e.g., number of epochs, batch size) via the `run_model` function in `main.py` or modify low-level settings directly in the `models` folder.
- **Resume Training:** Use `Resume_main.py` to continue training from a checkpoint (default checkpoint interval is every 1000 epochs, adjustable in `models/NN_Base.py`).

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lucasstarepravo/train-hok.git


2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

## Usage
1. Main Training

The primary entry point is `main.py`.
Before running, define the following:
- The location of your training data.
- The architecture of the model (number of layers and neurons per layer).
- The type of model to train
- The number of GPUs to use.
- The directory where the trained model should be saved.

To start training, run:
  ```bash
  python main.py
 ```

2. Parameter Adjustment
- **High-Level Settings:** Modify the `run_model` function in `main.py` to change parameters such as the number of epochs and batch size.
- **Low-Level Settings:** For more detailed customizations, edit the specific parameters in the classes within the `models` folder.

## Resuming Training

To resume training from a checkpoint, use the `Resume_main.py` script. Checkpoints are saved every 1000 epochs by default (this interval can be changed in `models/NN_Base.py`):

```bash
python Resume_main.py
```

## Requirements

- **Python:** Version 3.9 or later (tested with Python 3.12)
- **PyTorch:** Version 2.2.1 (see `requirements.txt` for the complete list of dependencies)

