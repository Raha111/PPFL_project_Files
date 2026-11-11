# CIFAR-10 Model Comparison

This project compares multilayer perceptron (MLP) and convolutional neural
network (CNN) architectures on the CIFAR-10 image-classification dataset. It
contains command-line training scripts and a notebook for interactive analysis.

## Models

| Model | Description |
| --- | --- |
| MLP | Fully connected baseline operating on flattened `32 x 32 x 3` images |
| CNN | Three convolutional layers followed by a two-layer classifier |
| TinyCNN | Smaller two-convolution model for a faster baseline |

All models produce logits for the 10 CIFAR-10 classes. Training images use
random crops and horizontal flips; evaluation images only use normalization.

## Project files

- `train_cifar_models.py` trains and evaluates the MLP and CNN for five epochs.
- `cifar10_model_compare.py` compares the MLP, CNN, and TinyCNN for three epochs.
- `cifar10_model_comparison.ipynb` provides interactive training, evaluation,
  plots, a confusion matrix, and a classification report.

## Setup

Python 3.11 is recommended. Create a virtual environment and install the
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib scikit-learn ipykernel
```

For Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

PyTorch installation can differ when using an NVIDIA GPU. Select the
appropriate build for your system from the official PyTorch installation
guide.

## Run the experiments

Run the main MLP/CNN experiment:

```powershell
python train_cifar_models.py
```

Run all three model variants:

```powershell
python cifar10_model_compare.py
```

The first run downloads CIFAR-10 into `data/`. CUDA is selected automatically
when supported by the installed PyTorch build; otherwise, training runs on the
CPU.

To use the notebook in VS Code, select `.venv` as the notebook kernel and run
the cells from the beginning. Restart the kernel after changing a model class so
that an older class definition is not retained in memory.

## Current experiment settings

- Batch size: 128
- Optimizer: Adam with a learning rate of `1e-3`
- Loss: cross-entropy
- Image size: `3 x 32 x 32`
- Normalization: CIFAR-10 channel means and standard deviations

The scripts currently use the official CIFAR-10 test split for per-epoch
evaluation. For rigorous model selection, create a validation split from the
training set and reserve the test split for one final evaluation.

## Suggested next experiments

- Add batch normalization and dropout to the CNN.
- Train for more epochs with a learning-rate scheduler.
- Compare parameter counts and training time in addition to accuracy.
- Add a CIFAR-adapted ResNet-18 baseline.
- Fix random seeds and repeat each experiment to measure variance.
