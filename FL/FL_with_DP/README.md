# Federated Learning with Differential Privacy

This repository explores privacy-preserving federated learning with TensorFlow
Federated (TFF). The main experiment trains an image classifier on federated
EMNIST digits, applies user-level differential privacy to model aggregation,
tracks the cumulative privacy budget, and compares several noise schedules in
terms of privacy and model utility.

The repository also includes an earlier federated-learning prototype and the
associated thesis report.

## Repository contents

| File | Purpose |
| --- | --- |
| `final_(1).ipynb` | Main EMNIST experiment with DP-FedAvg, privacy accounting, scheduled noise, validation, CSV logging, and comparative plots |
| `TFF.ipynb` | Exploratory TensorFlow Federated examples using EMNIST/MNIST, FedAvg, and initial differential-privacy experiments |
| `Thesis_report.pdf` | Thesis report describing the project and its findings |

## Main experiment

The main notebook uses the digits-only federated EMNIST dataset supplied by
TensorFlow Federated. Each writer is treated as a client, preserving EMNIST's
naturally federated structure.

The workflow includes:

- deterministic seeding for Python, NumPy, and TensorFlow;
- client-level training and validation splits;
- a multilayer perceptron with two 200-unit ReLU hidden layers;
- unweighted Federated Averaging with SGD client and server optimizers;
- differentially private model-update aggregation;
- Rényi differential privacy (RDP) accounting with `dp-accounting`;
- calibration against target privacy parameters;
- periodic validation and optional early stopping;
- per-round CSV logging of epsilon, accuracy, loss, and noise;
- privacy–utility analysis and comparison plots.

### Default privacy configuration

The notebook defines the following initial experiment targets:

| Parameter | Value |
| --- | ---: |
| Target epsilon | `3.0` |
| Target delta | `1e-5` |
| Calibration rounds | `100` |
| Noise-to-clients ratio | `0.01` |
| Validation-client fraction | `0.10` |
| Evaluation interval | Every 5 rounds |

The actual training helper accepts a configurable round limit and other
evaluation/early-stopping settings.

## Noise schedules

The main notebook implements and compares multiple noise schedules:

- `flat`: constant Gaussian noise;
- `step5_low_to_high`: five stages of increasing noise;
- `step5_high_to_low`: five stages of decreasing noise;
- `cosine`: smoothly decreasing noise using a cosine curve;
- `linear_decreasing`: linearly decreasing noise;
- `exponential_decay`: exponentially decreasing noise.

The included experiment cells directly run the flat and two five-step
schedules. Other schedules can be selected through the same training function.

## Reported endpoint comparison

The final notebook contains the following recorded endpoints for three runs:

| Run | Test accuracy | Test loss | Rounds | Epsilon |
| --- | ---: | ---: | ---: | ---: |
| Flat adaptive clipping | 0.8126 | 0.5950 | 135 | 3.282 |
| Step low-to-high | 0.7721 | 0.7806 | 80 | 2.441 |
| Step high-to-low | 0.7850 | 0.7040 | 140 | 2.411 |

These values are embedded in the analysis notebook as saved experimental
results; rerunning the complete experiment may produce different results due to
software versions, runtime configuration, and stochastic training.

## Running the notebooks

The notebooks are designed primarily for Google Colab. A fresh Colab runtime is
recommended because TensorFlow Federated supports a limited range of TensorFlow
and Python versions.

1. Open `final_(1).ipynb` in Google Colab.
2. Install the required packages in the first cell.
3. Restart the runtime if package installation requests it.
4. Run the setup, dataset, model, privacy-calibration, and training cells in
   order.
5. Update `base_dir` or `BASE_DIR` if Google Drive is not mounted at the paths
   used in the notebook.

A typical dependency set is:

```python
!pip install tensorflow tensorflow-federated tensorflow-privacy dp-accounting \
    numpy pandas matplotlib
```

Use mutually compatible versions of TensorFlow, TensorFlow Federated, and
TensorFlow Privacy. Installing their latest releases independently may result in
dependency conflicts.

## Outputs and analysis

Training runs log the following fields to CSV:

```text
round, epsilon, train_acc, train_loss, val_acc, val_loss, sigma
```

The analysis cells can generate:

- validation accuracy and loss across rounds;
- training-versus-validation comparisons;
- epsilon and noise level across rounds;
- validation accuracy as a function of epsilon;
- accuracy gained per unit of privacy budget;
- area under the privacy–utility curve;
- three-run comparison tables and 3D plots.

By default, some cells read and write under `/content/drive/MyDrive/tff_runs` or
`/content`. Change these paths before running locally or in a different Colab
workspace.

## Notes

- `TFF.ipynb` is an exploratory notebook and contains intermediate experiments;
  use `final_(1).ipynb` as the primary implementation.
- Differential-privacy guarantees depend on the sampling model, clipping,
  aggregation mechanism, and accountant assumptions. Recalculate epsilon after
  changing the number of clients, rounds, sampling rate, or noise schedule.
- The notebooks should be run sequentially in a clean runtime to avoid stale
  TensorFlow Federated state and incompatible model/process objects.
