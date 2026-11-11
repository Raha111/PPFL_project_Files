# PPFL Project Files

This repository collects the project files for research on privacy-preserving
federated learning (PPFL), TinyML, and resource-constrained IoT devices. It brings
together experimental code, literature-review materials, project documents,
presentations, and supporting research papers.

## Folder guide

| Folder | Contents and purpose |
| --- | --- |
| [`FL/`](FL/) | CIFAR-10 image-classification experiments comparing MLP, CNN, and TinyCNN models. Includes Python training scripts, an interactive notebook, and a README with setup and experiment instructions. |
| [`FL/FL_with_DP/`](FL/FL_with_DP/) | Federated learning with differential privacy using TensorFlow Federated and federated EMNIST digits. Includes the main `final_(1).ipynb` experiment, the exploratory `TFF.ipynb`, a thesis report, and documentation covering noise schedules and privacy accounting. |
| [`FL/data/`](FL/data/) | Local CIFAR-10 dataset files: the downloaded archive and extracted Python batches used by the classification experiments. The large archive is stored with Git LFS. |
| [`current_reviews/`](current_reviews/) | Survey and review papers covering federated learning, TinyML, microcontroller hardware, IoT, security, and constrained devices. |
| [`papers/`](papers/) | Research-paper collection supporting the literature review and project design, including communication efficiency, resource constraints, and federated TinyML. |
| [`papers/security/`](papers/security/) | Supporting papers on security, privacy, secure aggregation, and related IoT/federated-learning topics. |
| [`papers/similiar papers/`](papers/similiar%20papers/) | Related implementations and studies used for comparison, including edge-device federated learning, ESP32, LoRa, offloading, and TinyML. The original folder spelling is preserved. |
| [`Raha/`](Raha/) | Full-text screening package with 25 numbered papers, `Fulltext_check_Raha.xlsx`, and `READ ME FIRST.txt` explaining the review workflow. |
| [`__MACOSX/`](__MACOSX/) | macOS archive metadata retained from the original project files. |

## Documents and reports

These files are stored at the repository root. Descriptions identify their role
in the project collection; the files themselves contain the detailed material.

| File | Description |
| --- | --- |
| `Project Summary.docx` | Project summary document. |
| `Project_details.pdf` | Project details and background. |
| `Recommended overall approach.docx` | Proposed overall research approach. |
| `FedBGS.docx` | Supporting document named FedBGS. |
| `RQs (1).pdf` | Research questions. |
| `FL_PPFL_SLR_Paper.pdf` | Federated learning / privacy-preserving federated learning systematic literature review paper. |

## Presentations

| File | Description |
| --- | --- |
| `Custom_FL_CIFAR10_Results_Slides.pptx` | Slides presenting the custom federated-learning CIFAR-10 results. |
| `problem formulation - PPFL.pptx` | PPFL problem-formulation presentation. |
| `problem formulation - PPFL_feedback (1).pptx` | Feedback version of the problem-formulation presentation. |

## Literature-review and planning spreadsheets

| File | Description |
| --- | --- |
| `Federated_TinyML_Literature_Review.xlsx` | Literature-review workbook for federated TinyML. |
| `PPFL_Papers_GitHub_Framework_Results.xlsx` | Workbook relating PPFL papers to GitHub projects, frameworks, and results. |
| `PPFL_TinyML_Publication_Venue_Shortlist.xlsx` | Publication-venue shortlist for PPFL/TinyML work. |
| `PPFL_TinyML_Similar_Papers_List.xlsx` | List of related PPFL/TinyML papers. |
| `SLR SUMMARY.xlsx` | Systematic literature review summary workbook. |
| `till_ABSTRACT.xlsx` | Literature-screening workbook labeled through the abstract stage. |
| `25-26.xlsx` | Additional project workbook, retained under its original filename. |
| `final.xlsx` | Additional final workbook, retained under its original filename. |

## Working with the experiments

Start with [`FL/README.md`](FL/README.md) for the CIFAR-10 experiments and
[`FL/FL_with_DP/README.md`](FL/FL_with_DP/README.md) for the differentially private
federated-learning notebooks. Each experiment has its own dependencies and
execution instructions; this collection is not a single executable application.

To retrieve the large dataset archive after cloning, install Git LFS and run:

```bash
git lfs install
git lfs pull
```

The separate MCU simulator folder `mcu_class_fl_sim-main/` is intentionally
excluded. Local virtual environments, Python caches, notebook checkpoints, and
nested Git metadata are also excluded from this repository.
