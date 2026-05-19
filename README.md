# Seismic Neural Network Models

A collection of deep-learning models for seismic wavelet extraction and well-tie. The repository implements three distinct neural network architectures — a **Multi-Layer Perceptron (MLP)**, a **Dual-Task Autoencoder CNN**, and a **Time-Shift (TS) network** — each designed to tackle different aspects of seismic data analysis.

---

## Repository Structure

```text
.
├── models/
│   ├── dt.py              # Dual Task Neural Network (CNN Autoencoder with 2 branches) model - Wavelet Extraction
│   ├── mlp.py             # Multi-Layer Perceptron (MLP) model - Wavelet Extraction
│   └── ts.py              # Time Series model - Work in Progress (WIP) - Well-Tie
├── training/              # Output weights (Saved nets)
├── welltie/
│   ├── dataset.py         # Data loading (may need to be changed for specific data)
│   ├── geophysics.py      # Geophysical processing and utilities
│   ├── losses.py          # Custom loss functions
│   ├── model.py
│   └── network.py
├── .gitignore
├── make.bat               # Automation script for Windows
├── Makefile               # Automation script for Linux/macOS
├── parameters.yaml        # Parameter configuration file
├── README.md
├── requirements.txt       # Project dependencies
├── utils.py
└── utils_spectrum.py      # Spectrum processing utilities
```

---

## Prerequisites

| Dependency | Version |
|---|---|
| **Python** | 3.13.2+ |
| **PyTorch** | 2.8+ (CUDA optional) |
| **NumPy** | 2.3.3+ |
| **Matplotlib** | 3.10.6+ |
| **PyYAML** | 6.0.3+ |

> **Note:** GPU acceleration is automatically detected at runtime. If a CUDA-capable device is available, models will use it; otherwise, they fall back to CPU.

### Install Dependencies

```bash
pip install -r requirements.txt
```

A `parameters.yaml` configuration file is expected in the project root. Ensure it is present and correctly populated before running any model.

---

## How to Run

Each model supports two commands:

| Command | Description |
|---|---|
| `train` | Train the model from scratch and run inference afterwards |
| `run` | Load a previously trained model and run inference only |

### Linux

Use the provided **Makefile**:

```bash
# Train a model
make train file=mlp

# Run inference with a trained model
make run file=dt
```

### Windows

Use the provided **make.bat** script:

```batch
:: Train a model
make.bat train mlp

:: Run inference with a trained model
make.bat run dt
```

> **Tip:** The `.py` extension is optional — both `mlp.py` and `mlp` are accepted as the file argument.

---

## Model Architectures

| Feature | `dt.py` — DualTaskAE | `mlp.py` — MLPWaveletExtractor | `ts.py` — TimeShiftPredictor |
|---|---|---|---|
| **Architecture** | 1-D Conv Autoencoder (dual-head) | Fully-connected MLP (5 layers) | Siamese dual-branch 1-D CNN |
| **Task** | Seismic denoising + wavelet spectrum extraction | Direct wavelet estimation from trace | Time-shift prediction between traces |
| **Input** | 1-D seismic trace (variable length) | 300-sample seismic vector | Two 1-D traces (observed + synthetic) |
| **Output** | Reconstructed trace + wavelet spectrum | 97-sample normalised wavelet | Per-sample time-shift curve |
| **Activations** | ReLU | Tanh + L2 norm | ReLU + BatchNorm |
| **Loss** | MSE + α·Spectral MSE | Log-Cosh + Cosine Similarity | Masked MSE + Smoothness |
| **Optimizer** | Adam + StepLR | Adam + StepLR | Adam + StepLR |
| **Epochs** | 100 | 100 | 400 |
| **Status** | Operational | Operational | WIP |

---

### DualTaskAE — Dual-Task Autoencoder (`dt.py`)

A **1-D convolutional autoencoder** with two decoder heads (dual-task design):

| Sub-network | Layers | Details |
|---|---|---|
| **Encoder** | `Conv1d(1→32, k=3, s=2)` → `ReLU` → `Conv1d(32→32, k=3, s=2)` → `ReLU` → `Conv1d(32→8, k=3, s=1)` → `ReLU` | Compresses the input seismic trace into an 8-channel latent representation with 4× temporal downsampling. |
| **Seismic Decoder** | `ConvTranspose1d(8→32, k=4, s=2)` → `ReLU` → `ConvTranspose1d(32→1, k=4, s=2)` | Reconstructs the denoised seismic trace from the latent space (symmetric upsampling). Output is trimmed/padded to match input length. |
| **Wavelet Branch** | `ConvTranspose1d(8→8, k=4, s=2)` → `ReLU` → `ConvTranspose1d(8→8, k=3, s=1)` → `ReLU` → `ConvTranspose1d(8→8, k=3, s=1)` → `AdaptiveAvgPool1d(1)` → `Flatten` → `Linear(8→N)` | Extracts the source wavelet's amplitude spectrum. Output length `N` is dynamically computed from the seismic trace duration and sample interval (`dt`). |

**Loss function** — `DualTaskLoss` (composite):
- **Reconstruction loss:** MSE between input and reconstructed trace — `‖s − s'‖²`
- **Spectral loss:** MSE between the predicted wavelet spectrum and a Gaussian-smoothed, Ormsby-filtered version of the seismic amplitude spectrum — weighted by `α`
- **Total:** `L_reconstruction + α · L_spectral`

An optional **pre-training phase** replaces the spectral loss target with a 30 Hz Ricker wavelet spectrum for warm-start initialisation.

**Key parameters** (`parameters.yaml` → `dual_task`):

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | `32` | Mini-batch size |
| `learning_rate` | `0.001` | Initial Adam LR |
| `max_epochs` | `100` | Training epochs (after pre-training) |
| `pre_train_epochs` | `0` | Warm-start epochs with Ricker target (disabled by default) |
| `lr_decay_every_n_epoch` | `20` | StepLR scheduler interval |
| `lr_decay_rate` | `0.9` | StepLR gamma (multiplicative factor) |
| `train_sample` | `5000` | Number of training samples generated |
| `loss.alpha` | `0.1` | Weight of the spectral loss term |

> **✅ Operational.** Fully functional training and inference pipeline. Supports both the Angola and F3 Demo datasets (selectable via commented path blocks). Produces denoised seismic reconstructions, extracted wavelets, and amplitude spectra visualisations.

---

### MLPWaveletExtractor — Multi-Layer Perceptron (`mlp.py`)

A **fully-connected feedforward network** (5 dense layers) that maps a fixed-length seismic trace directly to a wavelet estimate:

```
Input (300) → Linear(300→300) → Tanh
           → Linear(300→300) → Tanh
           → Linear(300→200) → Tanh
           → Linear(200→97)  → Tanh
           → Linear(97→97)
           → L2 Normalisation
```

| Property | Value |
|---|---|
| **Input dimension** | 300 (fixed-length seismic trace samples) |
| **Output dimension** | 97 (wavelet samples) |
| **Hidden layers** | 4 (widths: 300 → 300 → 200 → 97) |
| **Activation** | `Tanh` on all hidden layers; none on output |
| **Output normalisation** | Energy-normalised: `w / √(Σw² + ε)` with `ε = 1e-8` |

**Loss function** — `MLPLoss` (composite):
- **Log-Cosh loss:** `Σ log(cosh(w − w̃))` — smooth approximation of L1, robust to outliers
- **Cosine Similarity loss:** `1 − mean(cos_sim(w, w̃))` — penalises shape mismatch
- **Total:** `L_logcosh + L_cosine`

**Key parameters** (`parameters.yaml` → `mlp_wavelet`):

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | `256` | Mini-batch size |
| `learning_rate` | `0.001` | Initial Adam LR |
| `max_epochs` | `100` | Training epochs |
| `lr_decay_every_n_epoch` | `80` | StepLR scheduler interval |
| `lr_decay_rate` | `0.5` | StepLR gamma |
| `train_sample` | `100000` | Synthetic training samples generated |
| `qt_seismic_distortions` | `10` | Noise distortion variants per sample |

> **✅ Operational.** Fully functional training and inference pipeline. Uses synthetically generated seismic data for training (via `MLPDataset`) and real well-log/SEG-Y data for evaluation. Reconstructs seismic traces via convolution of the estimated wavelet with reflectivity.

---

### TimeShiftPredictor — Siamese CNN (`ts.py`)

A **Siamese-style dual-branch 1-D CNN** that predicts sample-wise time shifts between an observed and synthetic seismic trace:

| Sub-network | Layers | Details |
|---|---|---|
| **Truth Branch** | `Conv1d(1→32, k=9, same)` → `BatchNorm1d(32)` → `ReLU` → `Conv1d(32→64, k=9, same, dilation=2)` → `BatchNorm1d(64)` → `ReLU` | Encodes the observed seismic trace. Dilated convolution expands the receptive field. |
| **Synthetic Branch** | Identical architecture to Truth Branch | Encodes the synthetic seismic trace (weights are **not** shared — pseudo-Siamese design). |
| **Concat + Prediction** | `Conv1d(128→16, k=9, same)` → `BatchNorm1d(16)` → `ReLU` → `Conv1d(16→1, k=9, same)` | Concatenates the two 64-channel feature maps (128 channels total) and predicts a per-sample time shift. |

**Loss function** — `TimeShiftLoss` (composite):
- **MSE loss:** Masked MSE between predicted and ground-truth time shifts — applied only within the valid well interval
- **Smoothness loss:** Mean absolute first-order difference of the predicted time shift — `mean(|Δts|)`
- **Total:** `L_MSE + 0.5 · L_smooth`

**Key parameters** (`parameters.yaml` → `time_shift`):

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | `8` | Mini-batch size |
| `learning_rate` | `0.001` | Initial Adam LR |
| `max_epochs` | `400` | Training epochs |
| `lr_decay_every_n_epoch` | `100` | StepLR scheduler interval |
| `lr_decay_rate` | `0.9` | StepLR gamma |
| `train_distortions` | `500` | Synthetic time-shift distortions for training augmentation |

> **⚠️ Work in Progress (WIP).** The network architecture and training loop are implemented, but this model has a hard dependency on a pre-trained `DualModel` (loaded at runtime to generate wavelet-derived synthetic traces via `TimeShiftDataset`). The full pipeline — including dataset generation, training, and the post-inference time-domain warping/interpolation step — requires further validation and is **not considered production-ready**.

---

## License

*To be determined.*
