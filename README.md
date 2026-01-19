```markdown
# State-Space Modeling and Filtering Framework

This repository provides a unified, modular implementation of **state-space models (SSMs)** and **Bayesian filtering algorithms** for both linear-Gaussian and nonlinear/non-Gaussian systems. It supports synthetic data generation, Kalman filtering (KF/EKF/UKF), and advanced particle flow methods including **Exact Daum–Huang (EDH)** and **Particle Flow Particle Filters (PFPF)**.

Built with **TensorFlow 2 + TensorFlow Probability**, the code is numerically stable, GPU-ready, and designed for research reproducibility in time series analysis, financial engineering, and signal processing.

---

## 📁 Project Structure

```
.
├── src/
│   ├── data/               # Synthetic SSM data generators
│   │   └── data.py         # LinearGaussianSSM, StochasticVolatilityModel, RangeBearingSSM, Lorenz96NonGaussianSSM
│   └── models/             # Filtering algorithms
│       ├── kalman_filters.py   # KF, EKF, UKF
│       └── particle_filters.py # Standard PF, EDH, LEDH, PFPF-EDH, PFPF-LEDH
│
├── experiments/            # Reproducible experiment scripts
│   ├── kf_linear_gaussian_stability.py     # KF optimality & numerical stability
│   ├── compare_range_bearing_filters.py    # EKF/UKF vs. particle flows on radar tracking
│   ├── svm_experiment.py                   # Stochastic volatility model filtering
│   └── multidim_nongaussian_ssm.py         # High-dimensional chaotic system (Lorenz-96)
│
├── test/                   # Unit tests for core components
│   ├── test_data.py        # Tests for SSM data generators
│   ├── test_kalman_filters.py  # Tests for KF, EKF, UKF
│   └── test_particle_flows.py  # Tests for EDH, LEDH, and flow functions
│
└── figures/                # Auto-generated plots (PDF format)
```

---

## 🔧 Core Features

### 📊 Data Generators (`src/data/data.py`)
- **`LinearGaussianSSM`**: Classic linear Gaussian state-space model.
- **`StochasticVolatilityModel`**: Financial time series with log-volatility dynamics.
- **`RangeBearingSSM`**: 2D constant-velocity target tracking with nonlinear range/bearing observations.
- **`Lorenz96NonGaussianSSM`**: High-dimensional chaotic system with Student-t noise and nonlinear observations.

### 🧮 Filtering Algorithms (`src/models/`)
#### Kalman Filters
- **`KalmanFilter`**: Joseph-form stabilized linear KF (numerically robust).
- **`ExtendedKalmanFilter`**: First-order Taylor linearization.
- **`UnscentedKalmanFilter`**: Sigma-point based nonlinear filtering.

#### Particle-Based Methods
- **`StandardParticleFilter`**: Bootstrap PF with systematic resampling.
- **`EDHFlow` / `LEDHFlow`**: Deterministic particle flow (global/local Jacobian).
- **`PFPF_EDH` / `PFPF_LEDH`**: Full particle flow particle filters with weight updates and ESS tracking.
- **`algorithm1_DH_filter` / `algorithm2_modified_DH_filter`**: Reference implementations of Daum–Huang exact flow.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Conda environment (recommended: `jpmsummer`)
- Dependencies:
  ```bash
  tensorflow==2.15.1
  tensorflow-probability==0.23.0
  numpy
  scipy
  matplotlib
  ```

### Example: Run Linear-Gaussian KF Stability Test
```bash
cd experiments
python kf_linear_gaussian_stability.py
```
Generates:
- `./figures/lg_ssm_rmse.pdf`
- `./figures/lg_ssm_condition_number.pdf`
- `./figures/lg_ssm_eigenvalues.pdf`
- Console diagnostics on SPD preservation and condition number

---

## 🧪 Testing

A comprehensive test suite ensures correctness of data generators, filters, and particle flows.

### Run All Tests
```bash
# From project root
python -m pytest test/ -v
```

### Run Individual Tests
```bash
python -m unittest test.test_data
python -m unittest test.test_kalman_filters
python -m unittest test.test_particle_flows
```

### Test Coverage
| Test File | Validates |
|----------|-----------|
| `test_data.py` | Shape, dtype, and statistical properties of all SSMs |
| `test_kalman_filters.py` | Output shapes, covariance definiteness, and basic functionality of KF/EKF/UKF |
| `test_particle_flows.py` | Numerical stability and shape correctness of EDH/LEDH flow functions |

> 💡 **Note**: Ensure no local folders named `tensorflow`, `keras`, or `tf_keras` exist in `src/` to avoid import conflicts.

---

## 📈 Experiments Included

| Experiment | Task | Key Metrics |
|-----------|------|-------------|
| `kf_linear_gaussian_stability.py` | Validate KF optimality & Joseph-form stability | RMSE, condition number, covariance eigenvalues |
| `compare_range_bearing_filters.py` | Radar tracking (nonlinear) | Position RMSE, trajectory plot, runtime |
| `svm_experiment.py` | Financial stochastic volatility | Latent state RMSE, estimated volatility |
| `multidim_nongaussian_ssm.py` | 20D Lorenz-96 chaos | Full-state RMSE, ESS, flow magnitude |

All experiments save vector-quality PDFs to `./figures/`.

---

## ✅ Design Principles

- **Numerical Robustness**: Joseph-form covariance updates, Cholesky-based inversions, jitter regularization.
- **Modularity**: Drop-in replacement of SSMs and filters.
- **Reproducibility**: Fixed seeds, explicit dtype control (`tf.float64` by default).
- **Performance**: Vectorized operations, TensorFlow graph compilation (`@tf.function`), memory-efficient TensorArrays.

---

## 📚 References

- Daum, F., & Huang, J. (2008). *Exact particle flow for nonlinear filters*. SPIE.
- Li, X. R., & Coates, M. (2017). *Particle flow particle filter using exact DAUM-HUANG flow*. IEEE TSP.
- Sarkka, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press.

---

> **Note**: This framework is actively used in research on **basis arbitrage**, **time series forecasting**, and **risk-aware portfolio optimization**. For extensions (e.g., square-root KF, adaptive resampling), contact the author or open an issue.
```

---
