# JPM Project

This project implements a Kalman filter using TensorFlow and TensorFlow Probability, designed for financial time series analysis.

## Features

 - Extended (nonlinear) Kalman filter implemented with TensorFlow + TensorFlow Probability (EKF)
 - A lightweight NumPy-based EKF demo helper is also included for quick experiments

## Installation
Create a simple 1D Kalman filter
### Prerequisites

```bash
# Run all tests (requires tensorflow and tensorflow-probability installed)
pytest

# Run specific test file
pytest test/test_kalman.py
pytest test/test_extended_kalman.py

# Run standalone tests (if provided)
python scripts/run_kalman_tests.py
```

Note: the Extended Kalman Filter test uses TensorFlow; install dev extras
to get pytest and other development tools:

# JPM Project

This project implements Kalman filtering tools using TensorFlow and TensorFlow Probability, primarily focused on time-series (financial) applications. It includes both a linear Kalman Filter and an Extended Kalman Filter (EKF) that uses automatic differentiation for Jacobians.

## Features

- Linear Kalman filter implementation with TensorFlow backend
- Extended (nonlinear) Kalman filter implemented with TensorFlow + TensorFlow Probability (EKF)
- A lightweight NumPy-based EKF demo helper is included for quick experiments and tests
- Support for 1D and multi-dimensional state spaces
- Log-likelihood computation for model comparison and diagnostics
- Example notebook for stock data analysis
 - Particle Filter (PF) implementation with diagnostics:
     - Tracks effective sample size (ESS) and normalized particle weights per step
     - Provides `last_ess`, `ess_history` and `weights_history` attributes for plotting degeneracy
     - See `kalman_filter_demo.ipynb` for an ESS visualization and particle-count sweep example

## Installation

### Prerequisites
# JPMSummer

This repository contains implementations of Kalman filters and particle filters, plus experiment scripts comparing methods (Kalman Filter, UKF, Bootstrap PF, PF-PF with Daum–Huang flows, EDH, LEDH).

**Features**
- **Kalman & EKF**: Linear Kalman filter and Extended Kalman Filter (EKF) implemented with TensorFlow and automatic differentiation.
- **Unscented Kalman Filter (UKF)**: Implemented with numerically-stable updates.
- **Particle Filters**: Bootstrap PF and PF-PF (Particle Flow Proposal) with global (IEDH) and local (ILEDH / LEDH) invertible Daum–Huang flows.
- **Flows & Diagnostics**: Robust linear-algebra helpers (`_safe_cholesky`, `_safe_inv`) with optional debug tracing (`PF_PF_DEBUG` env var).
- **Experiments**: `experiments_pf_pf.py` contains comparison experiments and a sensitivity study `experiment_linear_gaussian_part2` that perturbs predictive covariances.

**Quick Notes**
- **Debug**: set `PF_PF_DEBUG=1` to enable diagnostic prints from PF-PF helpers (shows min eigenvalues and jitter escalation). In PowerShell: `$env:PF_PF_DEBUG='1'`.
- **Smoke tests**: many experiments support a `smoke` flag or small defaults for quick validation.

**Project Structure**
- **`src/models/`**: Kalman, EKF, UKF, ParticleFilter, PF_PF (flows), EDH, LEDH implementations.
- **`experiments_pf_pf.py`**: experiment harnesss (acoustic tracking, linear Gaussian spatial network, skewed-t counts) and the sensitivity experiment `experiment_linear_gaussian_part2`.
- **`test/`**: unit tests.

**Dependencies**
- Python 3.8+; tested with Python 3.11
- Required (examples): `numpy`, `tensorflow`, `tensorflow_probability`, `scipy`
- Install with pip (recommended in a venv):
```powershell
python -m pip install -e .
python -m pip install "tensorflow>=2.10" "tensorflow-probability>=0.18" scipy
```

**Running experiments**
- Run the default experiments script (calls `experiment_linear_gaussian` by default):
```powershell
python experiments_pf_pf.py
```
- Run the sensitivity experiment (perturbed predictive covariance) from Python:
```powershell
python -c "from experiments_pf_pf import experiment_linear_gaussian_part2; experiment_linear_gaussian_part2(T=10, trials=5, Np=500, sigma_p_list=[0.0,0.1,0.2], smoke=False)"
```
- Quick smoke test (fast):
```powershell
python -c "from experiments_pf_pf import experiment_linear_gaussian_part2; experiment_linear_gaussian_part2(T=3, trials=2, Np=50, sigma_p_list=[0.0,0.2], smoke=True)"
```

**Enabling PF-PF diagnostics**
- PowerShell:
```powershell
$env:PF_PF_DEBUG='1'
python -c "from experiments_pf_pf import experiment_linear_gaussian_part2; experiment_linear_gaussian_part2(T=5, trials=2, Np=200, sigma_p_list=[0.0,0.2], smoke=True)"
```
- Bash/macOS/Linux:
```bash
PF_PF_DEBUG=1 python -c "from experiments_pf_pf import experiment_linear_gaussian_part2; experiment_linear_gaussian_part2(T=5, trials=2, Np=200, sigma_p_list=[0.0,0.2], smoke=True)"
```

**Notes on numerical stability**
- The PF-PF flows use Cholesky and matrix inversions; we add tiny diagonal jitter and attempt increasing jitter when operations fail. Use `PF_PF_DEBUG=1` to see jitter/diagnostic messages.
- For large `d` and large `Np` experiments, prefer running smoke/medium runs first to confirm numerical stability and resource usage.

If you'd like, I can:
- add capture-and-save for matrices that trigger jitter escalation, or
- run the full-scale sensitivity experiment (trials=100, T=10, Np=10000) — note this is computationally heavy and may take a long time.

## Contributing
Follow the normal fork → branch → PR workflow. Keep changes small and test locally with smoke runs before pushing.

## License
MIT — see `LICENSE`.
