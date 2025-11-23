JPMSummer — Particle flow & Kalman filter experiments
=====================================================

**Project**
- **Summary:**: JPMSummer contains experimental implementations of Kalman filters and particle-based filters, including particle-flow filters (EDH/LEDH), kernelized particle flows (scalar & matrix kernel variants), and particle filters. It also includes data generators and notebooks for running and visualizing experiments.
- **Purpose:**: Research and teaching experiments for sequential Bayesian filtering, particle flows, and kernel flows. Useful as a reference and starting point for algorithm exploration and comparison.

**Repository Structure**
- **Root files:**: `pyproject.toml`, `setup.py`, `test-requirements.txt`, `LICENSE`, `README.md`, and example notebooks (`experiments_part_1.ipynb`, `kalman_filter_demo.ipynb`).
- **Source:**: `src/` contains the core code:
  - `src/models/` — implementations of `KalmanFilter`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, `ParticleFilter`, `ParticleFlow` variants (`PFPF.py`, `EDH.py`, `LEDH.py`), and `KernelFlow.py` (kernelized flows).
  - `src/data/` — data generators and helper functions (e.g., `StochasticVariationalData`).
  - `src/tf_keras/` — small TF/Keras helper wrappers used by demos.
- **Tests & Demos:**: `test/` contains unit tests and `experiments_part_1.ipynb` contains an interactive demo and experiments.

**Features**
- **Particle flows:**: Implementations of EDH (global linearization) and LEDH (local linearization) particle-flow filters.
- **Kernel flows:**: KernelScalarFlow and KernelMatrixFlow implementations with single-step and sequential `filter` APIs.
- **Particle filter:**: A standard particle filter with predict/update/resample routines and a convenience `filter` API for sequential observations.
- **Data generators:**: `StochasticVariationalData` supports multi-dimensional state and observation generation for repeatable experiments.
- **Notebooks & demos:**: Interactive experiments for comparing filters and visualizing particle behavior.

**Requirements**
- **Python:**: 3.8+ recommended (code tested with Python 3.11 in the development environment).
- **Core packages:**: Installable from `test-requirements.txt` for a reproducible environment. Typical packages include `numpy`, `scipy`, `matplotlib`, `tensorflow`, `tensorflow-probability`, and `pytest`.

Quick setup (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r test-requirements.txt
# Optionally install the package in editable mode
python -m pip install -e .
```

If you prefer installing only runtime dependencies, ensure `tensorflow` and `tensorflow-probability` are installed for TF-based demos:
```powershell
python -m pip install tensorflow tensorflow-probability
```

**Running tests**
- Run the test suite with:
```powershell
python -m pytest -q
```

**Running demos & examples**
- KernelFlow smoke test (single-script demo):
```powershell
python src\models\KernelFlow.py
```
- Particle-flow PF demo / smoke test:
```powershell
python src\models\PFPF.py
```
- Run the interactive notebook (execute in-place):
```powershell
jupyter nbconvert --execute experiments_part_1.ipynb --to notebook --inplace
```

**Code Usage Examples**
- Run a quick particle-filter sequence (Python REPL or script):
```python
from src.models.ParticleFilter import ParticleFilter
# construct ParticleFilter, provide model functions, initial particles, and observations
# pf.filter(observations) returns stacked means, covariances and log-likelihoods
```
- Use kernel flow to filter a sequence:
```python
from src.models.KernelFlow import KernelScalarFlow
kf = KernelScalarFlow(kernel_param=1.0)
means, covs, lls = kf.filter(observations, particles, H, R, h_func=my_h)
```

**API Summary**
- `src/models/ParticleFilter.py` — `ParticleFilter` with `.step()` and `.filter()` convenience API.
- `src/models/PFPF.py` — `EDH_ParticleFlowPF`, `LEDH_ParticleFlowPF` with `step()` and `filter()` APIs for sequential observations.
- `src/models/KernelFlow.py` — `KernelScalarFlow` and `KernelMatrixFlow` with `flow()` and `filter()`.
- `src/data/data.py` — `StochasticVariationalData` for synthetic data generation supporting `n_state` and `n_obs`.

**Notes & Known Issues**
- Parity between EDH and LEDH particle-flow variants may not be exact for all settings; the repository includes a parity self-test that may highlight differences requiring algorithmic investigation.
- Some module entry points and convenience functions use TensorFlow-to-NumPy conversions for simplicity; converting to fully TF-graph-based implementations is left as future work.

**Contributing**
- Contributions are welcome. Please open issues or pull requests.
- For code changes, add tests in the `test/` directory and keep changes minimal and focused.

**License**
- See the `LICENSE` file at the repository root for license details.

**Contact / Maintainer**
- Repository owner: `Yechang618` (local workspace).

**Next steps**
- Run the test suite locally: `python -m pytest -q`.
- If you want, I can help debug the EDH vs LEDH parity issue and propose a fix or more diagnostic tests.
