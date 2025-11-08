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

## Installation

### Prerequisites

- Python 3.8 or later
- pip (Python package installer)

### Quick Install

```bash
# Install from source (recommended for development)
python -m pip install -e .

# Install with development tools (pytest, black, etc.)
python -m pip install -e ".[dev]"

# Ensure TensorFlow and TensorFlow Probability are installed (required for EKF tests)
python -m pip install "tensorflow>=2.10" "tensorflow-probability>=0.18"
```

### Dependencies

Core dependencies (installed automatically):
- tensorflow >= 2.10.0
- tensorflow-probability >= 0.18.0
- numpy >= 1.19.0

Development dependencies (optional):
- pytest >= 7.0
- black
- flake8
- mypy

## Project Structure

```
jpm-project/
├── src/
│   ├── models/
│   │   ├── KalmanFilter.py    # Linear Kalman filter (TensorFlow)
│   │   └── ExtendedKalmanFilter.py  # EKF (TensorFlow + TFP)
│   └── data/                  # Data loading and processing utilities
├── test/
│   ├── test_kalman.py         # Linear Kalman filter tests
│   └── test_extended_kalman.py# Extended Kalman filter tests
├── scripts/
│   └── run_kalman_tests.py   # Optional standalone test runner
└── Test.ipynb                # Example notebook
```

## Usage

### Linear Kalman Filter (TensorFlow)

```python
import numpy as np
from src.models.KalmanFilter import KalmanFilter

# Create a simple 1D Kalman filter
kf = KalmanFilter(
    transition_matrix=np.array([[1.0]]),      # State stays constant
    observation_matrix=np.array([[1.0]]),     # Observe state directly
    transition_cov=np.array([[0.1]]),         # Small state noise
    observation_cov=np.array([[0.5]]),        # Larger observation noise
    initial_mean=np.zeros((1,)),              # Start at zero
    initial_cov=np.eye(1),                    # Prior uncertainty
)

# Filter some observations (shape (T, obs_dim))
observations = np.array([[1.0], [1.1], [0.9], [1.2]])
filtered_means, filtered_covs, loglik = kf.filter(observations)
```

### Extended (nonlinear) Kalman Filter (EKF)

The EKF uses TensorFlow automatic differentiation to compute Jacobians. Provide `f` and `h` using TensorFlow ops.

```python
import tensorflow as tf
from src.models import ExtendedKalmanFilter

def f(x: tf.Tensor) -> tf.Tensor:
    return x + 0.05 * tf.square(x)

def h(x: tf.Tensor) -> tf.Tensor:
    return x + 0.1 * tf.square(x)

ekf = ExtendedKalmanFilter(
    f=f,
    h=h,
    Q=tf.constant([[0.01]], dtype=tf.float64),
    R=tf.constant([[0.1]], dtype=tf.float64),
    initial_mean=tf.constant([0.0], dtype=tf.float64),
    initial_cov=tf.constant([[1.0]], dtype=tf.float64),
)

# observations can be a NumPy 1D array or TF tensor; outputs are TF tensors
means, covs, loglik = ekf.filter(observations)
print(means.numpy()[:5])
```

## Running Tests

```bash
# Run all tests (requires TensorFlow and TFP for EKF tests)
pytest

# Run specific test files
pytest test/test_kalman.py
pytest test/test_extended_kalman.py

# Run standalone tests (if provided)
python scripts/run_kalman_tests.py
```

If you don't have dev dependencies installed yet, install them with:

```bash
python -m pip install -e ".[dev]"
python -m pip install "tensorflow>=2.10" "tensorflow-probability>=0.18"
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.