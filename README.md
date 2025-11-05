# JPM Project

This project implements a Kalman filter using TensorFlow and TensorFlow Probability, designed for financial time series analysis.

## Features

- Kalman filter implementation with TensorFlow backend
- Support for both 1D and multi-dimensional state spaces
- Log-likelihood computation for model comparison
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
│   │   └── KalmanFilter.py    # Main Kalman filter implementation
│   └── data/                  # Data loading and processing utilities
├── test/
│   └── test_kalman.py        # Unit tests
├── scripts/
│   └── run_kalman_tests.py   # Standalone test runner
└── Test.ipynb                # Example notebook
```

## Usage

### Basic Example

```python
import numpy as np
from models.KalmanFilter import KalmanFilter

# Create a simple 1D Kalman filter
kf = KalmanFilter(
    transition_matrix=np.array([[1.0]]),      # State stays constant
    observation_matrix=np.array([[1.0]]),     # Observe state directly
    transition_cov=np.array([[0.1]]),         # Small state noise
    observation_cov=np.array([[0.5]]),        # Larger observation noise
    initial_mean=np.zeros((1,)),             # Start at zero
    initial_cov=np.eye(1),                   # High initial uncertainty
)

# Filter some observations
observations = np.array([[1.0], [1.1], [0.9], [1.2]])  # Example measurements
filtered_means, filtered_covs, loglik = kf.filter(observations)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_kalman.py

# Run standalone tests
python scripts/run_kalman_tests.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.