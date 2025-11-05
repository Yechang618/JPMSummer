import numpy as np
# import unittest
import pytest
from importlib.util import find_spec


# Check if required packages are available
TENSORFLOW_AVAILABLE = find_spec("tensorflow") is not None
TFP_AVAILABLE = find_spec("tensorflow_probability") is not None
SKIP_REASON = "Missing required package: " + (
    "tensorflow" if not TENSORFLOW_AVAILABLE else 
    "tensorflow_probability" if not TFP_AVAILABLE else 
    "unknown"
)


@pytest.mark.skipif(not (TENSORFLOW_AVAILABLE and TFP_AVAILABLE),
                   reason=SKIP_REASON)
def test_import_kalman():
    """The KalmanFilter module should import."""
    # Import the KalmanFilter class
    from models.KalmanFilter import KalmanFilter  # noqa: E402


# @pytest.mark.skipif(not (TENSORFLOW_AVAILABLE and TFP_AVAILABLE),
#                    reason=SKIP_REASON)
def test_kalman_filter_1d_recovery():
    """Simulate a 1D system and ensure the Kalman filter recovers the state.

    This test is intentionally lenient: it checks that the RMSE between the
    filtered means and the true latent states is below a reasonable threshold
    for the chosen noise levels.
    """
    from models.KalmanFilter import KalmanFilter

    # Seed RNGs for determinism
    np.random.seed(1)

    T = 100
    A = np.array([[1.0]], dtype=float)
    H = np.array([[1.0]], dtype=float)
    Q = np.array([[0.1]], dtype=float)
    R = np.array([[0.5]], dtype=float)

    # Simulate
    x = np.zeros((T + 1, 1))
    y = np.zeros((T, 1))
    for t in range(1, T + 1):
        x[t] = A @ x[t - 1] + np.random.normal(scale=np.sqrt(Q[0, 0]))
    for t in range(T):
        y[t] = H @ x[t + 1] + np.random.normal(scale=np.sqrt(R[0, 0]))

    kf = KalmanFilter(
        transition_matrix=A,
        observation_matrix=H,
        transition_cov=Q,
        observation_cov=R,
        initial_mean=np.zeros((1,)),
        initial_cov=np.eye(1) * 1.0,
    )

    fm, Fc, ll = kf.filter(y)

    # Convert to numpy for assertions
    fm_np = np.asarray(fm).squeeze()
    true_np = x[1:].squeeze()

    # Basic shape checks
    assert fm_np.shape == true_np.shape

    # RMSE should be reasonably small (threshold chosen by noise levels)
    rmse = np.sqrt(np.mean((fm_np - true_np) ** 2))
    assert rmse < 1.0, f"RMSE too large: {rmse}"

    # Log-likelihood should be finite
    assert np.isfinite(float(ll))
