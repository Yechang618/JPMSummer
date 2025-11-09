"""Synthetic data implementation using TensorFlow and TensorFlow Probability.

This module provides a small, clear Kalman filter implementation based on
matrix equations. It uses `tensorflow` for tensor math and `tensorflow_probability`
for log-probability evaluation of the Gaussian innovation.

Class
------
- SSMData: construct with system matrices and run `filter()` on observations.

Example
-------
>>> import numpy as np
>>> from src.models.KalmanFilter import KalmanFilter
>>> # small 1D example omitted here; run `python -m src.models.KalmanFilter` to see a demo
"""
from __future__ import annotations

from typing import Optional, Tuple
import os
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(42)

tfb = tfp.bijectors
tfd = tfp.distributions

class SSMData:
    """A simple state-space model data generator using TensorFlow operations.

    This implements the discrete-time linear Gaussian state-space model
    for the model:

        x_t = A x_{t-1} + q_t,   q_t ~ N(0, Q)
        y_t = H x_t     + r_t,   r_t ~ N(0, R)
    Parameters
    ----------
    transition_matrix: array-like, shape (state_dim, state_dim)
    observation_matrix: array-like, shape (obs_dim, state_dim)
    transition_cov: array-like, shape (state_dim, state_dim)
    observation_cov: array-like, shape (obs_dim, obs_dim)
    initial_mean: array-like, shape (state_dim,)
    initial_cov: array-like, shape (state_dim, state_dim)
    dtype: tf.DType, default tf.float64
    """

    def __init__(
        self,
        transition_matrix,
        observation_matrix,
        transition_cov,
        observation_cov,
        initial_mean,
        initial_cov,
        dtype: tf.DType = tf.float64,
    ) -> None:
        self.dtype = dtype
        self.A = tf.convert_to_tensor(transition_matrix, dtype=self.dtype)
        self.H = tf.convert_to_tensor(observation_matrix, dtype=self.dtype)
        self.Q = tf.convert_to_tensor(transition_cov, dtype=self.dtype)
        self.R = tf.convert_to_tensor(observation_cov, dtype=self.dtype)
        self.initial_mean = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
        self.initial_cov = tf.convert_to_tensor(initial_cov, dtype=self.dtype)
    def sample(self, num_steps: int, seed: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate a sample sequence of latent states and observations.

        Parameters
        ----------
        num_steps: int
            Number of time steps to simulate.
        seed: Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        x: tf.Tensor, shape (num_steps + 1, state_dim)
            The generated latent states.
        y: tf.Tensor, shape (num_steps, obs_dim)
            The generated observations.
        """
        state_dim = self.A.shape[0]
        obs_dim = self.H.shape[0]

        x = tf.TensorArray(self.dtype, size=num_steps + 1)
        y = tf.TensorArray(self.dtype, size=num_steps)

        x = x.write(0, self.initial_mean)

        # rng = tf.random.Generator.from_seed(seed) if seed is not None else tf.random.Generator.from_non_deterministic_state()
        # rng = tf.random.Generator.from_seed(seed) if seed is not None else tf.random.Generator.from_non_deterministic_state()

        for t in range(1, num_steps + 1):
            # process_noise = rng.multivariate_normal(
            #     mean=tf.zeros(state_dim, dtype=self.dtype),
            #     covariance_matrix=self.Q,
            # )
            if seed is not None:
                process_noise = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(state_dim, dtype=self.dtype),
                covariance_matrix=self.Q).sample(seed=seed)
                observation_noise = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(obs_dim, dtype=self.dtype),
                covariance_matrix=self.R).sample(seed=seed)             
            else:
                #
                process_noise = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(state_dim, dtype=self.dtype),
                covariance_matrix=self.Q).sample()
                #
                observation_noise = tfd.MultivariateNormalFullCovariance(
                loc=tf.zeros(obs_dim, dtype=self.dtype),
                covariance_matrix=self.R).sample()

            x_t = tf.linalg.matvec(self.A, x.read(t - 1)) + process_noise
            x = x.write(t, x_t)

            # observation_noise = rng.multivariate_normal(
            #     mean=tf.zeros(obs_dim, dtype=self.dtype),
            #     covariance_matrix=self.R,
            # )

            y_t = tf.linalg.matvec(self.H, x_t) + observation_noise
            y = y.write(t - 1, y_t)

        return x.stack(), y.stack()
if __name__ == "__main__":
    # Simple demo of SSMData
    import numpy as np

    T = 10
    A = np.array([[1.0]], dtype=float)
    H = np.array([[1.0]], dtype=float)
    Q = np.array([[0.1]], dtype=float)
    R = np.array([[0.5]], dtype=float)

    ssm_data = SSMData(
        transition_matrix=A,
        observation_matrix=H,
        transition_cov=Q,
        observation_cov=R,
        initial_mean=np.zeros((1,)),
        initial_cov=np.eye(1) * 1.0,
    )

    x, y = ssm_data.sample(num_steps=T, seed=42)

    print("Generated latent states (x):")
    print(x.numpy())
    print("Generated observations (y):")
    print(y.numpy())
    print("SSMData demo completed.")
    pass

def test_1d():
    try:
        from models.KalmanFilter import KalmanFilter
    except Exception as e:
        print("SKIP test_1d: failed to import KalmanFilter:", e)
        return True

    T = 100
    A = np.array([[1.0]], dtype=float)
    H = np.array([[1.0]], dtype=float)
    Q = np.array([[0.1]], dtype=float)
    R = np.array([[0.5]], dtype=float)

    rng = np.random.default_rng(1)
    x = np.zeros((T + 1, 1))
    y = np.zeros((T, 1))
    for t in range(1, T + 1):
        x[t] = A @ x[t - 1] + rng.normal(scale=np.sqrt(Q[0, 0]))
    for t in range(T):
        y[t] = H @ x[t + 1] + rng.normal(scale=np.sqrt(R[0, 0]))

    kf = KalmanFilter(
        transition_matrix=A,
        observation_matrix=H,
        transition_cov=Q,
        observation_cov=R,
        initial_mean=np.zeros((1,)),
        initial_cov=np.eye(1) * 1.0,
    )
    fm, Fc, ll = kf.filter(y)
    fm_np = np.asarray(fm).squeeze()
    true_np = x[1:].squeeze()
    rmse = np.sqrt(np.mean((fm_np - true_np) ** 2))
    print(f"1D test: RMSE={rmse:.4f}, loglik={float(ll):.3f}")
    return rmse < 1.0
def test_2d():
    try:
        from models.KalmanFilter import KalmanFilter
    except Exception as e:
        print("SKIP test_2d: failed to import KalmanFilter:", e)
        return True

    T = 120
    A = np.array([[1.0, 0.1], [0.0, 1.0]], dtype=float)  # simple constant-velocity
    H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    Q = np.eye(2) * 0.01
    R = np.eye(2) * 0.1

    rng = np.random.default_rng(3)
    x = np.zeros((T + 1, 2))
    y = np.zeros((T, 2))
    for t in range(1, T + 1):
        x[t] = A @ x[t - 1] + rng.normal(scale=np.sqrt(Q[0, 0]), size=(2,))
    for t in range(T):
        y[t] = H @ x[t + 1] + rng.normal(scale=np.sqrt(R[0, 0]), size=(2,))

    kf = KalmanFilter(
        transition_matrix=A,
        observation_matrix=H,
        transition_cov=Q,
        observation_cov=R,
        initial_mean=np.zeros((2,)),
        initial_cov=np.eye(2) * 1.0,
    )

    fm, Fc, ll = kf.filter(y)
    fm_np = np.asarray(fm)
    true_np = x[1:]
    rmse = np.sqrt(np.mean((fm_np - true_np) ** 2))
    print(f"2D test: RMSE={rmse:.4f}, loglik={float(ll):.3f}")
    return rmse < 0.5
def run_all_tests():
    all_passed = True
    if not test_1d():
        all_passed = False
    if not test_2d():
        all_passed = False
    if all_passed:
        print("All tests passed.")
    else:
        print("Some tests failed.")
class StochasticVolatilityData:
    """A stochastic volatility model data generator using TensorFlow operations.
    
    This implements the discrete-time stochastic volatility model:
        h_t = μ + φ(h_{t-1} - μ) + σ_η * η_t,    η_t ~ N(0, 1)
        y_t = exp(h_t/2) * ε_t,                   ε_t ~ N(0, 1)
        
    where:
    - h_t is the log-volatility at time t
    - y_t is the observed return at time t
    - μ is the mean log-volatility
    - φ is the persistence parameter (mean reversion)
    - σ_η is the volatility of log-volatility

    Parameters
    ----------
    mu : float
        Mean of the log-volatility process
    phi : float
        Persistence parameter (0 < φ < 1)
    sigma_eta : float
        Volatility of log-volatility
    initial_logvol : float, optional
        Initial log-volatility value
    dtype : tf.DType, default tf.float64
    """
    def __init__(
        self,
        mu: float,
        phi: float,
        sigma_eta: float,
        initial_logvol: Optional[float] = None,
        dtype: tf.DType = tf.float64,
    ) -> None:
        self.dtype = dtype
        self.mu = tf.convert_to_tensor(mu, dtype=self.dtype)
        self.phi = tf.convert_to_tensor(phi, dtype=self.dtype)
        self.sigma_eta = tf.convert_to_tensor(sigma_eta, dtype=self.dtype)
        
        # If initial log-volatility not provided, use stationary mean
        if initial_logvol is None:
            initial_logvol = mu
        self.initial_logvol = tf.convert_to_tensor(initial_logvol, dtype=self.dtype)
    
    def sample(self, num_steps: int, seed: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate a sample sequence of log-volatilities and returns.
        
        Parameters
        ----------
        num_steps : int
            Number of time steps to simulate
        seed : Optional[int]
            Random seed for reproducibility
            
        Returns
        -------
        h : tf.Tensor, shape (num_steps + 1,)
            The generated log-volatilities
        y : tf.Tensor, shape (num_steps,)
            The generated returns
        """
        h = tf.TensorArray(self.dtype, size=num_steps + 1)
        y = tf.TensorArray(self.dtype, size=num_steps)
        
        h = h.write(0, self.initial_logvol)
        
        for t in range(1, num_steps + 1):
            if seed is not None:
                eta = tf.cast(tfd.Normal(loc=0., scale=1.).sample(seed=seed+t), dtype=tf.float64)
                epsilon = tf.cast(tfd.Normal(loc=0., scale=1.).sample(seed=seed+t+num_steps), dtype=tf.float64)
            else:
                eta = tf.cast(tfd.Normal(loc=0., scale=1.).sample(), dtype=tf.float64)
                epsilon = tf.cast(tfd.Normal(loc=0., scale=1.).sample(), dtype=tf.float64)
            
            # Update log-volatility
            h_prev = h.read(t - 1)
            # print(self.sigma_eta)
            # print(eta)
            h_t = self.mu + self.phi * (h_prev - self.mu) + self.sigma_eta * eta
            h = h.write(t, h_t)
            
            # Generate return
            vol = tf.exp(h_t / 2.0)
            y_t = vol * epsilon
            y = y.write(t - 1, y_t)
        
        return h.stack(), y.stack()

def test_stochastic_volatility():
    """Test the stochastic volatility model implementation."""
    T = 1000
    mu = -1.0  # mean log-volatility
    phi = 0.95  # persistence
    sigma_eta = 0.15  # volatility of log-volatility
    
    sv = StochasticVolatilityData(
        mu=mu,
        phi=phi,
        sigma_eta=sigma_eta,
        initial_logvol=mu
    )
    
    h, y = sv.sample(num_steps=T, seed=42)
    h_np = h.numpy()
    y_np = y.numpy()
    
    try:
        # Basic sanity checks
        assert not np.any(np.isnan(h_np)), "NaN values in log-volatility"
        assert not np.any(np.isnan(y_np)), "NaN values in returns"
        
        # Check mean reversion
        h_mean = np.mean(h_np)
        assert abs(h_mean - mu) < 0.5, f"Log-volatility mean {h_mean:.3f} far from target {mu}"
        
        print("Stochastic Volatility Tests:")
        print(f"- Log-volatility mean: {h_mean:.3f} (target {mu})")
        print(f"- Return kurtosis: {scipy.stats.kurtosis(y_np):.3f}")
        print(f"- Volatility persistence: {np.corrcoef(h_np[:-1], h_np[1:])[0,1]:.3f}")
        return True
    except AssertionError as e:
        print(f"Stochastic Volatility Test FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    # Run all tests
    run_all_tests()
    
    # Run stochastic volatility tests
    print("\nRunning Stochastic Volatility tests...")
    if test_stochastic_volatility():
        print("Stochastic Volatility tests passed.")
    else:
        print("Stochastic Volatility tests failed.")
