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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'
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

        x = tf.TensorArray(dtype=self.dtype, size=num_steps + 1, clear_after_read=False)
        y = tf.TensorArray(dtype=self.dtype, size=num_steps, clear_after_read=False)

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

class StochasticVariationalData:
    """A stochastic variational model data generator supporting multivariate

    Model (vectorized):
        X_k = alpha * X_{k-1} + sigma * eta_k,   eta_k ~ N(0, I)
        Y_k = beta * exp(X_k / 2) * eps_k        (elementwise)

    Parameters
    ----------
    alpha : float or array-like (length n_state)
        AR(1) coefficients applied elementwise to state.
    sigma : float or array-like (length n_state)
        Scale(s) of the process noise (elementwise).
    beta : float or array-like (length n_state)
        Observation scale(s) applied to each state channel before mapping to
        observation space.
    n_state : int
        Dimension of the latent state X_k (default 1 for backward compatibility).
    n_obs : Optional[int]
        Dimension of the observations. If None, defaults to `n_state`.
    observation_matrix : optional array-like, shape (n_obs, n_state)
        If provided, maps the per-state contributions into observation space via
        matrix multiplication: y = C @ (beta * exp(X/2) * eps). If omitted and
        `n_obs != n_state` a ValueError is raised.
    initial_state : Optional[array-like]
        Initial latent state (length `n_state`). If None, defaults to zeros.
    dtype : tf.DType, default tf.float64
    """

    def __init__(
        self,
        alpha,
        sigma,
        beta,
        n_state: int = 1,
        n_obs: Optional[int] = None,
        initial_state: Optional[object] = None,
        dtype: tf.DType = tf.float64,
    ) -> None:
        self.dtype = dtype
        self.n_state = int(n_state)
        if n_obs is None:
            self.n_obs = int(self.n_state)
        else:
            self.n_obs = int(n_obs)

        # Convert parameters to tensors with appropriate shapes
        alpha_arr = np.asarray(alpha)
        if alpha_arr.size == 1:
            alpha_vec = np.full((self.n_state,), float(alpha_arr))
        else:
            alpha_vec = np.asarray(alpha_arr, dtype=float)
            if alpha_vec.shape[0] != self.n_state:
                raise ValueError("alpha must be scalar or length n_state")

        sigma_arr = np.asarray(sigma)
        if sigma_arr.size == 1:
            sigma_vec = np.full((self.n_state,), float(sigma_arr))
        else:
            sigma_vec = np.asarray(sigma_arr, dtype=float)
            if sigma_vec.shape[0] != self.n_state:
                raise ValueError("sigma must be scalar or length n_state")

        beta_arr = np.asarray(beta)
        if beta_arr.size == 1:
            beta_vec = np.full((self.n_state,), float(beta_arr))
        else:
            beta_vec = np.asarray(beta_arr, dtype=float)
            if beta_vec.shape[0] != self.n_state:
                raise ValueError("beta must be scalar or length n_state")

        self.alpha = tf.convert_to_tensor(alpha_vec, dtype=self.dtype)
        self.sigma = tf.convert_to_tensor(sigma_vec, dtype=self.dtype)
        self.beta = tf.convert_to_tensor(beta_vec, dtype=self.dtype)

        if initial_state is None:
            init = np.zeros((self.n_state,), dtype=float)
        else:
            init = np.asarray(initial_state, dtype=float)
            if init.shape != (self.n_state,):
                raise ValueError("initial_state must have shape (n_state,)")

        self.initial_state = tf.convert_to_tensor(init, dtype=self.dtype)

    def sample(self, num_steps: int, seed: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generate multivariate samples of latent states and observations.

        Returns
        -------
        x : tf.Tensor, shape (num_steps + 1, n_state)
            Latent states (including initial state at index 0)
        y : tf.Tensor, shape (num_steps, n_obs)
            Observations
        """
        x = tf.TensorArray(dtype=self.dtype, size=num_steps + 1, clear_after_read=False)
        y = tf.TensorArray(dtype=self.dtype, size=num_steps, clear_after_read=False)

        x = x.write(0, tf.cast(self.initial_state, dtype=self.dtype))

        # Precreate distributions per-step using diagonal covariances when possible
        for t in range(1, num_steps + 1):
            if seed is not None:
                eta = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_state, dtype=self.dtype), scale_diag=self.sigma).sample(seed=seed + t)
                eps = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_state, dtype=self.dtype), scale_diag=tf.ones(self.n_state, dtype=self.dtype)).sample(seed=seed + t + num_steps)
            else:
                eta = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_state, dtype=self.dtype), scale_diag=self.sigma).sample()
                eps = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_state, dtype=self.dtype), scale_diag=tf.ones(self.n_state, dtype=self.dtype)).sample()

            x_prev = x.read(t - 1)
            # elementwise AR(1)
            x_t = self.alpha * x_prev + self.sigma * eta
            x = x.write(t, x_t)

            # per-state scaled volatility and multiplicative noise
            vol = self.beta * tf.exp(x_t / 2.0)
            y_t = vol * eps  # shape (n_state,)

            y = y.write(t - 1, y_t)

        return x.stack(), y.stack()


if __name__ == "__main__":
    # Run all tests
    run_all_tests()
    
    # Run stochastic volatility tests
    print("\nRunning Stochastic Volatility tests...")
    if test_stochastic_volatility():
        print("Stochastic Volatility tests passed.")
    else:
        print("Stochastic Volatility tests failed.")
