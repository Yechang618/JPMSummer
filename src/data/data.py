# ./src/data/data.py
"""Unified synthetic data generators for state-space models (SSMs)."""

from __future__ import annotations
from typing import Optional, Tuple, Union
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import multivariate_t

# Suppress TensorFlow warnings
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

tfd = tfp.distributions


class BaseSSM:
    """Base class for state-space model data generators."""
    def sample(self, num_steps: int, seed: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError


class LinearGaussianSSM(BaseSSM):
    r"""Linear Gaussian state-space model:

        x_t = A x_{t-1} + q_t,   q_t ~ N(0, Q)
        y_t = H x_t     + r_t,   r_t ~ N(0, R)

    Parameters
    ----------
    transition_matrix : array-like, shape (state_dim, state_dim)
    observation_matrix : array-like, shape (obs_dim, state_dim)
    transition_cov : array-like, shape (state_dim, state_dim)
    observation_cov : array-like, shape (obs_dim, obs_dim)
    initial_mean : array-like, shape (state_dim,)
    initial_cov : array-like, shape (state_dim, state_dim)
    dtype : tf.DType, default tf.float64
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
        state_dim = tf.shape(self.A)[0]
        obs_dim = tf.shape(self.H)[0]

        x = tf.TensorArray(self.dtype, size=num_steps + 1, clear_after_read=False)
        y = tf.TensorArray(self.dtype, size=num_steps, clear_after_read=False)

        x = x.write(0, self.initial_mean)

        for t in range(1, num_steps + 1):
            # Sample process noise
            if seed is not None:
                q_noise = tfd.MultivariateNormalFullCovariance(
                    loc=tf.zeros(state_dim, dtype=self.dtype),
                    covariance_matrix=self.Q
                ).sample(seed=seed + t)
                r_noise = tfd.MultivariateNormalFullCovariance(
                    loc=tf.zeros(obs_dim, dtype=self.dtype),
                    covariance_matrix=self.R
                ).sample(seed=seed + t + num_steps)
            else:
                q_noise = tfd.MultivariateNormalFullCovariance(
                    loc=tf.zeros(state_dim, dtype=self.dtype),
                    covariance_matrix=self.Q
                ).sample()
                r_noise = tfd.MultivariateNormalFullCovariance(
                    loc=tf.zeros(obs_dim, dtype=self.dtype),
                    covariance_matrix=self.R
                ).sample()

            x_prev = x.read(t - 1)
            x_t = tf.linalg.matvec(self.A, x_prev) + q_noise
            x = x.write(t, x_t)

            y_t = tf.linalg.matvec(self.H, x_t) + r_noise
            y = y.write(t - 1, y_t)

        return x.stack(), y.stack()


class StochasticVolatilityModel(BaseSSM):
    r"""Stochastic volatility model (multivariate):

        X_k = alpha * X_{k-1} + sigma * eta_k,   eta_k ~ N(0, I)
        Y_k = beta * exp(X_k / 2) * eps_k,       eps_k ~ N(0, I)

    Parameters
    ----------
    alpha : float or array-like (length n_state)
    sigma : float or array-like (length n_state)
    beta : float or array-like (length n_state)
    n_state : int
    n_obs : Optional[int], defaults to n_state
    initial_state : Optional[array-like], shape (n_state,)
    dtype : tf.DType
    """

    def __init__(
        self,
        alpha: Union[float, list, np.ndarray],
        sigma: Union[float, list, np.ndarray],
        beta: Union[float, list, np.ndarray],
        n_state: int = 1,
        n_obs: Optional[int] = None,
        initial_state: Optional[np.ndarray] = None,
        dtype: tf.DType = tf.float64,
    ) -> None:
        self.dtype = dtype
        self.n_state = int(n_state)
        self.n_obs = int(n_obs) if n_obs is not None else self.n_state

        if self.n_obs != self.n_state:
            raise ValueError("Currently only supports n_obs == n_state.")

        def _broadcast(param, name):
            arr = np.asarray(param, dtype=float)
            if arr.size == 1:
                return np.full(self.n_state, float(arr))
            elif arr.shape[0] == self.n_state:
                return arr
            else:
                raise ValueError(f"{name} must be scalar or length {self.n_state}")

        self.alpha = tf.convert_to_tensor(_broadcast(alpha, "alpha"), dtype=self.dtype)
        self.sigma = tf.convert_to_tensor(_broadcast(sigma, "sigma"), dtype=self.dtype)
        self.beta = tf.convert_to_tensor(_broadcast(beta, "beta"), dtype=self.dtype)

        init = np.zeros(self.n_state) if initial_state is None else np.asarray(initial_state, dtype=float)
        if init.shape != (self.n_state,):
            raise ValueError("initial_state must have shape (n_state,)")
        self.initial_state = tf.convert_to_tensor(init, dtype=self.dtype)

    def sample(self, num_steps: int, seed: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.TensorArray(self.dtype, size=num_steps + 1, clear_after_read=False)
        y = tf.TensorArray(self.dtype, size=num_steps, clear_after_read=False)

        x = x.write(0, self.initial_state)

        for t in range(1, num_steps + 1):
            if seed is not None:
                eta = tfd.MultivariateNormalDiag(
                    loc=tf.zeros(self.n_state, dtype=self.dtype),
                    scale_diag=self.sigma
                ).sample(seed=seed + t)
                eps = tfd.MultivariateNormalDiag(
                    loc=tf.zeros(self.n_state, dtype=self.dtype),
                    scale_diag=tf.ones(self.n_state, dtype=self.dtype)
                ).sample(seed=seed + t + num_steps)
            else:
                eta = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_state), scale_diag=self.sigma).sample()
                eps = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_state), scale_diag=tf.ones(self.n_state)).sample()

            x_prev = x.read(t - 1)
            x_t = self.alpha * x_prev + eta
            x = x.write(t, x_t)

            vol = self.beta * tf.exp(x_t / 2.0)
            y_t = vol * eps
            y = y.write(t - 1, y_t)

        return x.stack(), y.stack()


class RangeBearingSSM(BaseSSM):
    r"""2D constant-velocity model with nonlinear range-bearing observations.

    State: [x, y, vx, vy]
    Dynamics: x_t = A x_{t-1} + q_t
    Observation: [r, b] = [sqrt(x^2+y^2), atan2(y,x)] + noise

    Parameters
    ----------
    dt : float
    process_noise_std : float
    range_std : float
    bearing_std : float
    initial_state : array-like, shape (4,)
    dtype : tf.DType
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 0.1,
        range_std: float = 0.5,
        bearing_std: float = 0.05,
        initial_state: Optional[list] = None,
        dtype: tf.DType = tf.float64,
    ):
        self.dtype = dtype
        self.dt = float(dt)
        self.range_std = float(range_std)
        self.bearing_std = float(bearing_std)

        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        self.A = tf.constant(A, dtype=self.dtype)

        q = process_noise_std ** 2
        Q = np.zeros((4, 4))
        Q[:2, :2] = q * (dt ** 3 / 3) * np.eye(2)
        Q[:2, 2:] = q * (dt ** 2 / 2) * np.eye(2)
        Q[2:, :2] = Q[:2, 2:].T
        Q[2:, 2:] = q * dt * np.eye(2)
        self.Q = tf.constant(Q, dtype=self.dtype)
        self.R = tf.constant([[range_std ** 2, 0], [0, bearing_std ** 2]], dtype=self.dtype)

        init = [0.0, 0.0, 1.0, 0.5] if initial_state is None else initial_state
        self.initial_state = tf.constant(init, dtype=self.dtype)

    @staticmethod
    def h_tf(x: tf.Tensor) -> tf.Tensor:
        x_pos, y_pos = x[..., 0], x[..., 1]
        r = tf.sqrt(x_pos ** 2 + y_pos ** 2)
        b = tf.math.atan2(y_pos, x_pos)
        return tf.stack([r, b], axis=-1)

    def sample(self, num_steps: int, seed: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.TensorArray(self.dtype, size=num_steps + 1, clear_after_read=False)
        y = tf.TensorArray(self.dtype, size=num_steps, clear_after_read=False)

        x = x.write(0, self.initial_state)

        chol_Q = tf.linalg.cholesky(self.Q)
        chol_R = tf.linalg.cholesky(self.R)

        for t in range(1, num_steps + 1):
            if seed is not None:
                q_noise = tfd.MultivariateNormalTriL(
                    loc=tf.zeros(4, dtype=self.dtype), scale_tril=chol_Q
                ).sample(seed=seed + t)
                r_noise = tfd.MultivariateNormalTriL(
                    loc=tf.zeros(2, dtype=self.dtype), scale_tril=chol_R
                ).sample(seed=seed + t + num_steps)
            else:
                q_noise = tfd.MultivariateNormalTriL(loc=tf.zeros(4), scale_tril=chol_Q).sample()
                r_noise = tfd.MultivariateNormalTriL(loc=tf.zeros(2), scale_tril=chol_R).sample()

            x_prev = x.read(t - 1)
            x_t = tf.linalg.matvec(self.A, x_prev) + q_noise
            x = x.write(t, x_t)

            y_true = self.h_tf(x_t)
            y_t = y_true + r_noise
            y = y.write(t - 1, y_t)

        return x.stack(), y.stack()


class MultidimNonGaussianSSM:
    def __init__(self, dim=20, F=4.0, dt=0.005, df_process=3.0, df_obs=3.0, seed=42):
        self.dim = dim
        self.F = F
        self.dt = dt
        self.df_process = df_process
        self.df_obs = df_obs
        self.rng = np.random.default_rng(seed)
        self.obs_indices = np.arange(dim)
        
    def lorenz96_step(self, x):
        dx = np.zeros_like(x)
        for i in range(self.dim):
            im1 = (i - 1) % self.dim
            im2 = (i - 2) % self.dim
            ip1 = (i + 1) % self.dim
            dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + self.F
        
        x_new = x + self.dt * dx
        # Clip to prevent numerical overflow
        return np.clip(x_new, -20.0, 20.0)
    
    def sample_student_t_noise(self, size, df, scale=0.1):
        """Sample Student-t noise with given degrees of freedom."""
        if df == np.inf:
            return self.rng.normal(scale=scale, size=size)
        else:
            # Use multivariate_t from scipy
            return multivariate_t.rvs(
                loc=np.zeros(size), 
                shape=np.eye(size) * (scale ** 2),
                df=df,
                size=1,
                random_state=self.rng
            ).reshape(size)
        
    def generate_data(self, T=100):
        x = np.zeros((T + 1, self.dim))
        y = np.zeros((T, self.dim))
        
        # Bounded initial conditions
        x[0] = self.rng.uniform(-1.0, 1.0, self.dim)
        
        for t in range(1, T + 1):
            q_noise = self.sample_student_t_noise(self.dim, self.df_process, scale=0.1)
            x[t] = self.lorenz96_step(x[t - 1]) + q_noise
            
            # Observation
            obs_nonlinear = np.zeros(self.dim)
            for i in range(self.dim):
                if i % 2 == 0:
                    obs_nonlinear[i] = x[t, i] ** 2
                else:
                    obs_nonlinear[i] = np.sin(x[t, i])
            
            r_noise = self.sample_student_t_noise(self.dim, self.df_obs, scale=0.2)
            y[t - 1] = obs_nonlinear + r_noise
            
        return x[1:], y
class Lorenz96NonGaussianSSM:
    r"""Non-Gaussian SSM based on Lorenz-96 dynamics with Student-t noise.

    State evolves via chaotic Lorenz-96 ODE.
    Observations are nonlinear (even: x^2, odd: sin(x)) with Student-t noise.

    Parameters
    ----------
    dim : int
    F : float (forcing term)
    dt : float (Euler step)
    df_process : float (degrees of freedom for process noise)
    df_obs : float (degrees of freedom for observation noise)
    seed : int
    """

    def __init__(
        self,
        dim: int = 20,
        F: float = 4.0,
        dt: float = 0.005,
        df_process: float = 3.0,
        df_obs: float = 3.0,
        seed: int = 42,
    ):
        self.dim = dim
        self.F = F
        self.dt = dt
        self.df_process = df_process
        self.df_obs = df_obs
        self.rng = np.random.default_rng(seed)

    def lorenz96_step(self, x: np.ndarray) -> np.ndarray:
        dx = np.zeros_like(x)
        for i in range(self.dim):
            im1 = (i - 1) % self.dim
            im2 = (i - 2) % self.dim
            ip1 = (i + 1) % self.dim
            dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + self.F
        return np.clip(x + self.dt * dx, -20.0, 20.0)

    def _sample_student_t(self, size: int, df: float, scale: float) -> np.ndarray:
        if np.isinf(df):
            return self.rng.normal(scale=scale, size=size)
        else:
            return multivariate_t.rvs(
                loc=np.zeros(size),
                shape=np.eye(size) * (scale ** 2),
                df=df,
                size=1,
                random_state=self.rng
            ).reshape(size)

    def generate_data(self, T: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate NumPy arrays (not TensorFlow tensors)."""
        x = np.zeros((T + 1, self.dim))
        y = np.zeros((T, self.dim))

        x[0] = self.rng.uniform(-1.0, 1.0, self.dim)

        for t in range(1, T + 1):
            q = self._sample_student_t(self.dim, self.df_process, scale=0.1)
            x[t] = self.lorenz96_step(x[t - 1]) + q

            obs_nonlinear = np.where(
                np.arange(self.dim) % 2 == 0,
                x[t] ** 2,
                np.sin(x[t])
            )
            r = self._sample_student_t(self.dim, self.df_obs, scale=0.2)
            y[t - 1] = obs_nonlinear + r

        return x[1:], y  # (T, dim), (T, dim)


# =============================================================================
# Utility function (kept for backward compatibility)
# =============================================================================

def generate_svm_data(T: int = 200, alpha: float = 0.95, sigma: float = 0.2, beta: float = 1.0, seed: int = 42):
    """Legacy helper for 1D stochastic volatility model (NumPy only)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T + 1)
    y = np.zeros(T)
    x[0] = 0.0
    for t in range(1, T + 1):
        x[t] = alpha * x[t - 1] + sigma * rng.normal()
        y[t - 1] = beta * np.exp(x[t] / 2) * rng.normal()
    return x[1:], y
