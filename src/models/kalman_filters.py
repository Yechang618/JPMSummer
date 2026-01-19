# ./src/models/kalman_filters.py
"""
Unified Kalman Filter Module

Implements:
- KalmanFilter (linear Gaussian)
- ExtendedKalmanFilter (first-order linearization)
- UnscentedKalmanFilter (sigma-point nonlinear)

All filters use TensorFlow 2 + TFP, avoid deprecated APIs,
and are compatible with TF 2.15.1 + TFP 0.23.0 on Python 3.11.
"""

from __future__ import annotations
import os
import math
from typing import Callable, Optional, Tuple, List
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Disable oneDNN custom ops to avoid numerical variation warnings (optional)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

tfd = tfp.distributions


# =============================================================================
# 1. Linear Kalman Filter (KF)
# =============================================================================

class KalmanFilter:
    r"""Linear Kalman Filter for:
        x_t = A x_{t-1} + q_t,   q_t ~ N(0, Q)
        y_t = H x_t     + r_t,   r_t ~ N(0, R)
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
        verbose: bool = False,
    ):
        self.dtype = dtype
        self.A = tf.convert_to_tensor(transition_matrix, dtype=self.dtype)
        self.H = tf.convert_to_tensor(observation_matrix, dtype=self.dtype)
        self.Q = tf.convert_to_tensor(transition_cov, dtype=self.dtype)
        self.R = tf.convert_to_tensor(observation_cov, dtype=self.dtype)
        self.m0 = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
        self.P0 = tf.convert_to_tensor(initial_cov, dtype=self.dtype)
        self.verbose = verbose

        self.state_dim = int(self.A.shape[0])
        self.obs_dim = int(self.H.shape[0])

        assert self.A.shape == (self.state_dim, self.state_dim)
        assert self.H.shape == (self.obs_dim, self.state_dim)
        assert self.Q.shape == (self.state_dim, self.state_dim)
        assert self.R.shape == (self.obs_dim, self.obs_dim)

        self.cond_history: List[float] = []

    def filter(self, observations: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        observations = tf.convert_to_tensor(observations, dtype=self.dtype)
        T = tf.shape(observations)[0]

        m = tf.identity(self.m0)
        P = tf.identity(self.P0)
        means = tf.TensorArray(self.dtype, size=T)
        covs = tf.TensorArray(self.dtype, size=T)
        total_ll = tf.constant(0.0, dtype=self.dtype)
        I = tf.eye(self.state_dim, dtype=self.dtype)

        for t in tf.range(T):
            y = observations[t]
            # Predict
            m_pred = tf.linalg.matvec(self.A, m)
            P_pred = self.A @ P @ tf.transpose(self.A) + self.Q
            # Update
            y_pred = tf.linalg.matvec(self.H, m_pred)
            S = self.H @ P_pred @ tf.transpose(self.H) + self.R
            
            # Condition number (diagnostic)
            try:
                svals = tf.linalg.svd(S, compute_uv=False)
                cond_S = tf.reduce_max(svals) / (tf.reduce_min(svals) + 1e-12)
                self.cond_history.append(float(cond_S.numpy()))
            except Exception:
                self.cond_history.append(math.nan)

            # Use Cholesky for stable inversion
            chol_S = tf.linalg.cholesky(S)
            S_inv = tf.linalg.cholesky_solve(chol_S, tf.eye(self.obs_dim, dtype=self.dtype))
            K = P_pred @ tf.transpose(self.H) @ S_inv

            innovation = y - y_pred
            m = m_pred + tf.linalg.matvec(K, innovation)
            
            assert K.shape == (self.state_dim, self.obs_dim)
            assert self.R.shape == (self.obs_dim, self.obs_dim)

            # Joseph form for covariance update (numerical stability)            
            KH = K @ self.H
            P = (I - KH) @ P_pred @ tf.transpose(I - KH) + K @ self.R @ tf.transpose(K)

            # ✅ Use MultivariateNormalTriL instead of deprecated FullCovariance
            mvn = tfd.MultivariateNormalTriL(
                loc=y_pred,
                scale_tril=tf.linalg.cholesky(S)
            )
            total_ll += mvn.log_prob(y)

            means = means.write(t, m)
            covs = covs.write(t, P)

        return means.stack(), covs.stack(), total_ll


# =============================================================================
# 2. Extended Kalman Filter (EKF)
# =============================================================================

class ExtendedKalmanFilter:
    r"""Extended Kalman Filter using Jacobian linearization."""

    def __init__(
        self,
        f: Callable[[tf.Tensor], tf.Tensor],
        h: Callable[[tf.Tensor], tf.Tensor],
        Q,
        R,
        initial_mean,
        initial_cov,
        dtype: tf.DType = tf.float64,
        verbose: bool = False,
    ):
        self.f = f
        self.h = h
        self.dtype = dtype
        self.Q = tf.convert_to_tensor(Q, dtype=self.dtype)
        self.R = tf.convert_to_tensor(R, dtype=self.dtype)
        self.m0 = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
        self.P0 = tf.convert_to_tensor(initial_cov, dtype=self.dtype)
        self.verbose = verbose
        
        self.m = tf.identity(self.m0)
        self.P = tf.identity(self.P0)

        self.state_dim = int(self.P0.shape[0])
        self.obs_dim = int(self.R.shape[0])

        assert self.Q.shape == (self.state_dim, self.state_dim)
        assert self.R.shape == (self.obs_dim, self.obs_dim)

        self.rmse_history: List[float] = []
        self.ll_history: List[float] = []

    def _jacobian(self, func: Callable[[tf.Tensor], tf.Tensor], x: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.convert_to_tensor(func(x), dtype=self.dtype)
        J = tape.jacobian(y, x)
        return tf.reshape(J, (tf.shape(y)[0], tf.shape(x)[0]))

    def step(self, observation, true_state=None):
        y = tf.convert_to_tensor(observation, dtype=self.dtype)
        if y.shape.rank == 0:
            y = tf.reshape(y, (1,))

        # Predict
        F = self._jacobian(self.f, self.m)
        m_pred = tf.linalg.matvec(F, self.m)
        P_pred = F @ self.P @ tf.transpose(F) + self.Q

        # Update
        H = self._jacobian(self.h, m_pred)
        y_pred = tf.linalg.matvec(H, m_pred)
        S = H @ P_pred @ tf.transpose(H) + self.R

        chol_S = tf.linalg.cholesky(S)
        K = tf.transpose(tf.linalg.cholesky_solve(chol_S, tf.transpose(P_pred @ tf.transpose(H))))

        innovation = y - y_pred
        self.m = m_pred + tf.linalg.matvec(K, innovation)
        # Safe covariance update: P = P_pred - K @ S @ K^T
        self.P = P_pred - K @ S @ tf.transpose(K)

        # ✅ Use TriL
        mvn = tfd.MultivariateNormalTriL(loc=y_pred, scale_tril=tf.linalg.cholesky(S))
        loglik = mvn.log_prob(y)
        self.ll_history.append(float(loglik.numpy()))

        if true_state is not None:
            rmse = tf.sqrt(tf.reduce_mean(tf.square(self.m - tf.convert_to_tensor(true_state, dtype=self.dtype))))
            self.rmse_history.append(float(rmse.numpy()))

        return self.m, self.P, loglik

    def filter(self, observations, true_states=None):
        self.m = tf.identity(self.m0)
        self.P = tf.identity(self.P0)

        obs = tf.convert_to_tensor(observations, dtype=self.dtype)
        if obs.shape.rank == 1:
            obs = tf.reshape(obs, (-1, max(1, self.obs_dim)))
        T = tf.shape(obs)[0]

        true_ts = None
        if true_states is not None:
            true_ts = tf.convert_to_tensor(true_states, dtype=self.dtype)
            if true_ts.shape.rank == 1:
                true_ts = tf.reshape(true_ts, (-1, max(1, self.state_dim)))

        means = tf.TensorArray(self.dtype, size=T)
        covs = tf.TensorArray(self.dtype, size=T)
        total_ll = tf.constant(0.0, dtype=self.dtype)

        for t in tf.range(T):
            y_t = obs[t]
            true_t = true_ts[t] if true_ts is not None else None
            m, P, ll = self.step(y_t, true_state=true_t)
            total_ll += ll
            means = means.write(t, m)
            covs = covs.write(t, P)

        return means.stack(), covs.stack(), total_ll


# =============================================================================
# 3. Unscented Kalman Filter (UKF)
# =============================================================================

class UnscentedKalmanFilter:
    r"""Unscented Kalman Filter using sigma points."""

    def __init__(
        self,
        f: Callable[[tf.Tensor], tf.Tensor],
        h: Callable[[tf.Tensor], tf.Tensor],
        Q,
        R,
        initial_mean,
        initial_cov,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: Optional[float] = None,
        dtype: tf.DType = tf.float64,
        verbose: bool = False,
    ):
        self.f = f
        self.h = h
        self.dtype = dtype
        self.Q = tf.convert_to_tensor(Q, dtype=self.dtype)
        self.R = tf.convert_to_tensor(R, dtype=self.dtype)
        self.m0 = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
        self.P0 = tf.convert_to_tensor(initial_cov, dtype=self.dtype)
        self.verbose = verbose

        self.state_dim = int(self.P0.shape[0])
        self.obs_dim = int(self.R.shape[0])

        n = float(self.state_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa) if kappa is not None else float(3.0 - n)
        self.lambda_ = (self.alpha ** 2) * (n + self.kappa) - n

        self._calculate_weights()

        self.rmse_history: List[float] = []
        self.ll_history: List[float] = []

    def _calculate_weights(self):
        n = float(self.state_dim)
        lam = self.lambda_
        wm0 = lam / (n + lam)
        w_other = 1.0 / (2.0 * (n + lam))
        wc0 = wm0 + (1.0 - self.alpha ** 2 + self.beta)

        m = 2 * int(n) + 1
        self.wm = tf.concat([[wm0], [w_other] * (m - 1)], axis=0)
        self.wc = tf.concat([[wc0], [w_other] * (m - 1)], axis=0)
        self.wm = tf.cast(self.wm, self.dtype)
        self.wc = tf.cast(self.wc, self.dtype)

    def _generate_sigma_points(self, mean: tf.Tensor, cov: tf.Tensor) -> tf.Tensor:
        n = self.state_dim
        L = tf.linalg.cholesky(cov + 1e-9 * tf.eye(n, dtype=self.dtype))
        gamma = tf.sqrt(tf.constant(n + self.lambda_, dtype=self.dtype)) * L
        sigmas = [mean]
        for i in range(n):
            sigmas.append(mean + gamma[:, i])
        for i in range(n):
            sigmas.append(mean - gamma[:, i])
        return tf.stack(sigmas, axis=0)

    def filter(self, observations, true_states=None):
        obs = tf.convert_to_tensor(observations, dtype=self.dtype)
        if obs.shape.rank == 1:
            obs = tf.reshape(obs, (-1, max(1, self.obs_dim)))
        T = tf.shape(obs)[0]

        true_ts = None
        if true_states is not None:
            true_ts = tf.convert_to_tensor(true_states, dtype=self.dtype)
            if true_ts.shape.rank == 1:
                true_ts = tf.reshape(true_ts, (-1, max(1, self.state_dim)))

        m = tf.identity(self.m0)
        P = tf.identity(self.P0)
        means = tf.TensorArray(self.dtype, size=T)
        covs = tf.TensorArray(self.dtype, size=T)
        total_ll = tf.constant(0.0, dtype=self.dtype)

        for t in tf.range(T):
            y = obs[t]

            # Predict
            chi = self._generate_sigma_points(m, P)
            chi_pred = tf.map_fn(self.f, chi, fn_output_signature=tf.TensorSpec(shape=(self.state_dim,), dtype=self.dtype))
            m_pred = tf.reduce_sum(self.wm[:, None] * chi_pred, axis=0)
            diffs = chi_pred - m_pred
            P_pred = tf.transpose(diffs) @ (tf.reshape(self.wc, [-1, 1]) * diffs) + self.Q

            # Update
            gamma_obs = tf.map_fn(self.h, chi_pred, fn_output_signature=tf.TensorSpec(shape=(self.obs_dim,), dtype=self.dtype))
            y_pred = tf.reduce_sum(self.wm[:, None] * gamma_obs, axis=0)
            diffs_y = gamma_obs - y_pred
            S = tf.transpose(diffs_y) @ (tf.reshape(self.wc, [-1, 1]) * diffs_y) + self.R
            Pxy = tf.transpose(diffs) @ (tf.reshape(self.wc, [-1, 1]) * diffs_y)

            K = Pxy @ tf.linalg.inv(S)
            m = m_pred + tf.linalg.matvec(K, y - y_pred)
            P = P_pred - K @ S @ tf.transpose(K)
            P = 0.5 * (P + tf.transpose(P))  # enforce symmetry

            # ✅ Use TriL
            mvn = tfd.MultivariateNormalTriL(loc=y_pred, scale_tril=tf.linalg.cholesky(S))
            ll = mvn.log_prob(y)
            total_ll += ll
            self.ll_history.append(float(ll.numpy()))

            if true_ts is not None:
                rmse = tf.sqrt(tf.reduce_mean(tf.square(m - true_ts[t])))
                self.rmse_history.append(float(rmse.numpy()))

            means = means.write(t, m)
            covs = covs.write(t, P)

        return means.stack(), covs.stack(), total_ll


# =============================================================================
# 4. Model-Specific Filters (Stochastic Volatility Model)
# =============================================================================

class SVMEKF:
    """EKF for Stochastic Volatility Model using log(y^2) observation."""
    
    def __init__(self, alpha, sigma, beta, initial_mean=0.0, initial_cov=1.0, dtype=np.float64):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.dtype = dtype
        self.m = float(initial_mean)
        self.P = float(initial_cov)

    def step(self, obs_log_y2):
        y_log = float(obs_log_y2)
        # Predict
        m_pred = self.alpha * self.m
        P_pred = (self.alpha ** 2) * self.P + self.sigma ** 2

        # Observation: h(x) = log(beta^2) + x
        log_beta_sq = np.log(self.beta ** 2)
        h_x = log_beta_sq + m_pred
        H = 1.0

        # Approximate measurement noise variance for log(y^2): Var ≈ 4
        R = 4.0
        S = H * P_pred * H + R
        K = P_pred * H / S

        innovation = y_log - h_x
        self.m = m_pred + K * innovation
        self.P = P_pred - K * S * K

        return self.m, self.P


class SVMUKF:
    """UKF for Stochastic Volatility Model using log(y^2) observation."""
    
    def __init__(self, alpha, sigma, beta, initial_mean=0.0, initial_cov=1.0, dtype=np.float64):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.m = float(initial_mean)
        self.P = float(initial_cov)
        self.dtype = dtype

        n = 1
        self.kappa = 0.0
        self.alpha_ukf = 1e-3
        self.beta_ukf = 2.0
        self.lambda_ = (self.alpha_ukf ** 2) * (n + self.kappa) - n

        wm0 = self.lambda_ / (n + self.lambda_)
        wc0 = wm0 + (1 - self.alpha_ukf ** 2 + self.beta_ukf)
        w_other = 1.0 / (2.0 * (n + self.lambda_))
        self.wm = np.array([wm0, w_other, w_other], dtype=dtype)
        self.wc = np.array([wc0, w_other, w_other], dtype=dtype)

    def _gen_sigma_points(self, mean, cov):
        gamma = np.sqrt((1 + self.lambda_) * cov)
        return np.array([mean, mean + gamma, mean - gamma], dtype=self.dtype)

    def step(self, obs_log_y2):
        y_log = float(obs_log_y2)
        # Predict
        sigma_pred = self._gen_sigma_points(self.m, self.P)
        sigma_prop = self.alpha * sigma_pred
        m_pred = np.sum(self.wm * sigma_prop)
        P_pred = np.sum(self.wc * (sigma_prop - m_pred)**2) + self.sigma**2

        # Update
        obs_sigma = np.log(self.beta**2) + sigma_prop  # h(x) = log(beta^2) + x
        y_pred = np.sum(self.wm * obs_sigma)
        Pyy = np.sum(self.wc * (obs_sigma - y_pred)**2) + 4.0  # R=4
        Pxy = np.sum(self.wc * (sigma_prop - m_pred) * (obs_sigma - y_pred))

        K = Pxy / Pyy
        self.m = m_pred + K * (y_log - y_pred)
        self.P = P_pred - K * Pyy * K

        return self.m, self.P