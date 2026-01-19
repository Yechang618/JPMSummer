# ./src/models/particle_filters.py
"""
Particle filter implementations consistent with kalman_filters.py.
Includes:
- Standard Particle Filter (resampling)
- EDH / LEDH flows
- Daum-Huang Exact Flow (Algorithm 1 & 2)
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Use consistent Kalman filters from local module
from .kalman_filters import ExtendedKalmanFilter, UnscentedKalmanFilter

# Environment setup
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

tfd = tfp.distributions


def compute_kalman_gain_robust(P, H, R, jitter=1e-6):
    """Compute Kalman gain K = P H^T (H P H^T + R)^{-1} robustly."""
    S = H @ P @ tf.transpose(H) + R
    S = 0.5 * (S + tf.transpose(S))
    dy = tf.shape(S)[0]
    S_jittered = S + jitter * tf.eye(dy, dtype=S.dtype)
    
    try:
        L = tf.linalg.cholesky(S_jittered)
    except tf.errors.InvalidArgumentError:
        S_jittered = S + 1e-4 * tf.eye(dy, dtype=S.dtype)
        L = tf.linalg.cholesky(S_jittered)

    HPt = H @ tf.transpose(P)
    X = tf.linalg.triangular_solve(L, HPt, lower=True)
    return tf.transpose(X)


def safe_inv(matrix, jitter=1e-6):
    """Robust matrix inversion with consistent regularization."""
    dim = tf.shape(matrix)[0]
    dtype = matrix.dtype
    
    # Add jitter and use Cholesky
    jitter_mat = jitter * tf.eye(dim, dtype=dtype)
    try:
        L = tf.linalg.cholesky(matrix + jitter_mat)
        return tf.linalg.cholesky_solve(L, tf.eye(dim, dtype=dtype))
    except tf.errors.InvalidArgumentError:
        # Fallback with stronger jitter
        L = tf.linalg.cholesky(matrix + 1e-6 * tf.eye(dim, dtype=dtype))
        return tf.linalg.cholesky_solve(L, tf.eye(dim, dtype=dtype))


# =============================================================================
# 1. Standard Particle Filter (with resampling)
# =============================================================================

class StandardParticleFilter:
    def __init__(self, f, Q, initial_mean, initial_cov, num_particles=3000, seed=42, beta=1.0):
        self.f = f
        self.Q = Q
        self.beta = beta
        self.num_particles = num_particles
        self.rng = np.random.default_rng(seed)
        # Normalize inputs to arrays
        if np.isscalar(initial_mean):
            self.initial_mean = np.array([float(initial_mean)])
            if np.isscalar(initial_cov):
                self.initial_cov = np.array([[float(initial_cov)]])
            else:
                self.initial_cov = np.array(initial_cov).reshape(1, 1)
        else:
            self.initial_mean = np.array(initial_mean)
            self.initial_cov = np.array(initial_cov)
        
                
        self.particles = self.rng.multivariate_normal(
            mean=self.initial_mean,
            cov=self.initial_cov,
            size=num_particles
        )
        self.weights = np.ones(num_particles) / num_particles
        self.ess_history = []

    def predict(self):
        if np.isscalar(self.Q):
            # 1D case: scalar variance
            noise = self.rng.normal(scale=np.sqrt(self.Q), size=self.num_particles)
        else:
            # Multivariate case: covariance matrix
            noise = self.rng.multivariate_normal(
                mean=np.zeros(self.Q.shape[0]),
                cov=self.Q,
                size=self.num_particles
            )
        self.particles = self.f(self.particles) + noise

    def update(self, observation):
        # observation: (d,)
        # Compute predicted observation for each particle
        if self.particles.ndim == 1:
            particles = self.particles[None, :]  # (1, d)
        else:
            particles = self.particles  # (N, d)

        # Observation model: [x1^2, sin(x2), x3^2, ...]
        obs_pred = np.empty_like(particles)
        for i in range(particles.shape[1]):
            if i % 2 == 0:
                obs_pred[:, i] = particles[:, i] ** 2
            else:
                obs_pred[:, i] = np.sin(particles[:, i])

        # Assume R = I (Gaussian noise); extend to Student-t if needed
        diff = observation - obs_pred  # (N, d)
        log_lik = -0.5 * np.sum(diff ** 2, axis=1)  # log N(obs | h(x), I)

        # Update weights
        max_log_lik = np.max(log_lik)
        self.weights *= np.exp(log_lik - max_log_lik)
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

        # ESS
        ess = 1.0 / np.sum(self.weights ** 2)
        self.ess_history.append(ess)

        # Resample if needed
        if ess < self.num_particles / 2:
            u0 = self.rng.random() / self.num_particles
            positions = (u0 + np.arange(self.num_particles)) % 1
            cumsum_weights = np.cumsum(self.weights)
            indices = np.searchsorted(cumsum_weights, positions)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        mean = np.sum(self.weights[:, None] * self.particles, axis=0)
        return mean

    def filter(self, observations):
        means = []
        for y in observations:
            self.predict()
            self.update(y)
            m = self.estimate()
            means.append(m)
        return np.array(means)


# =============================================================================
# 2. EDH Flow (Exact Daum-Huang, global covariance)
# =============================================================================

class EDHFlow:
    def __init__(self, f, Q, initial_mean, initial_cov, num_particles=2000, seed=42, beta=1.0):
        self.f = f
        self.Q = Q
        self.beta = beta
        self.num_particles = num_particles
        self.rng = np.random.default_rng(seed)
        # Normalize inputs to arrays
        if np.isscalar(initial_mean):
            self.initial_mean = np.array([float(initial_mean)])
            if np.isscalar(initial_cov):
                self.initial_cov = np.array([[float(initial_cov)]])
            else:
                self.initial_cov = np.array(initial_cov).reshape(1, 1)
        else:
            self.initial_mean = np.array(initial_mean)
            self.initial_cov = np.array(initial_cov)
                
        self.particles = self.rng.multivariate_normal(
            mean=initial_mean,
            cov=initial_cov,
            size=num_particles
        )
        self.flow_history = []

    def predict(self):
        if np.isscalar(self.Q):
            # 1D case: scalar variance
            noise = self.rng.normal(scale=np.sqrt(self.Q), size=self.num_particles)
        else:
            # Multivariate case: covariance matrix
            noise = self.rng.multivariate_normal(
                mean=np.zeros(self.Q.shape[0]),
                cov=self.Q,
                size=self.num_particles
            )
        self.particles = self.f(self.particles) + noise

    def compute_flow_edh(self, observation):
        x_tf = tf.Variable(self.particles.astype(np.float64))
        y_tf = tf.constant(observation.astype(np.float64), dtype=tf.float64)

        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            # Observation model: [x1^2, sin(x2), x3^2, ...]
            y_pred_list = []
            for i in range(x_tf.shape[1]):
                if i % 2 == 0:
                    y_pred_list.append(tf.square(x_tf[:, i]))
                else:
                    y_pred_list.append(tf.sin(x_tf[:, i]))
            y_pred = tf.stack(y_pred_list, axis=1)  # (N, d)

            diff = y_tf - y_pred
            log_lik = -0.5 * tf.reduce_sum(tf.square(diff), axis=1)  # (N,)

        grad_log_lik = tape.gradient(log_lik, x_tf)  # (N, d) TensorFlow tensor
        grad_log_lik_np = grad_log_lik.numpy()      # Convert to NumPy array

        cov = np.cov(self.particles.T)  # (d, d)
        flow = 0.5 * (cov @ grad_log_lik_np.T).T   # (N, d)
        return flow

    def update(self, observation):
        flow = self.compute_flow_edh(observation)
        self.particles += flow
        self.flow_history.append(np.mean(np.abs(flow)))

    def estimate(self):
        return np.mean(self.particles, axis=0)

    def filter(self, observations):
        means = []
        for y in observations:
            self.predict()
            self.update(y)
            m = self.estimate()
            means.append(m)
        return np.array(means)


# =============================================================================
# 3. LEDH Flow (Local Linearization)
# =============================================================================

class LEDHFlow:
    def __init__(self, f, Q, initial_mean, initial_cov, num_particles=2000, seed=42, beta=1.0):
        self.f = f
        self.Q = Q
        self.beta = beta
        self.num_particles = num_particles
        self.rng = np.random.default_rng(seed)

        # Normalize inputs to arrays
        if np.isscalar(initial_mean):
            self.initial_mean = np.array([float(initial_mean)])
            if np.isscalar(initial_cov):
                self.initial_cov = np.array([[float(initial_cov)]])
            else:
                self.initial_cov = np.array(initial_cov).reshape(1, 1)
        else:
            self.initial_mean = np.array(initial_mean)
            self.initial_cov = np.array(initial_cov)
                
        self.particles = self.rng.multivariate_normal(
            mean=initial_mean,
            cov=initial_cov,
            size=num_particles
        )
        self.flow_history = []

    def predict(self):
        if np.isscalar(self.Q):
            # 1D case: scalar variance
            noise = self.rng.normal(scale=np.sqrt(self.Q), size=self.num_particles)
        else:
            # Multivariate case: covariance matrix
            noise = self.rng.multivariate_normal(
                mean=np.zeros(self.Q.shape[0]),
                cov=self.Q,
                size=self.num_particles
            )
        self.particles = self.f(self.particles) + noise

    def compute_flow_ledh(self, observation):
        x_mean = np.mean(self.particles, axis=0)  # (d,)
        P = np.cov(self.particles.T)  # (d, d)

        # Compute Jacobian H at mean: diag([2*x1, cos(x2), 2*x3, ...])
        H_diag = np.empty(self.particles.shape[1])
        for i in range(len(H_diag)):
            if i % 2 == 0:
                H_diag[i] = 2.0 * x_mean[i]
            else:
                H_diag[i] = np.cos(x_mean[i])
        H = np.diag(H_diag)  # (d, d)

        R = np.eye(len(H_diag))  # Assume R = I
        S = H @ P @ H.T + R  # (d, d)
        K = P @ H.T @ np.linalg.inv(S)  # (d, d)

        # Predicted observation at mean
        y_pred_mean = np.empty_like(x_mean)
        for i in range(len(x_mean)):
            if i % 2 == 0:
                y_pred_mean[i] = x_mean[i] ** 2
            else:
                y_pred_mean[i] = np.sin(x_mean[i])

        # Innovation per particle
        innov = np.empty_like(self.particles)
        for i in range(self.particles.shape[0]):
            y_pred_i = np.empty_like(self.particles[i])
            for j in range(len(self.particles[i])):
                if j % 2 == 0:
                    y_pred_i[j] = self.particles[i, j] ** 2
                else:
                    y_pred_i[j] = np.sin(self.particles[i, j])
            innov[i] = observation - y_pred_i

        # Flow = K @ innovation
        flow = innov @ K.T  # (N, d)
        return flow

    def update(self, observation):
        flow = self.compute_flow_ledh(observation)
        self.particles += flow
        self.flow_history.append(np.mean(np.abs(flow)))

    def estimate(self):
        return np.mean(self.particles, axis=0)

    def filter(self, observations):
        means = []
        for y in observations:
            self.predict()
            self.update(y)
            m = self.estimate()
            means.append(m)
        return np.array(means)


# =============================================================================
# 4. Daum-Huang Exact Flow Filters (Algorithm 1 & 2)
# =============================================================================

@tf.function
def exact_flow_step_global(x_particles, x_bar, P_pred, H, R, z_obs, n_lambda=10, dtype=tf.float64):
    N = tf.shape(x_particles)[0]
    dx = tf.shape(x_particles)[1]
    dy = tf.shape(R)[0]

    x_particles = tf.cast(x_particles, dtype)
    x_bar = tf.cast(x_bar, dtype)
    P_pred = tf.cast(P_pred, dtype)
    H = tf.cast(H, dtype)
    R = tf.cast(R, dtype)
    z_obs = tf.cast(z_obs, dtype)

    dlam = tf.cast(1.0 / tf.cast(n_lambda, dtype), dtype)
    x = x_particles
    I = tf.eye(dx, dtype=dtype)
    z_obs_col = tf.expand_dims(z_obs, axis=-1)

    HPHT = H @ P_pred @ tf.transpose(H)
    jitter = tf.cast(1e-6, dtype)
    R_reg = R + jitter * tf.eye(dy, dtype=dtype)
    HPHT_reg = HPHT + jitter * tf.eye(dy, dtype=dtype)

    for j in range(int(n_lambda)):  # ← Key change
        lam = (j + 1.0) * dlam

        denom = lam * HPHT_reg + R_reg
        denom = 0.5 * (denom + tf.transpose(denom))
        denom_reg = denom + jitter * tf.eye(dy, dtype=dtype)
        inv_denom = safe_inv(denom_reg, jitter=1e-6)

        A = -0.5 * P_pred @ tf.transpose(H) @ inv_denom @ H

        sys_mat = I + 2.0 * lam * A
        sys_mat = 0.5 * (sys_mat + tf.transpose(sys_mat))
        sys_mat_reg = sys_mat + jitter * tf.eye(dx, dtype=dtype)

        inv_R = safe_inv(R_reg, jitter=1e-6)
        term1 = (I + lam * A) @ P_pred @ tf.transpose(H) @ inv_R @ z_obs_col
        term2 = A @ tf.expand_dims(x_bar, -1)
        rhs = term1 + term2

        b = sys_mat_reg @ rhs

        x_col = tf.expand_dims(x, axis=-1)

        Ax = A @ x_col
        dxdlam = tf.squeeze(Ax, axis=-1) + tf.squeeze(b, axis=-1)

        # Limit maximum flow magnitude per step
        max_flow = tf.constant(5.0, dtype=dtype)
        flow_norm = tf.norm(dxdlam, axis=1, keepdims=True)
        scale_factor = tf.minimum(max_flow / (flow_norm + 1e-8), 1.0)
        dxdlam = dxdlam * scale_factor

        x = x + dlam * dxdlam

        # Clip particles during flow to prevent singularity
        x = tf.clip_by_value(x, -500.0, 500.0)

    return x

@tf.function
def exact_flow_step_local(x_particles, x_bar, P_pred, R, z_obs, dgamma_dx, n_lambda=10, dtype=tf.float64):
    N = tf.shape(x_particles)[0]
    dx = tf.shape(x_particles)[1]
    dy = tf.shape(R)[0]

    x_particles = tf.cast(x_particles, dtype)
    x_bar = tf.cast(x_bar, dtype)
    P_pred = tf.cast(P_pred, dtype)
    R = tf.cast(R, dtype)
    z_obs = tf.cast(z_obs, dtype)

    dlam = tf.cast(1.0 / tf.cast(n_lambda, dtype), dtype)
    x = x_particles
    I = tf.eye(dx, dtype=dtype)
    z_obs_col = tf.expand_dims(z_obs, axis=-1)
    jitter = tf.cast(1e-6, dtype)
    R_reg = R + jitter * tf.eye(dy, dtype=dtype)

    for j in range(int(n_lambda)):  # ← Key change
        lam = (j + 1.0) * dlam

        def flow_per_particle(xi):
            xi = tf.cast(xi, dtype)
            # Clip particle state to prevent singularity in Jacobian
            xi_clipped = tf.clip_by_value(xi, -500.0, 500.0)
            H_i = dgamma_dx(xi_clipped)
            HPHT_i = H_i @ P_pred @ tf.transpose(H_i)
            HPHT_i_reg = HPHT_i + jitter * tf.eye(dy, dtype=dtype)
            
            denom_i = lam * HPHT_i_reg + R_reg
            denom_i = 0.5 * (denom_i + tf.transpose(denom_i))
            denom_i_reg = denom_i + jitter * tf.eye(dy, dtype=dtype)
            inv_denom_i = safe_inv(denom_i_reg, jitter=1e-6)

            A_i = -0.5 * P_pred @ tf.transpose(H_i) @ inv_denom_i @ H_i

            sys_mat_i = I + 2.0 * lam * A_i
            sys_mat_i = 0.5 * (sys_mat_i + tf.transpose(sys_mat_i))
            sys_mat_i_reg = sys_mat_i + jitter * tf.eye(dx, dtype=dtype)

            inv_R_i = safe_inv(R_reg, jitter=1e-6)
            term1_i = (I + lam * A_i) @ P_pred @ tf.transpose(H_i) @ inv_R_i @ z_obs_col
            term2_i = A_i @ tf.expand_dims(x_bar, -1)
            rhs_i = term1_i + term2_i

            b_i = sys_mat_i_reg @ rhs_i

            xi_col = tf.expand_dims(xi_clipped, axis=-1)
            Ax_i = A_i @ xi_col
            result = tf.squeeze(Ax_i, axis=-1) + tf.squeeze(b_i, axis=-1)
            
            # Limit flow magnitude per particle
            max_step = tf.constant(5.0, dtype=dtype)
            flow_norm = tf.norm(result, ord=2)
            scale = tf.minimum(max_step / (flow_norm + 1e-8), 1.0)
            return result * scale

        dxdlam = tf.vectorized_map(flow_per_particle, x)
        x = x + dlam * dxdlam
        
        # Global particle clipping
        x = tf.clip_by_value(x, -500.0, 500.0)

    return x


def algorithm1_DH_filter(
    y_seq, T, N, 
    Psi, Q, R,
    gamma, dgamma_dx,
    x0_mean, x0_cov,
    n_lambda=10,
    dtype=tf.float64
):
    dx = x0_mean.shape[0]
    dy = R.shape[0]

    ekf = ExtendedKalmanFilter(
        f=lambda x: tf.linalg.matvec(Psi, x),
        h=gamma,
        Q=Q, R=R,
        initial_mean=x0_mean,
        initial_cov=x0_cov,
        dtype=dtype
    )

    particles = tf.Variable(
        tfd.MultivariateNormalTriL(
            loc=tf.cast(x0_mean, dtype),
            scale_tril=tf.linalg.cholesky(tf.cast(x0_cov, dtype))
        ).sample(N)
    )

    estimates = []
    chol_Q = tf.linalg.cholesky(Q)
    I_dx = tf.eye(dx, dtype=dtype)

    for k in range(T):
        # Prediction
        std_normal = tfd.Normal(
            loc=tf.constant(0.0, dtype=dtype),
            scale=tf.constant(1.0, dtype=dtype)
        )
        noise = std_normal.sample((N, dx)) @ chol_Q
        dynamics = tf.linalg.matvec(Psi, particles)
        particles.assign(dynamics + noise)

        # EKF prediction
        ekf.m = tf.linalg.matvec(Psi, ekf.m)
        ekf.P = Psi @ ekf.P @ tf.transpose(Psi) + Q
        ekf.P = 0.5 * (ekf.P + tf.transpose(ekf.P))
        # assert ekf.P.shape == (dx, dx)
        # assert ekf.m.shape == (dx,)

        # Stabilize EKF covariance
        if tf.reduce_any(tf.math.is_nan(ekf.P)) or tf.reduce_any(tf.math.is_inf(ekf.P)):
            ekf.P = tf.eye(dx, dtype=dtype)
        else:
            min_eig = tf.reduce_min(tf.linalg.eigvalsh(ekf.P))
            if min_eig < 1e-6:
                ekf.P = ekf.P + (1e-6 - min_eig) * I_dx

        # Global linearization
        H = dgamma_dx(ekf.m)
        assert H.shape == (dy, dx)
        particles.assign(
            exact_flow_step_global(
                particles, ekf.m, ekf.P, H, R, y_seq[k], n_lambda=n_lambda, dtype=dtype
            )
        )

        # Final particle clipping
        particles.assign(tf.clip_by_value(particles, -500.0, 500.0))

        x_mean = tf.reduce_mean(particles, axis=0)
        estimates.append(x_mean.numpy())

        # EKF update with Joseph form
        y_pred = gamma(ekf.m)
        innov = y_seq[k] - y_pred
        S = H @ ekf.P @ tf.transpose(H) + R
        S = 0.5 * (S + tf.transpose(S))
        K = compute_kalman_gain_robust(ekf.P, H, R, jitter=1e-6)

        I_KH = I_dx - K @ H

        innov_col = tf.expand_dims(innov, axis=-1)
        ekf.m = ekf.m + tf.squeeze(K @ innov_col, axis=-1)
        ekf.P = I_KH @ ekf.P @ tf.transpose(I_KH) + K @ R @ tf.transpose(K)
        ekf.P = 0.5 * (ekf.P + tf.transpose(ekf.P))

        # Final stabilization
        if tf.reduce_any(tf.math.is_nan(ekf.P)) or tf.reduce_any(tf.math.is_inf(ekf.P)):
            ekf.P = tf.eye(dx, dtype=dtype)
        else:
            min_eig = tf.reduce_min(tf.linalg.eigvalsh(ekf.P))
            if min_eig < 1e-6:
                ekf.P = ekf.P + (1e-6 - min_eig) * I_dx

    return np.array(estimates)


def algorithm2_modified_DH_filter(
    y_seq, T, N,
    Psi, Q, R,
    gamma, dgamma_dx,
    x0_mean, x0_cov,
    n_lambda=10,
    dtype=tf.float64
):
    dx = x0_mean.shape[0]
    dy = R.shape[0]

    ekf = ExtendedKalmanFilter(
        f=lambda x: tf.linalg.matvec(Psi, x),
        h=gamma,
        Q=Q, R=R,
        initial_mean=x0_mean,
        initial_cov=x0_cov,
        dtype=dtype
    )

    particles = tf.Variable(
        tfd.MultivariateNormalTriL(
            loc=tf.cast(x0_mean, dtype),
            scale_tril=tf.linalg.cholesky(tf.cast(x0_cov, dtype))
        ).sample(N)
    )

    estimates = []
    chol_Q = tf.linalg.cholesky(Q)
    I_dx = tf.eye(dx, dtype=dtype)

    for k in range(T):
        # Prediction
        std_normal = tfd.Normal(
            loc=tf.constant(0.0, dtype=dtype),
            scale=tf.constant(1.0, dtype=dtype)
        )
        noise = std_normal.sample((N, dx)) @ chol_Q
        dynamics = tf.linalg.matvec(Psi, particles)
        particles.assign(dynamics + noise)

        # EKF prediction
        ekf.m = tf.linalg.matvec(Psi, ekf.m)
        ekf.P = Psi @ ekf.P @ tf.transpose(Psi) + Q
        ekf.P = 0.5 * (ekf.P + tf.transpose(ekf.P))

        # Stabilize EKF covariance
        if tf.reduce_any(tf.math.is_nan(ekf.P)) or tf.reduce_any(tf.math.is_inf(ekf.P)):
            ekf.P = tf.eye(dx, dtype=dtype)
        else:
            min_eig = tf.reduce_min(tf.linalg.eigvalsh(ekf.P))
            if min_eig < 1e-6:
                ekf.P = ekf.P + (1e-6 - min_eig) * I_dx

        # Local linearization
        particles.assign(
            exact_flow_step_local(
                particles, ekf.m, ekf.P, R, y_seq[k], dgamma_dx, n_lambda=n_lambda, dtype=dtype
            )
        )

        # Final particle clipping
        particles.assign(tf.clip_by_value(particles, -500.0, 500.0))

        x_mean = tf.reduce_mean(particles, axis=0)
        estimates.append(x_mean.numpy())

        # Feedback to EKF
        ekf.m = tf.cast(x_mean, dtype)
        H_mean = dgamma_dx(ekf.m)
        y_pred = gamma(ekf.m)
        innov = y_seq[k] - y_pred
        
        S = H_mean @ ekf.P @ tf.transpose(H_mean) + R
        S = 0.5 * (S + tf.transpose(S))
        K = compute_kalman_gain_robust(ekf.P, H_mean, R, jitter=1e-6)

        I_KH = I_dx - K @ H_mean
        ekf.P = I_KH @ ekf.P @ tf.transpose(I_KH) + K @ R @ tf.transpose(K)
        ekf.P = 0.5 * (ekf.P + tf.transpose(ekf.P))

        # Final stabilization
        if tf.reduce_any(tf.math.is_nan(ekf.P)) or tf.reduce_any(tf.math.is_inf(ekf.P)):
            ekf.P = tf.eye(dx, dtype=dtype)
        else:
            min_eig = tf.reduce_min(tf.linalg.eigvalsh(ekf.P))
            if min_eig < 1e-6:
                ekf.P = ekf.P + (1e-6 - min_eig) * I_dx

    return np.array(estimates)

# =============================================================================
# 4. Particle Flow Particle Filter (PFPF) – Algorithm 1 (LEDH) & Algorithm 2 (EDH)
# =============================================================================

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

class PFPF_LEDH:
    """
    Particle Flow Particle Filter with Localized Exact Daum-Huang (LEDH) flow.
    Implements Algorithm 1 from Li & Coates (2017) using TensorFlow.
    
    Assumptions:
      - State: x ∈ ℝ^dx
      - Observation: y ∈ ℝ^dy
      - Process noise: Gaussian, N(0, Q)
      - Observation noise: Gaussian, N(0, R) — can be extended
    """

    def __init__(
        self,
        f,                     # state transition: f(x) → x_next (callable)
        Q,                     # process noise covariance (dx, dx)
        h,                     # observation function: h(x) → y (callable, tf.function compatible)
        dh_dx,                 # Jacobian of h: dh_dx(x) → H (dy, dx)
        initial_mean,          # (dx,)
        initial_cov,           # (dx, dx)
        R = None,
        num_particles=500,
        n_lambda=29,
        seed=42,
        dtype=tf.float64,
        dy=None
    ):
        self.f = f
        self.h = h
        self.dh_dx = dh_dx
        self.Q = tf.constant(Q, dtype=dtype)
        self.R = tf.constant(R, dtype=dtype) if R is not None else tf.eye(tf.shape(self.Q)[0], dtype=dtype)  # Default: R = I; override if needed
        self.num_particles = num_particles
        self.n_lambda = n_lambda
        self.dtype = dtype
        self.seed = seed
        self.rng = tf.random.Generator.from_seed(seed)

        self.dx = tf.shape(initial_mean)[0]
        self.dy = dy if dy is not None else self.dx  # inferred from first observation

        # Initialize particles
        self.particles = tfd.MultivariateNormalTriL(
            loc=tf.cast(initial_mean, dtype),
            scale_tril=tf.linalg.cholesky(tf.cast(initial_cov, dtype))
        ).sample(num_particles, seed=42)

        # Log weights (for numerical stability)
        self.log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, dtype)))
        self.ess_history = []
        self.flow_history = []

        # Exponentially spaced step sizes (q = 1.2, sum = 1)
        q = 1.2
        eps0 = (1.0 - q) / (1.0 - q ** n_lambda)
        self.epsilon_steps = tf.constant([eps0 * (q ** j) for j in range(n_lambda)], dtype=dtype)

    @tf.function
    def _predict_step(self, particles):
        """Predict particles using dynamics f and Gaussian noise."""
        noise = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.dx, dtype=self.dtype),
            scale_tril=tf.linalg.cholesky(self.Q)
        ).sample(tf.shape(particles)[0])
        return self.f(particles) + noise

    @tf.function
    def _compute_flow_per_particle(self, particle, z_obs, P_pred, x_bar):
        """Compute LEDH flow for a single particle."""
        H_i = self.dh_dx(particle)  # (dy, dx)
        if self.dy is None:
            self.dy = tf.shape(H_i)[0]

        e_i = self.h(particle) - tf.linalg.matvec(H_i, particle)  # (dy,)

        x_curr = particle
        log_det_J = tf.constant(0.0, dtype=self.dtype)
        lambd = tf.constant(0.0, dtype=self.dtype)
        for eps in self.epsilon_steps:
            lambd += eps
            HPHT = lambd * H_i @ P_pred @ tf.transpose(H_i)  # (dy, dy)
            denom = HPHT + self.R
            # Ensure symmetry
            denom = 0.5 * (denom + tf.transpose(denom))
            inv_denom = tf.linalg.inv(denom + 1e-6 * tf.eye(self.dy, dtype=self.dtype))

            A_i = -0.5 * P_pred @ tf.transpose(H_i) @ inv_denom @ H_i  # (dx, dx)
            rhs = tf.expand_dims(z_obs - e_i, axis=-1)
            solved = tf.linalg.solve(self.R, rhs)
            x_bar_col = tf.expand_dims(x_bar, axis=-1)
            # solved = tf.squeeze(solved, axis=-1)
            term1 = (tf.eye(self.dx, dtype=self.dtype) + lambd * A_i) @ P_pred @ tf.transpose(H_i) @ solved
            term2 = A_i @ x_bar_col
            b_i = (tf.eye(self.dx, dtype=self.dtype) + 2*lambd * A_i) @ (term1 + term2)

            x_curr_col = tf.expand_dims(x_curr, axis=-1)
            delta = A_i @ x_curr_col + b_i
            x_curr = x_curr + eps * tf.squeeze(delta, axis=-1)

            J_step = tf.eye(self.dx, dtype=self.dtype) + eps * A_i
            sign, logdet = tf.linalg.slogdet(J_step)
            log_det_J += tf.cond(sign > 0, lambda: logdet, lambda: tf.math.log(tf.abs(tf.linalg.det(J_step)) + 1e-12))

        flow = x_curr - particle
        return flow, log_det_J

    def update(self, z_obs):
        """Perform particle flow and weight update."""
        z_obs = tf.cast(z_obs, self.dtype)
        x_bar = tf.reduce_mean(self.particles, axis=0)
        P_pred = tfp.stats.covariance(self.particles) + 1e-6 * tf.eye(self.dx, dtype=self.dtype)

        # Vectorize flow computation
        flows_and_logdets = tf.vectorized_map(
            lambda p: self._compute_flow_per_particle(p, z_obs, P_pred, x_bar),
            self.particles
        )
        flows = flows_and_logdets[0]  # (N, dx)
        log_det_Js = flows_and_logdets[1]  # (N,)

        new_particles = self.particles + flows

        # Observation log-likelihood: p(z | x_new)
        obs_error = z_obs - tf.vectorized_map(self.h, new_particles)  # (N, dy)
        obs_loglik = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.dy, dtype=self.dtype),
            covariance_matrix=self.R
        ).log_prob(obs_error)

        # Transition log-density: p(x_new | x_old) ≈ N(f(x_old), Q)
        trans_mean = tf.vectorized_map(self.f, self.particles)
        trans_error = new_particles - trans_mean
        trans_loglik = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.dx, dtype=self.dtype),
            covariance_matrix=self.Q
        ).log_prob(trans_error)

        # Prior log-density: p(x_old) — approximated via current particles (not needed if using full proposal)
        # In PFPF, key term is: log w ∝ log p(z|x_new) + log |det J|
        # We omit p(x_new|x_old)/p(x_old) ratio under assumption that proposal dominates
        log_weights_new = self.log_weights + obs_loglik + log_det_Js

        # Normalize
        max_logw = tf.reduce_max(log_weights_new)
        weights_norm = tf.exp(log_weights_new - max_logw)
        weights_norm /= tf.reduce_sum(weights_norm)
        self.log_weights = tf.math.log(weights_norm + 1e-30)

        # Track flow magnitude
        flow_mag = tf.reduce_mean(tf.abs(flows))
        self.flow_history.append(flow_mag.numpy())

        self.particles = new_particles

        self.particles = tf.clip_by_value(self.particles, -500.0, 500.0)

        # ESS
        ess = 1.0 / tf.reduce_sum(tf.exp(2 * self.log_weights))
        self.ess_history.append(ess.numpy())

        # Resample if ESS < N/2
        if ess < self.num_particles / 2:
            indices = tfp.distributions.Categorical(logits=self.log_weights).sample(self.num_particles, seed=42)
            self.particles = tf.gather(self.particles, indices)
            self.log_weights = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, self.dtype)))

    def predict(self):
        """Predict step."""
        self.particles = self._predict_step(self.particles)

    def estimate(self):
        """Return weighted mean estimate."""
        weights = tf.exp(self.log_weights)
        return tf.reduce_sum(weights[:, None] * self.particles, axis=0).numpy()

    def filter(self, observations):
        """Run full filtering sequence."""
        estimates = []
        for z in observations:
            self.predict()
            self.update(z)
            estimates.append(self.estimate())
        return np.array(estimates)
# ./src/models/particle_filters.py

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


class PFPF_EDH:
    """
    Particle Flow Particle Filter with Exact Daum-Huang (EDH) flow.
    Implements Algorithm 2 from Li & Coates (2017).
    
    Global flow: same A(λ), b(λ) for all particles.
    Jacobian determinant is identical across particles → cancels in weight normalization.
    """

    def __init__(
        self,
        f,                     # state transition: f(x) → x_next (callable)
        Q,                     # process noise covariance (dx, dx)
        h,                     # observation function: h(x) → y (callable, tf.function compatible)
        dh_dx,                 # Jacobian of h: dh_dx(x) → H (dy, dx)
        initial_mean,          # (dx,)
        initial_cov,           # (dx, dx)
        R = None,
        num_particles=500,
        n_lambda=29,
        seed=42,
        dtype=tf.float64,
        dy=None 
    ):
        self.f = f
        self.h = h
        self.dh_dx = dh_dx
        self.Q = tf.constant(Q, dtype=dtype)
        self.R = tf.constant(R, dtype=dtype) if R is not None else tf.eye(tf.shape(self.Q)[0], dtype=dtype)  # Default: R = I; override if needed
        self.num_particles = num_particles
        self.n_lambda = n_lambda
        self.dtype = dtype
        self.seed = seed
        self.rng = tf.random.Generator.from_seed(seed)

        self.dx = tf.shape(initial_mean)[0]
        self.dy = dy if dy is not None else self.dx  # inferred from first observation

        # Initialize particles
        self.particles = tfd.MultivariateNormalTriL(
            loc=tf.cast(initial_mean, dtype),
            scale_tril=tf.linalg.cholesky(tf.cast(initial_cov, dtype))
        ).sample(num_particles, seed=42)

        # Log weights
        self.log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, dtype)))
        self.ess_history = []
        self.flow_history = []

        # Exponentially spaced step sizes (q = 1.2, sum = 1)
        q = 1.2
        eps0 = (1.0 - q) / (1.0 - q ** n_lambda)
        self.epsilon_steps = tf.constant([eps0 * (q ** j) for j in range(n_lambda)], dtype=dtype)

        # EKF for global predictive covariance
        self.ekf_m = tf.Variable(tf.cast(initial_mean, dtype), trainable=False)
        self.ekf_P = tf.Variable(tf.cast(initial_cov, dtype), trainable=False)

    @tf.function
    def _predict_step(self, particles):
        """Predict particles using dynamics f and Gaussian noise."""
        noise = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.dx, dtype=self.dtype),
            scale_tril=tf.linalg.cholesky(self.Q)
        ).sample(tf.shape(particles)[0])
        return self.f(particles) + noise

    def predict(self):
        """Perform prediction step and update EKF."""
        self.particles = self._predict_step(self.particles)

        # EKF prediction (linearized or assumed linear)
        self.ekf_m.assign(self.f(self.ekf_m))
        P_pred = self.ekf_P + self.Q
        P_pred = 0.5 * (P_pred + tf.transpose(P_pred))
        min_eig = tf.reduce_min(tf.linalg.eigvalsh(P_pred))
        if min_eig < 1e-6:
            P_pred += (1e-6 - min_eig) * tf.eye(self.dx, dtype=self.dtype)
        self.ekf_P.assign(P_pred)

    @tf.function
    def _compute_global_flow(self, particles, z_obs, P_pred, x_bar):
        """Compute global EDH flow for all particles."""
        H = self.dh_dx(x_bar)  # (dy, dx)
        if self.dy is None:
            self.dy = tf.shape(H)[0]

        e = self.h(x_bar) - tf.linalg.matvec(H, x_bar)  # (dy,)
        R = self.R  # (dy, dy)

        x_curr = particles  # (N, dx)
        total_flow = tf.zeros_like(particles)
        lambd = tf.constant(0.0, dtype=self.dtype) 
        for eps in self.epsilon_steps:
            lambd += eps
            HPHT = lambd * H @ P_pred @ tf.transpose(H)  # (dy, dy)
            denom = HPHT + R
            denom = 0.5 * (denom + tf.transpose(denom))
            # inv_denom = tf.linalg.inv(denom + 1e-6 * tf.eye(self.dy, dtype=self.dtype))
            inv_denom = safe_inv(denom, jitter=1e-6)
            A = -0.5 * P_pred @ tf.transpose(H) @ inv_denom @ H  # (dx, dx)
            
            # Ensure z_obs is (dy,)
            if z_obs.shape.ndims > 1:
                z_obs = tf.squeeze(z_obs, axis=-1)  # Now (dy,)

            # e = H @ x_bar → (dy,)
            e = tf.linalg.matvec(H, x_bar)  # (dy,)

            # Innovation
            innov = z_obs - e  # (dy,)

            # Solve R^{-1} * innov → (dy,)
            # Since R is (dy, dy), and innov is (dy,), use solve with (dy, 1)
            innov_col = tf.expand_dims(innov, axis=-1)  # (dy, 1)
            solved = tf.linalg.solve(R, innov_col)   # (dy, 1)
            # solved = tf.squeeze(solved_col, axis=-1)    # (dy,)

            # --- FIX: Make RHS 2D for tf.linalg.solve ---
            # rhs_vec = z_obs - e  # (dy,)
            # rhs = tf.expand_dims(rhs_vec, axis=-1)  # (dy, 1)
            # solved = tf.linalg.solve(R, rhs)  # (dy, 1)
            # solved = tf.squeeze(solved, axis=-1)  # (dy,)
            x_bar_col = tf.expand_dims(x_bar, axis=-1)  # (dx, 1)

            term1 = (tf.eye(self.dx, dtype=self.dtype) + lambd*A) @ P_pred @ tf.transpose(H) @ solved  # (dx,)
            term2 = A @ x_bar_col  # (dx, 1)
            b = tf.squeeze((tf.eye(self.dx, dtype=self.dtype)+2*lambd*A) @ (term1 + term2), axis=-1)  # (dx,)

            # Apply flow to all particles
            delta = (A @ tf.transpose(x_curr)) + tf.transpose(tf.tile(b[None, :], [tf.shape(x_curr)[0], 1]))
            delta = tf.transpose(delta)  # (N, dx)
            x_curr = x_curr + eps * delta
            total_flow = total_flow + eps * delta

        return total_flow

    def update(self, z_obs):
        """Perform particle flow and weight update."""
        z_obs = tf.cast(z_obs, self.dtype)
        x_bar = tf.reduce_mean(self.particles, axis=0)
        P_pred = self.ekf_P

        flows = self._compute_global_flow(self.particles, z_obs, P_pred, x_bar)
        new_particles = self.particles + flows

        # Observation log-likelihood: p(z | x_new)
        obs_error = z_obs - tf.vectorized_map(self.h, new_particles)  # (N, dy)
        obs_loglik = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.dy, dtype=self.dtype),
            covariance_matrix=self.R
        ).log_prob(obs_error)

        # Transition log-density: p(x_new | x_old) ≈ N(f(x_old), Q)
        trans_mean = tf.vectorized_map(self.f, self.particles)
        trans_error = new_particles - trans_mean
        trans_loglik = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.dx, dtype=self.dtype),
            covariance_matrix=self.Q
        ).log_prob(trans_error)

        # Prior density: p(x_old) — approximated via current particles
        # In practice, for weight ratio, we use:
        #   w ∝ p(z|x_new) * p(x_new|x_old) / p(x_old|x_old_prev)
        # But since we don't track full trajectory, and proposal includes dynamics,
        # standard practice in PFPF is to use:
        #   w ∝ p(z|x_new)   (Jacobian cancels globally)
        # See Eq.(37) in paper: wi ∝ p(η1|xi)p(z|η1)/p(η0|xi) → but η0 = f(xi) + noise, so p(η0|xi) known
        # However, for simplicity and consistency with paper's Algorithm 2, we use:
        prior_loglik = tfd.MultivariateNormalFullCovariance(
            loc=trans_mean,
            covariance_matrix=self.Q
        ).log_prob(self.particles)

        # Full weight update (optional: include prior ratio)
        # For now, follow paper: weight ∝ p(z|x_new) * [p(x_new|x_old) / p(x_old|x_old_prev)]
        # But since p(x_old|x_old_prev) is embedded in previous weight, incremental weight is:
        log_weights_new = self.log_weights + obs_loglik

        # Normalize
        max_logw = tf.reduce_max(log_weights_new)
        weights_norm = tf.exp(log_weights_new - max_logw)
        weights_norm /= tf.reduce_sum(weights_norm)
        self.log_weights = tf.math.log(weights_norm + 1e-30)

        # Track flow magnitude
        flow_mag = tf.reduce_mean(tf.abs(flows))
        self.flow_history.append(flow_mag.numpy())

        self.particles = new_particles
        self.particles = tf.clip_by_value(self.particles, -500.0, 500.0)

        # ESS
        ess = 1.0 / tf.reduce_sum(tf.exp(2 * self.log_weights))
        self.ess_history.append(ess.numpy())

        # Resample if ESS < N/2
        if ess < self.num_particles / 2:
            indices = tfp.distributions.Categorical(logits=self.log_weights).sample(self.num_particles, seed=42)
            self.particles = tf.gather(self.particles, indices)
            self.log_weights = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, self.dtype)))

        # EKF update (optional but recommended)
        # y_pred = self.h(self.ekf_m)
        # innov = tf.expand_dims(z_obs- y_pred, axis=-1) 
        # H_ekf = self.dh_dx(self.ekf_m)
        # S = H_ekf @ self.ekf_P @ tf.transpose(H_ekf) + self.R
        # S = 0.5 * (S + tf.transpose(S))
        # K = self.ekf_P @ tf.transpose(H_ekf) @ tf.linalg.inv(S + 1e-6 * tf.eye(self.dy, dtype=self.dtype))
        # self.ekf_m.assign(self.ekf_m + tf.squeeze(K @ innov, axis=-1))
        # I_KH = tf.eye(self.dx, dtype=self.dtype) - K @ H_ekf
        # self.ekf_P.assign(I_KH @ self.ekf_P @ tf.transpose(I_KH) + K @ self.R @ tf.transpose(K))
        # self.ekf_P.assign(0.5 * (self.ekf_P + tf.transpose(self.ekf_P)))

        # EKF update with Joseph form
        y_pred = self.h(self.ekf_m)
        H_ekf = self.dh_dx(self.ekf_m)
        innov = z_obs - tf.squeeze(y_pred)  # Ensure (dy,)

        # Ensure innov is 1D
        innov = tf.reshape(innov, [-1])  # (dy,)

        S = H_ekf @ self.ekf_P @ tf.transpose(H_ekf) + self.R
        S = 0.5 * (S + tf.transpose(S))
        K = self.ekf_P @ tf.transpose(H_ekf) @ tf.linalg.inv(S + 1e-6 * tf.eye(self.dy, dtype=self.dtype))

        # Use matvec for correct shape
        update = tf.linalg.matvec(K, innov)  # (dx,)
        self.ekf_m.assign(self.ekf_m + update)

        I_KH = tf.eye(self.dx, dtype=self.dtype) - K @ H_ekf
        self.ekf_P.assign(I_KH @ self.ekf_P @ tf.transpose(I_KH) + K @ self.R @ tf.transpose(K))
        self.ekf_P.assign(0.5 * (self.ekf_P + tf.transpose(self.ekf_P)))

    def estimate(self):
        """Return weighted mean estimate."""
        weights = tf.exp(self.log_weights)
        return tf.reduce_sum(weights[:, None] * self.particles, axis=0).numpy()

    def filter(self, observations):
        """Run full filtering sequence."""
        estimates = []
        for z in observations:
            self.predict()
            self.update(z)
            estimates.append(self.estimate())
        return np.array(estimates)