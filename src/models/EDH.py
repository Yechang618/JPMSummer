"""
Exact Daum-Huang particle flow (affine/linear flow) implementation.

This implements the Exact Daum-Huang flow for linear-Gaussian likelihoods:
  y ~ N(H x, R)
Given an empirical prior represented by particles with mean m0 and covariance P0,
we transport particles deterministically through a sequence of affine maps so that
at each intermediate pseudo-time \lambda \in [0,1] the particle cloud matches
the intermediate Gaussian with mean m(\lambda) and covariance P(\lambda), where

  P(\lambda) = (P0^{-1} + \lambda H^T R^{-1} H)^{-1}
  m(\lambda) = P(\lambda) (P0^{-1} m0 + \lambda H^T R^{-1} y)

We update particles in small steps in \lambda and apply the affine transform

  x_{new} = m(\lambda) + A (x_old - m_prev),

where A = chol(P(\lambda)) @ inv(chol(P_prev)). This preserves the Gaussian
marginals exactly in the linear-Gaussian case.

This implementation uses TensorFlow for vectorized matrix operations.

References:
- Daum, F., & Huang, J. (2008). "Particle flow for nonlinear filters" (and later work)
"""
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import Callable, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def compute_empirical_mean_and_cov(particles: tf.Tensor, jitter: float, dtype=tf.float32):
    """Compute empirical mean and covariance with diagonal jitter.

    Returns (mean, cov) with shapes (d,) and (d,d).
    """
    particles = tf.cast(particles, dtype=dtype)
    mean = tf.reduce_mean(particles, axis=0)
    centered = particles - mean
    N = tf.cast(tf.shape(particles)[0], dtype)
    cov = tf.matmul(centered, centered, transpose_a=True) / N
    cov += tf.eye(tf.shape(particles)[1], dtype=dtype) * tf.cast(jitter, dtype)
    return mean, cov

def calculate_A_b(h: Callable[[tf.Tensor], tf.Tensor], m0: tf.Tensor, y: tf.Tensor, lam: tf.Tensor, P_pred: tf.Tensor, R: tf.Tensor, jitter: float, state_dim: int, dtype=tf.float32):
    """Compute A (dxd) and b (d,) for EDH inner lambda step.

    Returns (A, b_vec, H_curr) where H_curr is the linearization matrix at m0.
    """
    # Ensure m0 is a 1-D state vector of length `state_dim` (robust to stray shapes)
    m0_vec = tf.reshape(tf.convert_to_tensor(m0), [-1])
    m0_vec = tf.reshape(m0_vec[:state_dim], [-1])

    # Linearize observation at current mean m0_vec
    H_curr = lin_H(h, m0_vec, dtype=dtype)
    # effective observation for linearization at m0_vec
    y_eff = y - tf.convert_to_tensor(h(m0_vec), dtype=dtype) + tf.reshape(tf.matmul(H_curr, tf.reshape(m0_vec, [-1, 1])), [-1])

    # Build innovation covariance with small jitter for stability
    S = lam * H_curr @ P_pred @ tf.transpose(H_curr) + R
    S += tf.eye(tf.shape(S)[0], dtype=dtype) * tf.cast(jitter, dtype)
    chol_S = safe_cholesky(S, jitter, dtype=dtype)
    PHt = tf.linalg.matmul(P_pred, tf.transpose(H_curr))
    # Solve for Kalman-like gain K: S X = (PHt)^T  -> X = S^{-1} (PHt)^T
    X = tf.linalg.cholesky_solve(chol_S, tf.transpose(PHt))
    K = tf.transpose(X)  # shape [d, obs_dim]

    # A is dxd mapping
    A = -tf.matmul(K, H_curr)/2

    # b vector computation (produce a state-dim vector)
    I_state = tf.eye(state_dim, dtype=dtype)
    b_mat = I_state + 2.0 * lam * A
    # temp_vec = PHt @ R^{-1} @ y_eff  -> shape [d]
    temp_vec = tf.reshape(tf.matmul(tf.matmul(PHt, tf.linalg.inv(R)), tf.reshape(y_eff, [-1, 1])), [-1])
    b_vec = tf.matmul(b_mat, tf.reshape(temp_vec, [-1, 1]))
    b_vec = b_vec + tf.matmul(A, tf.reshape(m0_vec, [-1, 1]))
    b_vec = tf.reshape(b_vec, [-1])

    return A, b_vec, H_curr

def lin_F(f: Callable[[tf.Tensor], tf.Tensor], x0: tf.Tensor, dtype=tf.float32):
    """Linearize the observation function h_func at point x0 to obtain H and effective y."""
    with tf.GradientTape() as tape:
        tape.watch(x0)
        f0 = tf.convert_to_tensor(f(x0), dtype=dtype)
    J = tape.jacobian(f0, x0)
    return tf.reshape(J, (tf.shape(f0)[0], tf.shape(x0)[0]))


def safe_cholesky(mat: tf.Tensor, base_jitter: float = 1e-8, max_tries: int = 6, dtype=tf.float32):
    """Attempt Cholesky with escalating diagonal jitter until success.

    Returns the lower-triangular Cholesky factor. Raises the final error
    if all attempts fail.
    """
    jitter = float(base_jitter)
    eye = tf.eye(tf.shape(mat)[0], dtype=dtype)
    last_err = None
    for i in range(max_tries):
        mat_try = mat + eye * tf.cast(jitter, dtype)
        try:
            chol = tf.linalg.cholesky(mat_try)
        except tf.errors.InvalidArgumentError as e:
            last_err = e
            jitter *= 10.0
            continue
        # Some TF builds return a NaN-filled chol instead of raising; detect that
        if tf.reduce_any(tf.math.is_nan(chol)):
            last_err = RuntimeError("Cholesky produced NaNs")
            jitter *= 10.0
            continue
        return chol
    # final attempt (will raise to surface the last error)
    mat_final = mat + eye * tf.cast(jitter, dtype)
    return tf.linalg.cholesky(mat_final)

def lin_H(h: Callable[[tf.Tensor], tf.Tensor], x0: tf.Tensor, dtype=tf.float32):
    """Linearize the observation function h_func at point x0 to obtain H and effective y."""
    with tf.GradientTape() as tape:
        tape.watch(x0)
        h0 = tf.convert_to_tensor(h(x0), dtype=dtype)
    J = tape.jacobian(h0, x0)
    return tf.reshape(J, (tf.shape(h0)[0], tf.shape(x0)[0]))

def mP_predict(F: tf.Tensor, P:tf.Tensor, m: tf.Tensor,Q: tf.Tensor, R: tf.Tensor, jitter: float = 1e-6, dtype=tf.float32):
    """Compute P(lam) and m(lam) for given lam in [0,1]."""
    m_pred = tf.matmul(F, tf.reshape(m, [-1, 1]))
    P_pred = F @ P @ tf.transpose(F) + Q
    return P_pred, m_pred

def mP_update(P_pred: tf.Tensor, m_pred: tf.Tensor, H: tf.Tensor, R: tf.Tensor, y: tf.Tensor, jitter: float = 1e-6, dtype=tf.float32):
    # P_pred, m_pred: prior covariance and mean after prediction step
    # Ensure column shapes for vector-operations
    m_pred_col = tf.reshape(m_pred, [-1, 1])
    y_col = tf.reshape(y, [-1, 1])

    # Innovation (as column)
    y_pred_col = tf.matmul(H, m_pred_col)
    S = H @ P_pred @ tf.transpose(H) + R
    S += tf.eye(tf.shape(S)[0], dtype=dtype) * tf.cast(jitter, dtype)

    # Numerically stable inverse using cholesky with jitter escalation
    chol_S = safe_cholesky(S, base_jitter=jitter, dtype=dtype)
    # Solve S X = (PHt)^T  => X = S^{-1} (PHt)^T
    PHt = tf.matmul(P_pred, tf.transpose(H))
    X = tf.linalg.cholesky_solve(chol_S, tf.transpose(PHt))
    K = tf.transpose(X)  # shape [d, obs]

    innovation = y_col - y_pred_col

    # Update (keep m as column then return as 1-D)
    m_col = m_pred_col + tf.matmul(K, innovation)
    KH = tf.matmul(K, H)
    I = tf.eye(tf.shape(P_pred)[0], dtype=dtype)
    # Joseph form: (I-KH) P_pred (I-KH)^T + K R K^T
    P = (I - KH) @ P_pred @ tf.transpose(I - KH) + tf.matmul(K, tf.matmul(R, tf.transpose(K)))

    return P, tf.reshape(m_col, [-1])

class EDH:
    """
    Exact Daum-Huang affine particle flow for linear-Gaussian likelihoods.

    This class assumes the observation model is linear:
        y = H x + noise,   noise ~ N(0, R)

    and the prior is represented by particles (empirical mean and covariance).

    Usage:
      - Create with number of particles and state dim
      - Initialize with particles (or a distribution sample)
      - Call `predict` to propagate particles through dynamics (optional)
      - Call `update` with observation `y`, observation matrix `H`, and `R`

    Note: This flow is exact for linear-Gaussian likelihoods (and Gaussian prior).
    For nonlinear observation functions you would need a local linearization.
    """

    def __init__(self,
                 num_particles: int,
                 f: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 h: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 state_dim: int = None,
                 observation_dim: Optional[int] = None,
                 Q: Optional[tf.Tensor] = None,
                 R: Optional[tf.Tensor] = None,
                 dtype=tf.float32):
        
        self.num_particles = int(num_particles)
        # Backwards-compatible defaults: if `f` or `h` are not provided treat
        # them as identity maps; if `observation_dim` is not provided assume
        # observation_dim == state_dim.
        self.state_dim = int(state_dim)
        if observation_dim is None:
            observation_dim = self.state_dim
        self.observation_dim = int(observation_dim)
        self.f = f if f is not None else (lambda x: x)
        self.h = h if h is not None else (lambda x: x)
        self.dtype = dtype

        # Particles: shape [N, state_dim]
        self.particles: Optional[tf.Tensor] = None

        # Diagnostic quantities
        self.mean_history = []
        self.cov_history = []

        # Integration control for lambda in [0,1]
        self.n_flow_steps = 20
        self.jitter = 1e-6

        if Q is not None:
            self.Q = tf.cast(Q, dtype=self.dtype)
        else:
            self.Q = tf.zeros((self.state_dim, self.state_dim), dtype=self.dtype)

        if R is not None:
            self.R = tf.cast(R, dtype=self.dtype)
            self.R_inv = tf.linalg.inv(self.R)
        else:
            self.R = tf.zeros((self.observation_dim, self.observation_dim), dtype=self.dtype)
            self.R_inv = tf.zeros((self.observation_dim, self.observation_dim), dtype=self.dtype)
        
        # Flow diagnostics: per-lambda-step magnitudes and conditioning
        # These are lists of scalar tensors (one entry per lambda step)
        self.flow_A_norm_history = []
        self.flow_b_norm_history = []
        self.flow_disp_norm_history = []
        self.jacobian_cond_history = []
    
    def initialize(self, particles: Optional[tf.Tensor] = None, initial_dist: Optional[tfd.Distribution] = None):
        """Initialize the particle cloud. Provide either `particles` tensor or an
        initial TFP `Distribution` to sample from.
        """
        if particles is None and initial_dist is None:
            raise ValueError("Either `particles` or `initial_dist` must be provided")

        if particles is None:
            p = initial_dist.sample(self.num_particles)
            self.particles = tf.cast(p, dtype=self.dtype)
        else:
            p = tf.convert_to_tensor(particles)
            self.particles = tf.cast(p, dtype=self.dtype)

        # store mean (`m`) and covariance (`P`) in the natural order
        self.m, self.P = self._empirical_mean_and_cov(self.particles)
        # Clear histories
        self.mean_history = []
        self.cov_history = []
        # Clear flow diagnostics
        self.flow_A_norm_history = []
        self.flow_b_norm_history = []
        self.flow_disp_norm_history = []
        self.jacobian_cond_history = []


    def propagate(self):
        """Propagate particles through a dynamics function and optionally add noise."""
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        self.particles = self.f(self.particles)
        # Only add process-noise when Q is non-zero (avoids cholesky on zero)
        if self.Q is not None:
            # Check if Q is (near) zero; if so, skip sampling noise
            q_max = tf.reduce_max(tf.abs(self.Q))
            if tf.cast(q_max, self.dtype) > tf.cast(0.0, self.dtype):
                # noise = tf.random.normal(shape=(self.num_particles, self.state_dim), dtype=self.dtype)
                # # chol = tf.linalg.cholesky(tf.cast(self.Q, self.dtype))
                # # noise_scaled = tf.linalg.matmul(noise, chol)
                # # Add process noise samples
                q_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(self.state_dim, dtype=self.dtype), scale_tril=self.Q)
                # noise size (N, state_dim)
                noise = q_dist.sample(self.num_particles)
                noise = tf.cast(noise, dtype=self.dtype)
                # print(noise)
                self.particles = self.particles + noise

    def _empirical_mean_and_cov(self, particles: tf.Tensor):
        return compute_empirical_mean_and_cov(particles, self.jitter, dtype=self.dtype)

    def update(self, y: tf.Tensor):
        """Apply the Daum-Huang exact flow to move particles to posterior.

        Supports nonlinear observation functions via linearization at prior mean.
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        # Cast inputs
        y = tf.cast(tf.reshape(y, [-1]), dtype=self.dtype)
        if self.R is None:
            raise ValueError("Observation covariance R must be provided")
        R = tf.cast(self.R, dtype=self.dtype)
        P = tf.cast(self.P, dtype=self.dtype)
        m = tf.cast(self.m, dtype=self.dtype)

        # Propagate particles
        self.propagate()

        # m0 is the prior mean after dynamics
        m0, _ = self._empirical_mean_and_cov(self.particles)

        # If nonlinear observation provided and linearize_at_mean requested,
        # linearize at the *current* prior mean (m_prev) for this lambda
        # step to obtain an H and an effective observation y_eff.
        F = lin_F(self.f, tf.reshape(m, [self.state_dim]), dtype=self.dtype)

        # P_pred, m_pred represents prior after dynamics
        P_pred, m_pred = mP_predict(F, P, m, self.Q, self.R, self.jitter, dtype=self.dtype)

        # We'll step in lambda from 0 -> 1
        lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]

        # Loop over lambda steps and apply affine transforms using helpers
        for lam in lambdas:
            lam = tf.cast(lam, dtype=self.dtype)

            A, b_vec, H_curr = calculate_A_b(self.h, m0, y, lam, P_pred, R, self.jitter, self.state_dim, dtype=self.dtype)

            # Record per-lambda-step flow metrics computed at the linearization point
            # Norms (Frobenius / Euclidean) of A and b
            A_norm = tf.norm(A)
            b_norm = tf.norm(b_vec)
            # Jacobian of the per-step affine mapping is I + lam * A
            J_step = tf.eye(self.state_dim, dtype=self.dtype) + lam * A
            svals = tf.linalg.svd(J_step, compute_uv=False)
            # condition number: max(s)/min(s) (stabilize denominator)
            cond_eps = tf.cast(1e-12, dtype=self.dtype)
            jac_cond = svals[0] / (svals[-1] + cond_eps)

            # Snapshot particles before applying this lambda-step to compute average displacement
            particles_before = tf.identity(self.particles)

            for i in range(self.num_particles):
                xi = tf.reshape(self.particles[i], [self.state_dim])
                dxi = tf.matmul(A, tf.reshape(xi, [-1, 1])) + tf.reshape(b_vec, [-1, 1])
                xi = xi + lam * tf.reshape(dxi, [-1])
                self.particles = tf.tensor_scatter_nd_update(self.particles, [[i]], [xi])

            # Average displacement norm across particles for this lambda step
            disp = self.particles - particles_before
            per_particle_disp_norm = tf.norm(disp, axis=1)
            avg_disp = tf.reduce_mean(per_particle_disp_norm)

            # Append diagnostics (store tensors; notebook can convert to numpy later)
            self.flow_A_norm_history.append(A_norm)
            self.flow_b_norm_history.append(b_norm)
            self.flow_disp_norm_history.append(avg_disp)
            self.jacobian_cond_history.append(jac_cond)

            m0, _ = self._empirical_mean_and_cov(self.particles)

        # Final discrete Kalman-style update to produce posterior mean/cov
        # Use last linearization H_curr (from final lambda step) if available,
        # else linearize at final m0.
        try:
            H_final = H_curr
        except NameError:
            H_final = lin_H(self.h, m0, dtype=self.dtype)
        P, m = mP_update(P_pred, m_pred, H_final, R, y, self.jitter, dtype=self.dtype)
        # Evaluate \hat(x) 
        x_hat = self.get_state_estimate()
        cov = self.get_state_covariance()
        self.cov_history.append(cov)
        self.mean_history.append(x_hat)
        self.m = x_hat
        self.P = P
        return x_hat, P        

    def filter(self, observations, verbose: bool = False):
        """Filter a sequence of observations sequentially.

        For each observation y_t this method computes an approximate
        predictive log-likelihood using a linearization at the current
        mean, then calls `self.update(y_t)` to perform propagation and
        the particle flow update. Returns stacked tensors of posterior
        means, posterior covariances, and predictive log-likelihoods.
        """
        obs_arr = np.asarray(observations)
        if obs_arr.ndim == 0:
            obs_arr = np.expand_dims(obs_arr, axis=0)

        T = int(obs_arr.shape[0])
        means_list = []
        covs_list = []
        logliks_list = []

        for t in range(T):
            y_t = obs_arr[t]

            # Predictive mean and covariance using current m,P and linearization
            try:
                F = lin_F(self.f, tf.reshape(self.m, [self.state_dim]), dtype=self.dtype)
            except Exception:
                F = tf.eye(self.state_dim, dtype=self.dtype)

            P_pred, m_pred = mP_predict(F, tf.cast(self.P, dtype=self.dtype), tf.cast(self.m, dtype=self.dtype), self.Q, self.R, self.jitter, dtype=self.dtype)

            # Linearize observation at predicted mean for predictive likelihood
            try:
                H_pred = lin_H(self.h, tf.reshape(m_pred, [self.state_dim]), dtype=self.dtype)
            except Exception:
                H_pred = lin_H(self.h, tf.reshape(self.m, [self.state_dim]), dtype=self.dtype)

            y_loc = tf.reshape(tf.matmul(H_pred, tf.reshape(m_pred, [-1, 1])), [-1])
            S = H_pred @ P_pred @ tf.transpose(H_pred) + tf.cast(self.R, dtype=self.dtype)
            # stable cholesky
            chol_S = safe_cholesky(S, base_jitter=self.jitter, dtype=self.dtype)
            mvn = tfd.MultivariateNormalTriL(loc=y_loc, scale_tril=chol_S)
            y_vec = tf.convert_to_tensor(np.asarray(y_t), dtype=self.dtype)
            try:
                loglik = mvn.log_prob(y_vec)
            except Exception:
                # fallback: scalar cast
                loglik = mvn.log_prob(tf.reshape(y_vec, [-1]))

            # Run the full EDH update (propagate + flow + final Kalman update)
            x_hat, P_post = self.update(y=y_t)

            means_list.append(x_hat)
            covs_list.append(P_post)
            logliks_list.append(loglik)
            if verbose:
                try:
                    ll_val = float(loglik)
                except Exception:
                    ll_val = float(loglik.numpy()) if hasattr(loglik, 'numpy') else loglik
                print(f"EDH.filter t={t+1} mean={np.round(x_hat.numpy(),3)} loglik={ll_val}")

        means = tf.stack(means_list, axis=0)
        covs = tf.stack(covs_list, axis=0)
        logliks = tf.stack(logliks_list, axis=0)
        return means, covs, logliks


    def get_state_estimate(self):
        if self.particles is None:
            raise RuntimeError("Particles not initialized")
        return tf.reduce_mean(self.particles, axis=0)

    def get_state_covariance(self):
        if self.particles is None:
            raise RuntimeError("Particles not initialized")
        mean = self.get_state_estimate()
        centered = self.particles - mean
        return tf.matmul(centered, centered, transpose_a=True) / tf.cast(self.num_particles, self.dtype)


class LEDH:
    """Local Exact Daum-Huang flow: per-particle linearization and affine mapping.

    Inherits from `EDH` but performs linearization at each particle and computes
    per-particle affine mappings using the common helpers.
    """

    def __init__(self,
                 num_particles: int,
                 f: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 h: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 state_dim: int = None,
                 observation_dim: Optional[int] = None,
                 Q: Optional[tf.Tensor] = None,
                 R: Optional[tf.Tensor] = None,
                 dtype=tf.float32):
        
        self.num_particles = int(num_particles)
        # Backwards-compatible defaults: if `f` or `h` are not provided treat
        # them as identity maps; if `observation_dim` is not provided assume
        # observation_dim == state_dim.
        self.state_dim = int(state_dim)
        if observation_dim is None:
            observation_dim = self.state_dim
        self.observation_dim = int(observation_dim)
        self.f = f if f is not None else (lambda x: x)
        self.h = h if h is not None else (lambda x: x)
        self.dtype = dtype

        # Particles: shape [N, state_dim]
        self.particles: Optional[tf.Tensor] = None

        # Diagnostic quantities
        self.mean_history = []
        self.cov_history = []

        # Integration control for lambda in [0,1]
        self.n_flow_steps = 20
        self.jitter = 1e-6

        if Q is not None:
            self.Q = tf.cast(Q, dtype=self.dtype)
        else:
            self.Q = tf.eye(self.state_dim, dtype=self.dtype)

        if R is not None:
            self.R = tf.cast(R, dtype=self.dtype)
        else:
            self.R = tf.eye(self.observation_dim, dtype=self.dtype)
        self.R_inv = tf.linalg.inv(self.R)
    
    def initialize(self, particles: Optional[tf.Tensor] = None, initial_dist: Optional[tfd.Distribution] = None):
        """Initialize the particle cloud. Provide either `particles` tensor or an
        initial TFP `Distribution` to sample from.
        """
        if particles is None and initial_dist is None:
            raise ValueError("Either `particles` or `initial_dist` must be provided")

        if particles is None:
            p = initial_dist.sample(self.num_particles)
            self.particles = tf.cast(p, dtype=self.dtype)
        else:
            p = tf.convert_to_tensor(particles)
            self.particles = tf.cast(p, dtype=self.dtype)

        # store mean (`m`) and covariance (`P`) in the natural order
        self.m, self.P = self._empirical_mean_and_cov(self.particles)
        # Clear histories
        self.mean_history = []
        self.cov_history = []


    def propagate(self):
        """Propagate particles through a dynamics function and optionally add noise."""
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        self.particles = self.f(self.particles)
        # Only add process-noise when Q is non-zero (avoids cholesky on zero)
        if self.Q is not None:
            q_max = tf.reduce_max(tf.abs(self.Q))
            if tf.cast(q_max, self.dtype) > tf.cast(0.0, self.dtype):
                # noise = tf.random.normal(shape=(self.num_particles, self.state_dim), dtype=self.dtype)
                # # chol = tf.linalg.cholesky(tf.cast(self.Q, self.dtype))
                # # noise_scaled = tf.linalg.matmul(noise, chol)
                # # Add process noise samples
                q_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(self.state_dim, dtype=self.dtype), scale_tril=self.Q)
                # noise size (N, state_dim)
                noise = q_dist.sample(self.num_particles)
                noise = tf.cast(noise, dtype=self.dtype)
                self.particles = self.particles + noise

    def _empirical_mean_and_cov(self, particles: tf.Tensor):
        return compute_empirical_mean_and_cov(particles, self.jitter, dtype=self.dtype)

    def update(self, y: tf.Tensor):
        """Apply the Daum-Huang exact flow to move particles to posterior.

        Supports nonlinear observation functions via linearization at prior mean.
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        # Cast inputs
        y = tf.cast(tf.reshape(y, [-1]), dtype=self.dtype)
        if self.R is None:
            raise ValueError("Observation covariance R must be provided")
        R = tf.cast(self.R, dtype=self.dtype)
        P = tf.cast(self.P, dtype=self.dtype)
        m = tf.cast(self.m, dtype=self.dtype)

        # Empirical prior mean and covariance from current particles
        # m0, P0 = self._empirical_mean_and_cov(self.particles)

        # If nonlinear observation provided and linearize_at_mean requested,
        # linearize at the *current* prior mean (m_prev) for this lambda
        # step to obtain an H and an effective observation y_eff.
        F = lin_F(self.f, tf.reshape(m, [self.state_dim]), dtype=self.dtype)

        # P_pred, m_pred represents prior after dynamics
        P_pred, m_pred = mP_predict(F, P, m, self.Q, self.R, self.jitter, dtype=self.dtype)

        # Propagate particles
        self.propagate()

        # m0 is the prior mean after dynamics
        m0, _ = self._empirical_mean_and_cov(self.particles)

        # We'll step in lambda from 0 -> 1
        lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]

        # Loop over lambda steps and apply affine transforms using helpers
        for lam in lambdas:
            lam = tf.cast(lam, dtype=self.dtype)

            for i in range(self.num_particles):
                xi = tf.reshape(self.particles[i], [self.state_dim])
                A, b_vec, H_curr = calculate_A_b(self.h, xi, y, lam, P_pred, R, self.jitter, self.state_dim, dtype=self.dtype)
                dxi = tf.matmul(A, tf.reshape(xi, [-1, 1])) + tf.reshape(b_vec, [-1, 1])
                xi = xi + lam * tf.reshape(dxi, [-1])
                self.particles = tf.tensor_scatter_nd_update(self.particles, [[i]], [xi])

            m0, _ = self._empirical_mean_and_cov(self.particles)

        # Final discrete Kalman-style update to produce posterior mean/cov
        # Use last linearization H_curr (from final lambda step) if available,
        # else linearize at final m0.
        try:
            H_final = H_curr
        except NameError:
            H_final = lin_H(self.h, m0, dtype=self.dtype)
        P, m = mP_update(P_pred, m_pred, H_final, R, y, self.jitter, dtype=self.dtype)
        # Evaluate \hat(x) 
        x_hat = self.get_state_estimate()
        cov = self.get_state_covariance()
        self.cov_history.append(cov)
        self.mean_history.append(x_hat)
        self.m = x_hat
        self.P = P
        return x_hat, P
    
    def filter(self, observations, verbose: bool = False):
        """Filter a sequence of observations sequentially (LEDH version).

        For each observation y_t this method computes an approximate
        predictive log-likelihood using a linearization at the current
        mean, then calls `self.update(y_t)` to perform propagation and
        the local particle flow update. Returns stacked tensors of posterior
        means, posterior covariances, and predictive log-likelihoods.
        """
        obs_arr = np.asarray(observations)
        if obs_arr.ndim == 0:
            obs_arr = np.expand_dims(obs_arr, axis=0)

        T = int(obs_arr.shape[0])
        means_list = []
        covs_list = []
        logliks_list = []

        for t in range(T):
            y_t = obs_arr[t]

            # Predictive mean and covariance using current m,P and linearization
            try:
                F = lin_F(self.f, tf.reshape(self.m, [self.state_dim]), dtype=self.dtype)
            except Exception:
                F = tf.eye(self.state_dim, dtype=self.dtype)

            P_pred, m_pred = mP_predict(F, tf.cast(self.P, dtype=self.dtype), tf.cast(self.m, dtype=self.dtype), self.Q, self.R, self.jitter, dtype=self.dtype)

            # Linearize observation at predicted mean for predictive likelihood
            try:
                H_pred = lin_H(self.h, tf.reshape(m_pred, [self.state_dim]), dtype=self.dtype)
            except Exception:
                H_pred = lin_H(self.h, tf.reshape(self.m, [self.state_dim]), dtype=self.dtype)

            y_loc = tf.reshape(tf.matmul(H_pred, tf.reshape(m_pred, [-1, 1])), [-1])
            S = H_pred @ P_pred @ tf.transpose(H_pred) + tf.cast(self.R, dtype=self.dtype)
            chol_S = safe_cholesky(S, base_jitter=self.jitter, dtype=self.dtype)
            mvn = tfd.MultivariateNormalTriL(loc=y_loc, scale_tril=chol_S)
            y_vec = tf.convert_to_tensor(np.asarray(y_t), dtype=self.dtype)
            try:
                loglik = mvn.log_prob(y_vec)
            except Exception:
                loglik = mvn.log_prob(tf.reshape(y_vec, [-1]))

            # Run the LEDH update
            x_hat, P_post = self.update(y=y_t)

            means_list.append(x_hat)
            covs_list.append(P_post)
            logliks_list.append(loglik)
            if verbose:
                try:
                    ll_val = float(loglik)
                except Exception:
                    ll_val = float(loglik.numpy()) if hasattr(loglik, 'numpy') else loglik
                print(f"LEDH.filter t={t+1} mean={np.round(x_hat.numpy(),3)} loglik={ll_val}")

        means = tf.stack(means_list, axis=0)
        covs = tf.stack(covs_list, axis=0)
        logliks = tf.stack(logliks_list, axis=0)
        return means, covs, logliks

    def get_state_estimate(self):
        if self.particles is None:
            raise RuntimeError("Particles not initialized")
        return tf.reduce_mean(self.particles, axis=0)

    def get_state_covariance(self):
        if self.particles is None:
            raise RuntimeError("Particles not initialized")
        mean = self.get_state_estimate()
        centered = self.particles - mean
        return tf.matmul(centered, centered, transpose_a=True) / tf.cast(self.num_particles, self.dtype)




# if __name__ == "__main__":
#     # Small sequential demo: run T observations and update the flow each time
#     tf.random.set_seed(1)

#     N = 100
#     d = 2
#     dh = 2

#     # Observation model y = H x + noise
#     H = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float64)
#     R = tf.eye(tf.shape(H)[0], dtype=tf.float64) * 1.0
#     Q = tf.eye(d, dtype=tf.float64) * 1.0

#     # Prior: N(0, I)
#     prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
#     dhf = EDH(num_particles=N, state_dim=d, R=R, Q=Q, dtype=tf.float64)
#     dhf.initialize(initial_dist=prior)

#     # Static true state and T sequential noisy observations
#     x_t = tf.constant([0.0, 0.0], dtype=tf.float64)
#     T = 10
#     dhf.n_flow_steps = 6

#     print("Running sequential demo (T=", T, ")")
#     for t in range(T):
#         x_t = x_t + tf.random.normal([dh], stddev=np.sqrt(R.numpy()[0,0]), dtype=tf.float64)
#         y_t = tf.reshape(x_t, [-1])
#         dhf.update(y=y_t)
#         est = dhf.get_state_estimate().numpy()
#         cov = dhf.get_state_covariance().numpy()
#         print(f"t={t+1:2d}  true x = {x_t.numpy()} y={y_t.numpy()}  est={est}  cov_diag={np.diag(cov)}")

#     print("Final estimate:", est)
#     print("Final covariance:", cov)
if __name__ == "__main__":
    """Small self-test:

    - Creates a linear-Gaussian observation model y = H x + noise
    - Samples a small prior particle cloud
    - Runs `EDH.update(y)` and `LEDH.update(y)` and compares the analytic
        posterior mean/covariance produced by the two implementations.

    This is intentionally small and prints PASS/FAIL rather than aborting on
    assertions so it can be run as a script.
    """
    import sys

    tf.random.set_seed(1)

    N = 100
    d = 3

    # Linear observation H and covariance R
    H = tf.constant(np.eye(d), dtype=tf.float64)
    R = tf.constant(np.eye(d) * 1.0, dtype=tf.float64)
    Q = tf.constant(np.eye(d) * 1.0, dtype=tf.float64)

    # Build a simple observation function h(x) = H x
    def make_h(H):
        return lambda x: tf.reshape(tf.matmul(H, tf.reshape(x, [-1, 1])), [-1])

    h_fn = make_h(H)

    # Prior particle cloud (Gaussian)
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d, dtype=tf.float64), scale_diag=tf.ones(d, dtype=tf.float64))
    particles = prior.sample(N)

    # Create filters
    edh = EDH(num_particles=N, f=None, h=h_fn, state_dim=d, observation_dim=d, R=R, Q=Q, dtype=tf.float64)
    ledh = LEDH(num_particles=N, f=None, h=h_fn, state_dim=d, observation_dim=d, R=R, Q=Q, dtype=tf.float64)

    edh.initialize(particles=particles)
    ledh.initialize(particles=particles)

    edh.n_flow_steps = 5
    ledh.n_flow_steps = 5

    # sequential test: T=5, propagate true state with additive process noise drawn from Q
    true_x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
    T = 5

    # Distributions for process and measurement noise
    Q_tril = tf.linalg.cholesky(Q + tf.eye(d, dtype=tf.float64) * 1e-12)
    process_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(d, dtype=tf.float64), scale_tril=Q_tril)
    R_tril = tf.linalg.cholesky(R + tf.eye(d, dtype=tf.float64) * 1e-12)
    meas_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(d, dtype=tf.float64), scale_tril=R_tril)

    print(f"Running sequential self-test T={T} with process noise ~ N(0,Q)")
    for t in range(T):
        # propagate true state: x_{t+1} = x_t + w_t, w_t ~ N(0, Q)
        w_t = process_dist.sample()
        true_x = true_x + w_t

        # observation with measurement noise
        v_t = meas_dist.sample()
        y = true_x + v_t

        # run both filters
        edh_hat_tf, edh_P_tf = edh.update(y=y)
        ledh_hat_tf, ledh_P_tf = ledh.update(y=y)

        edh_hat = edh_hat_tf.numpy()
        edh_P = edh_P_tf.numpy()
        ledh_hat = ledh_hat_tf.numpy()
        ledh_P = ledh_P_tf.numpy()

        print(f"t={t+1:2d} true_x={true_x.numpy()} y={y.numpy()}")
        print(f"  EDH est={edh_hat}  LEDH est={ledh_hat}")
        print(f"  EDH P diag={np.diag(edh_P)}  LEDH P diag={np.diag(ledh_P)}")

    # Final comparison
    ok_mean = np.allclose(edh_hat, ledh_hat, atol=1e-1, rtol=1e-6)
    ok_cov = np.allclose(edh_P, ledh_P, atol=1e-1, rtol=1e-6)

    if ok_mean and ok_cov:
        print("SELF-TEST: PASS — EDH and LEDH analytic posteriors close (after T steps)")
        sys.exit(0)
    else:
        print("SELF-TEST: FAIL — differences detected (after T steps)")
        sys.exit(2)
