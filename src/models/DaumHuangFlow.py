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

from typing import Callable, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class DaumHuangFlow:
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

    def __init__(self, num_particles: int, state_dim: int, dtype=tf.float32):
        self.num_particles = int(num_particles)
        self.state_dim = int(state_dim)
        self.dtype = dtype

        # Particles: shape [N, state_dim]
        self.particles: Optional[tf.Tensor] = None

        # Diagnostic quantities
        self.mean_history = []
        self.cov_history = []

        # Integration control for lambda in [0,1]
        self.n_flow_steps = 20
        self.jitter = 1e-6

    def initialize(self, particles: Optional[tf.Tensor] = None, initial_dist: Optional[tfd.Distribution] = None):
        """
        Initialize the particle cloud. Provide either `particles` tensor or an
        initial TFP `Distribution` to sample from.

        Args:
            particles: tf.Tensor shape [N, state_dim]
            initial_dist: tfd.Distribution with .sample([N]) producing shape [N, state_dim]
        """
        if particles is None and initial_dist is None:
            raise ValueError("Either `particles` or `initial_dist` must be provided")

        if particles is None:
            p = initial_dist.sample(self.num_particles)
            self.particles = tf.cast(p, dtype=self.dtype)
        else:
            p = tf.convert_to_tensor(particles)
            self.particles = tf.cast(p, dtype=self.dtype)

        # Clear histories
        self.mean_history = []
        self.cov_history = []

    def predict(self, dynamics_fn: Callable[[tf.Tensor], tf.Tensor], process_noise_cov: Optional[tf.Tensor] = None):
        """
        Propagate particles through the dynamics function and optionally add process noise.

        Args:
            dynamics_fn: callable mapping state tensor [N, d] -> [N, d]
            process_noise_cov: optional [d, d] covariance (tf.Tensor)
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        self.particles = dynamics_fn(self.particles)

        if process_noise_cov is not None:
            noise = tf.random.normal(shape=(self.num_particles, self.state_dim), dtype=self.dtype)
            chol = tf.linalg.cholesky(tf.cast(process_noise_cov, self.dtype))
            noise_scaled = tf.linalg.matmul(noise, chol)
            self.particles = self.particles + noise_scaled

    def _empirical_mean_and_cov(self, particles: tf.Tensor):
        N = tf.cast(tf.shape(particles)[0], self.dtype)
        mean = tf.reduce_mean(particles, axis=0, keepdims=False)
        centered = particles - mean
        cov = (tf.matmul(centered, centered, transpose_a=True) / N)
        # Add jitter for numerical stability
        cov += tf.eye(self.state_dim, dtype=self.dtype) * tf.cast(self.jitter, self.dtype)
        return mean, cov

    def update(self, y: tf.Tensor, H: tf.Tensor, R: tf.Tensor):
        """
        Apply the Daum-Huang exact flow to move particles to posterior for
        linear observation y = H x + noise, noise ~ N(0, R).

        Args:
            y: observation vector shape [obs_dim] or [obs_dim, 1]
            H: observation matrix shape [obs_dim, state_dim]
            R: observation covariance shape [obs_dim, obs_dim]
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        # Cast inputs
        y = tf.cast(tf.reshape(y, [-1]), dtype=self.dtype)
        H = tf.cast(H, dtype=self.dtype)
        R = tf.cast(R, dtype=self.dtype)

        # Empirical prior mean and covariance from current particles
        m0, P0 = self._empirical_mean_and_cov(self.particles)

        # Precompute inverses
        P0_inv = tf.linalg.inv(P0)
        R_inv = tf.linalg.inv(R)

        # We'll step in lambda from 0 -> 1
        lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]  # exclude 0.0

        # Initialize previous mean/cov to prior
        m_prev = m0
        P_prev = P0

        x = self.particles

        # Loop over lambda steps and apply affine transforms
        for lam in lambdas:
            lam = tf.cast(lam, dtype=self.dtype)

            # Compute P(lambda) = inv(P0_inv + lam H^T R^{-1} H)
            Ht_Rinv = tf.matmul(tf.transpose(H), R_inv)
            S = P0_inv + lam * tf.matmul(Ht_Rinv, H)  # [d,d]
            P_lam = tf.linalg.inv(S)

            # Compute m(lambda) = P(lambda) (P0_inv m0 + lam H^T R^{-1} y)
            rhs = tf.matmul(P0_inv, tf.reshape(m0, [-1, 1])) + lam * tf.matmul(Ht_Rinv, tf.reshape(y, [-1, 1]))
            m_lam = tf.reshape(tf.matmul(P_lam, rhs), [-1])

            # Compute affine transform A mapping N(m_prev, P_prev) -> N(m_lam, P_lam)
            # A = chol(P_lam) @ inv(chol(P_prev))
            chol_prev = tf.linalg.cholesky(P_prev)
            chol_lam = tf.linalg.cholesky(P_lam)

            # Solve for inv(chol_prev) efficiently using triangular solve
            # We want A = chol_lam @ inv(chol_prev). For a vector v: inv(chol_prev) v = triangular_solve(chol_prev, v, lower=True)
            # We'll apply to (x - m_prev)^T when updating particles.
            # Build transformation matrix A explicitly (state_dim small typically)
            I = tf.eye(self.state_dim, dtype=self.dtype)
            inv_chol_prev = tf.linalg.triangular_solve(chol_prev, I, lower=True)
            A = tf.matmul(chol_lam, inv_chol_prev)

            # Apply affine transform to particles: x_new = m_lam + A @ (x - m_prev)
            centered = tf.transpose(tf.transpose(x) - m_prev)  # shape [N, d]
            # Compute A @ centered^T -> result [d, N], transpose back
            updated = tf.transpose(tf.matmul(A, tf.transpose(centered))) + m_lam

            x = updated

            # Store histories and advance
            self.mean_history.append(m_lam.numpy())
            self.cov_history.append(P_lam.numpy())
            m_prev = m_lam
            P_prev = P_lam

        # Assign updated particles
        self.particles = tf.cast(x, dtype=self.dtype)

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


if __name__ == "__main__":
    # Quick self-test on a 1D linear-Gaussian example
    tf.random.set_seed(1)

    N = 1000
    d = 1
    dh = 1

    # Prior: N(0, 1)
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
    dhf = DaumHuangFlow(num_particles=N, state_dim=d, dtype=tf.float64)
    dhf.initialize(initial_dist=prior)

    # Observation model y = H x + noise
    H = tf.constant([[1.0]], dtype=tf.float64)
    R = tf.constant([[0.5]], dtype=tf.float64)

    # Synthetic true state and observation
    true_x = tf.constant([2.0], dtype=tf.float64)
    y = tf.reshape(true_x + tf.random.normal([dh], stddev=np.sqrt(R.numpy()[0,0]), dtype=tf.float64), [-1])

    # Run flow
    dhf.n_flow_steps = 10
    dhf.update(y=y, H=H, R=R)

    est = dhf.get_state_estimate().numpy()
    cov = dhf.get_state_covariance().numpy()
    print("True x:", true_x.numpy())
    print("Estimated mean:", est)
    print("Estimated covariance (empirical):", cov)
    print("Posterior mean (analytic):")
    # analytic posterior (from prior N(0,1) and y observation)
    P0 = np.array([[1.0]])
    P_post = np.linalg.inv(np.linalg.inv(P0) + H.numpy().T @ np.linalg.inv(R.numpy()) @ H.numpy())
    m_post = (P_post @ (np.linalg.inv(P0) @ np.zeros((d,)) + H.numpy().T @ np.linalg.inv(R.numpy()) @ y.numpy()))
    print(m_post, P_post)
