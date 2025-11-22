# """
# Exact Daum-Huang particle flow (affine/linear flow) implementation.

# This implements the Exact Daum-Huang flow for linear-Gaussian likelihoods:
#   y ~ N(H x, R)
# Given an empirical prior represented by particles with mean m0 and covariance P0,
# we transport particles deterministically through a sequence of affine maps so that
# at each intermediate pseudo-time \lambda \in [0,1] the particle cloud matches
# the intermediate Gaussian with mean m(\lambda) and covariance P(\lambda), where

#   P(\lambda) = (P0^{-1} + \lambda H^T R^{-1} H)^{-1}
#   m(\lambda) = P(\lambda) (P0^{-1} m0 + \lambda H^T R^{-1} y)

# We update particles in small steps in \lambda and apply the affine transform

#   x_{new} = m(\lambda) + A (x_old - m_prev),

# where A = chol(P(\lambda)) @ inv(chol(P_prev)). This preserves the Gaussian
# marginals exactly in the linear-Gaussian case.

# This implementation uses TensorFlow for vectorized matrix operations.

# References:
# - Daum, F., & Huang, J. (2008). "Particle flow for nonlinear filters" (and later work)
# """

# import os

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# from typing import Callable, Optional
# import numpy as np
# import tensorflow as tf
# import tensorflow_probability as tfp

# tfd = tfp.distributions


# class LEDH:
#     """
#     Local Exact Daum-Huang affine particle flow for linear-Gaussian likelihoods.

#     This class assumes the observation model is linear:
#         y = H x + noise,   noise ~ N(0, R)

#     and the prior is represented by particles (empirical mean and covariance).

#     Usage:
#       - Create with number of particles and state dim
#       - Initialize with particles (or a distribution sample)
#       - Call `predict` to propagate particles through dynamics (optional)
#       - Call `update` with observation `y`, observation matrix `H`, and `R`

#     Note: This flow is exact for linear-Gaussian likelihoods (and Gaussian prior).
#     For nonlinear observation functions you would need a local linearization.
#     """

#     def __init__(self, num_particles: int, state_dim: int, dtype=tf.float32):
#         self.num_particles = int(num_particles)
#         self.state_dim = int(state_dim)
#         self.dtype = dtype

#         # Particles: shape [N, state_dim]
#         self.particles: Optional[tf.Tensor] = None

#         # Diagnostic quantities
#         self.mean_history = []
#         self.cov_history = []

#         # Integration control for lambda in [0,1]
#         self.n_flow_steps = 20
#         self.jitter = 1e-6

#     def initialize(self, particles: Optional[tf.Tensor] = None, initial_dist: Optional[tfd.Distribution] = None):
#         """
#         Initialize the particle cloud. Provide either `particles` tensor or an
#         initial TFP `Distribution` to sample from.

#         Args:
#             particles: tf.Tensor shape [N, state_dim]
#             initial_dist: tfd.Distribution with .sample([N]) producing shape [N, state_dim]
#         """
#         if particles is None and initial_dist is None:
#             raise ValueError("Either `particles` or `initial_dist` must be provided")

#         if particles is None:
#             p = initial_dist.sample(self.num_particles)
#             self.particles = tf.cast(p, dtype=self.dtype)
#         else:
#             p = tf.convert_to_tensor(particles)
#             self.particles = tf.cast(p, dtype=self.dtype)

#         # Clear histories
#         self.mean_history = []
#         self.cov_history = []

#     def predict(self, dynamics_fn: Callable[[tf.Tensor], tf.Tensor], process_noise_cov: Optional[tf.Tensor] = None):
#         """
#         Propagate particles through the dynamics function and optionally add process noise.

#         Args:
#             dynamics_fn: callable mapping state tensor [N, d] -> [N, d]
#             process_noise_cov: optional [d, d] covariance (tf.Tensor)
#         """
#         if self.particles is None:
#             raise RuntimeError("Particles not initialized")

#         self.particles = dynamics_fn(self.particles)

#         if process_noise_cov is not None:
#             noise = tf.random.normal(shape=(self.num_particles, self.state_dim), dtype=self.dtype)
#             chol = tf.linalg.cholesky(tf.cast(process_noise_cov, self.dtype))
#             noise_scaled = tf.linalg.matmul(noise, chol)
#             self.particles = self.particles + noise_scaled

#     def _empirical_mean_and_cov(self, particles: tf.Tensor):
#         N = tf.cast(tf.shape(particles)[0], self.dtype)
#         mean = tf.reduce_mean(particles, axis=0, keepdims=False)
#         centered = particles - mean
#         cov = (tf.matmul(centered, centered, transpose_a=True) / N)
#         # Add jitter for numerical stability
#         cov += tf.eye(self.state_dim, dtype=self.dtype) * tf.cast(self.jitter, self.dtype)
#         return mean, cov

#     def update(self, y: tf.Tensor, H: Optional[tf.Tensor] = None, R: Optional[tf.Tensor] = None, h_func: Optional[Callable[[tf.Tensor], tf.Tensor]] = None):
#         """
#         Apply the Local Exact Daum-Huang flow to move particles to posterior.

#         This implements LEDH: each particle is locally linearized using the
#         Jacobian of the observation function `h_func` at that particle. For
#         purely linear observation models you may pass `H` (matrix) instead.

#         Parameters
#         ----------
#         y : tf.Tensor
#             Observation vector shape [obs_dim] or [obs_dim, 1]
#         H : tf.Tensor, optional
#             Linear observation matrix (obs_dim x state_dim). If provided, a
#             shared linearization is used for all particles.
#         R : tf.Tensor
#             Observation noise covariance (obs_dim x obs_dim)
#         h_func : callable, optional
#             Observation function h(x) -> y. If provided, the Jacobian of
#             `h_func` at each particle is used as the local H_i.
#         """
#         if self.particles is None:
#             raise RuntimeError("Particles not initialized")

#         # Cast inputs
#         y = tf.cast(tf.reshape(y, [-1]), dtype=self.dtype)
#         if R is None:
#             raise ValueError("Observation covariance R must be provided")
#         R = tf.cast(R, dtype=self.dtype)

#         # Empirical prior mean and covariance from current particles
#         m0, P0 = self._empirical_mean_and_cov(self.particles)

#         # Precompute common inverses
#         P0_inv = tf.linalg.inv(P0)
#         R_inv = tf.linalg.inv(R)

#         # Number of particles
#         N = int(self.num_particles)

#         # Determine per-particle linearization H_i
#         if h_func is not None:
#             # Compute Jacobian of h_func at each particle (obs_dim x state_dim)
#             H_list = []
#             for i in range(N):
#                 x_i = tf.reshape(self.particles[i], [self.state_dim])
#                 with tf.GradientTape() as tape:
#                     tape.watch(x_i)
#                     y_i = tf.convert_to_tensor(h_func(x_i), dtype=self.dtype)
#                 J = tape.jacobian(y_i, x_i)
#                 J = tf.reshape(J, (tf.shape(y_i)[0], tf.shape(x_i)[0]))
#                 H_list.append(J)
#             H_stack = tf.stack(H_list, axis=0)  # shape (N, obs_dim, state_dim)
#         else:
#             if H is None:
#                 raise ValueError("Either H matrix or h_func callable must be provided")
#             H = tf.cast(H, dtype=self.dtype)
#             # same H for all particles
#             H_stack = tf.tile(tf.expand_dims(H, axis=0), [N, 1, 1])

#         # Prepare per-particle previous mean/cov (start from global prior)
#         m_prev = [m0 for _ in range(N)]
#         P_prev = [P0 for _ in range(N)]

#         # Current particles as list for in-place updates
#         x_list = [tf.reshape(self.particles[i], [-1]) for i in range(N)]

#         # Step through lambda and apply local affine transforms per particle
#         lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]
#         for lam in lambdas:
#             lam = tf.cast(lam, dtype=self.dtype)
#             for i in range(N):
#                 Hi = tf.cast(H_stack[i], dtype=self.dtype)  # (obs_dim, state_dim)
#                 Ht_Rinv = tf.matmul(tf.transpose(Hi), R_inv)
#                 S = P0_inv + lam * tf.matmul(Ht_Rinv, Hi)  # (d,d)
#                 P_lam = tf.linalg.inv(S)

#                 rhs = tf.matmul(P0_inv, tf.reshape(m0, [-1, 1])) + lam * tf.matmul(Ht_Rinv, tf.reshape(y, [-1, 1]))
#                 m_lam = tf.reshape(tf.matmul(P_lam, rhs), [-1])

#                 chol_prev = tf.linalg.cholesky(P_prev[i])
#                 chol_lam = tf.linalg.cholesky(P_lam)
#                 I = tf.eye(self.state_dim, dtype=self.dtype)
#                 inv_chol_prev = tf.linalg.triangular_solve(chol_prev, I, lower=True)
#                 A = tf.matmul(chol_lam, inv_chol_prev)

#                 centered = x_list[i] - m_prev[i]
#                 updated = tf.matmul(A, tf.reshape(centered, [-1, 1]))
#                 updated = tf.reshape(updated, [-1]) + m_lam

#                 x_list[i] = updated

#                 # store histories
#                 self.mean_history.append(m_lam.numpy())
#                 self.cov_history.append(P_lam.numpy())

#                 # advance per-particle
#                 m_prev[i] = m_lam
#                 P_prev[i] = P_lam

#         # Reassemble particle tensor
#         x_new = tf.stack(x_list, axis=0)
#         self.particles = tf.cast(x_new, dtype=self.dtype)

#     def get_state_estimate(self):
#         if self.particles is None:
#             raise RuntimeError("Particles not initialized")
#         return tf.reduce_mean(self.particles, axis=0)

#     def get_state_covariance(self):
#         if self.particles is None:
#             raise RuntimeError("Particles not initialized")
#         mean = self.get_state_estimate()
#         centered = self.particles - mean
#         return tf.matmul(centered, centered, transpose_a=True) / tf.cast(self.num_particles, self.dtype)


# if __name__ == "__main__":
#     # Quick self-test on a 1D linear-Gaussian example
#     tf.random.set_seed(1)

#     N = 1000
#     d = 1
#     dh = 1

#     # Prior: N(0, 1)
#     prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
#     dhf = LEDH(num_particles=N, state_dim=d, dtype=tf.float64)
#     dhf.initialize(initial_dist=prior)

#     # Observation model y = H x + noise
#     H = tf.constant([[1.0]], dtype=tf.float64)
#     R = tf.constant([[0.5]], dtype=tf.float64)

#     # Synthetic true state and observation
#     true_x = tf.constant([2.0], dtype=tf.float64)
#     y = tf.reshape(true_x + tf.random.normal([dh], stddev=np.sqrt(R.numpy()[0,0]), dtype=tf.float64), [-1])

#     # Run flow
#     dhf.n_flow_steps = 10
#     dhf.update(y=y, H=H, R=R)

#     est = dhf.get_state_estimate().numpy()
#     cov = dhf.get_state_covariance().numpy()
#     print("True x:", true_x.numpy())
#     print("Estimated mean:", est)
#     print("Estimated covariance (empirical):", cov)
#     print("Posterior mean (analytic):")
#     # analytic posterior (from prior N(0,1) and y observation)
#     P0 = np.array([[1.0]])
#     P_post = np.linalg.inv(np.linalg.inv(P0) + H.numpy().T @ np.linalg.inv(R.numpy()) @ H.numpy())
#     m_post = (P_post @ (np.linalg.inv(P0) @ np.zeros((d,)) + H.numpy().T @ np.linalg.inv(R.numpy()) @ y.numpy()))
#     print(m_post, P_post)
