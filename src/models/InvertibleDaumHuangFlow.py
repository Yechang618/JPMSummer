# """
# Invertible Daum-Huang particle flow utilities.

# This module provides a small helper that performs the affine Daum-Huang
# flow (same math as `DaumHuangFlow`) but additionally returns the
# log-absolute-determinant of the overall mapping so it can be used in
# importance-weight corrections (as in PF-PF / invertible particle flow).

# The affine transforms at each pseudo-time step are of the form
#     x_new = m_lam + A (x_old - m_prev)
# and the Jacobian is A for that step; the total Jacobian determinant is
# product(det(A_step)) and its log is the sum of logdet(A_step).

# This implementation returns the mapped particles and the scalar
# `logabsdet` (same for all particles because the mapping is affine).

# Note: this implementation assumes a linear observation model y = H x + noise
# with noise covariance R. For nonlinear observation functions, a local
# linearization would be required.
# """

# from typing import Optional, Tuple
# import numpy as np
# import tensorflow as tf
# import tensorflow_probability as tfp

# tfd = tfp.distributions


# class InvertibleDaumHuangFlow:
#     """Perform affine Daum-Huang flow and return mapping + log-determinant.

#     Parameters
#     ----------
#     n_flow_steps : int
#         Number of lambda integration steps (default 20).
#     jitter : float
#         Small diagonal jitter added to empirical covariances for stability.
#     dtype : tf.DType
#         TensorFlow dtype to use (default tf.float64).
#     """

#     def __init__(self, n_flow_steps: int = 20, jitter: float = 1e-6, dtype=tf.float64):
#         self.n_flow_steps = int(n_flow_steps)
#         self.jitter = float(jitter)
#         self.dtype = dtype

#     def _empirical_mean_and_cov(self, particles: tf.Tensor):
#         N = tf.cast(tf.shape(particles)[0], self.dtype)
#         mean = tf.reduce_mean(particles, axis=0)
#         centered = particles - mean
#         cov = tf.matmul(centered, centered, transpose_a=True) / N
#         cov += tf.eye(tf.shape(particles)[1], dtype=self.dtype) * tf.cast(self.jitter, self.dtype)
#         return mean, cov

#     def flow(self, particles: tf.Tensor, y: tf.Tensor, H: tf.Tensor, R: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
#         """Apply Daum-Huang affine flow to `particles` and return (x_mapped, logabsdet).

#         Parameters
#         ----------
#         particles : tf.Tensor, shape [N, d]
#         y : tf.Tensor, shape [obs_dim] or [obs_dim,]
#         H : tf.Tensor, shape [obs_dim, d]
#         R : tf.Tensor, shape [obs_dim, obs_dim]

#         Returns
#         -------
#         x_mapped : tf.Tensor, shape [N, d]
#         logabsdet : tf.Tensor scalar (dtype=self.dtype)
#             Log absolute determinant of the overall mapping (same for all particles)
#         """
#         x = tf.convert_to_tensor(particles)
#         x = tf.cast(x, dtype=self.dtype)
#         y = tf.convert_to_tensor(y)
#         y = tf.cast(tf.reshape(y, [-1]), dtype=self.dtype)
#         H = tf.convert_to_tensor(H)
#         H = tf.cast(H, dtype=self.dtype)
#         R = tf.convert_to_tensor(R)
#         R = tf.cast(R, dtype=self.dtype)

#         # Empirical prior mean and covariance
#         m0, P0 = self._empirical_mean_and_cov(x)
#         P0_inv = tf.linalg.inv(P0)
#         R_inv = tf.linalg.inv(R)

#         lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]

#         m_prev = m0
#         P_prev = P0

#         total_logabsdet = tf.constant(0.0, dtype=self.dtype)

#         for lam in lambdas:
#             lam = tf.cast(lam, dtype=self.dtype)

#             # Compute P(lambda) = inv(P0_inv + lam H^T R^{-1} H)
#             Ht_Rinv = tf.matmul(tf.transpose(H), R_inv)
#             S = P0_inv + lam * tf.matmul(Ht_Rinv, H)
#             P_lam = tf.linalg.inv(S)

#             # Compute m(lambda) = P(lambda) (P0_inv m0 + lam H^T R^{-1} y)
#             rhs = tf.matmul(P0_inv, tf.reshape(m0, [-1, 1])) + lam * tf.matmul(Ht_Rinv, tf.reshape(y, [-1, 1]))
#             m_lam = tf.reshape(tf.matmul(P_lam, rhs), [-1])

#             # Compute affine mapping matrix A
#             chol_prev = tf.linalg.cholesky(P_prev)
#             chol_lam = tf.linalg.cholesky(P_lam)

#             # A = chol_lam @ inv(chol_prev)
#             I = tf.eye(tf.shape(chol_prev)[0], dtype=self.dtype)
#             inv_chol_prev = tf.linalg.triangular_solve(chol_prev, I, lower=True)
#             A = tf.matmul(chol_lam, inv_chol_prev)

#             # Update x by affine mapping
#             centered = x - m_prev
#             x = tf.linalg.matmul(centered, tf.transpose(A)) + m_lam

#             # logabsdet contribution: log|det(A)|
#             # For triangular Cholesky matrices, det(A) = det(chol_lam) / det(chol_prev)
#             # logdet(A) = sum(log(diag(chol_lam))) - sum(log(diag(chol_prev)))
#             logdet_chol_lam = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(chol_lam))))
#             logdet_chol_prev = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(chol_prev))))
#             logabsdet_A = logdet_chol_lam - logdet_chol_prev
#             total_logabsdet += tf.cast(logabsdet_A, dtype=self.dtype)

#             # advance
#             m_prev = m_lam
#             P_prev = P_lam

#         return x, total_logabsdet


# if __name__ == "__main__":
#     # tiny self-check: mapping determinant should match analytic difference of post/prior cholesky
#     import numpy as np

#     tf.random.set_seed(1)
#     N = 1000
#     d = 1
#     prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d), scale_diag=tf.ones(d))
#     particles = prior.sample(N)

#     H = tf.constant([[1.0]], dtype=tf.float64)
#     R = tf.constant([[0.5]], dtype=tf.float64)
#     true_x = tf.constant([2.0], dtype=tf.float64)
#     y = tf.reshape(true_x + tf.random.normal([1], stddev=np.sqrt(R.numpy()[0,0]), dtype=tf.float64), [-1])

#     flow = InvertibleDaumHuangFlow(n_flow_steps=8, dtype=tf.float64)
#     x_mapped, logabsdet = flow.flow(particles, y, H, R)
#     print("mapped mean:", tf.reduce_mean(x_mapped, axis=0).numpy())
#     print("logabsdet:", float(logabsdet.numpy()))
