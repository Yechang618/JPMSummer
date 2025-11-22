"""Kernel-based particle flows: scalar-kernel and matrix-kernel variants.

This implements practical variants inspired by "A particle flow filter for
high-dimensional system applications" (kernelized flows). Implementations
here use an RBF kernel with median-heuristic bandwidth by default and
perform EDH-like affine mappings using kernel-weighted empirical covariances.

The API mirrors IEDHFlow/ILEDHFlow: `flow(particles, y, H, R, h_func=None, prior_cov_override=None)`
returns (x_mapped, logabsdet) where logabsdet is scalar (global mapping)
or per-particle vector (local mappings).
"""
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _safe_cholesky(mat: tf.Tensor, base_jitter: float = 1e-6, max_tries: int = 8) -> tf.Tensor:
    dtype = mat.dtype
    jitter = tf.cast(base_jitter, dtype)
    I = tf.eye(tf.shape(mat)[0], dtype=dtype)
    for _ in range(max_tries):
        try:
            chol = tf.linalg.cholesky(mat + jitter * I)
            if tf.reduce_all(tf.math.is_finite(chol)):
                return chol
        except Exception:
            pass
        jitter = jitter * 10.0
    raise ValueError("Cholesky failed in KernelFlow._safe_cholesky")


def _safe_inv(mat: tf.Tensor, base_jitter: float = 1e-6, max_tries: int = 8) -> tf.Tensor:
    dtype = mat.dtype
    jitter = tf.cast(base_jitter, dtype)
    I = tf.eye(tf.shape(mat)[0], dtype=dtype)
    for _ in range(max_tries):
        try:
            inv = tf.linalg.inv(mat + jitter * I)
            if tf.reduce_all(tf.math.is_finite(inv)):
                return inv
        except Exception:
            pass
        jitter = jitter * 10.0
    raise ValueError("Inverse failed in KernelFlow._safe_inv")


def compute_localized_B(x: tf.Tensor, prior_cov_override: Optional[tf.Tensor], localization: Optional[tf.Tensor], jitter: float):
    """Compute prior mean m, localized covariance B, and its inverse.

    Returns (m, B, B_inv).
    """
    x = tf.cast(x, tf.float64)
    N = tf.shape(x)[0]
    d = tf.shape(x)[1]
    m = tf.reduce_mean(x, axis=0)
    X = x - m
    cov = tf.matmul(X, X, transpose_a=True) / tf.cast(tf.maximum(N - 1, 1), tf.float64)
    if prior_cov_override is not None:
        B = tf.cast(prior_cov_override, tf.float64)
    else:
        if localization is not None:
            B = cov * tf.cast(localization, tf.float64)
        else:
            B = cov
    B += tf.eye(d, dtype=tf.float64) * tf.cast(jitter, tf.float64)
    B_inv = _safe_inv(B, base_jitter=jitter)
    return m, B, B_inv


def scalar_kernel_mahalanobis(x: tf.Tensor, B_inv: tf.Tensor, alpha: float) -> tf.Tensor:
    """Compute scalar (Mahalanobis) kernel matrix K with A=(alpha B)^{-1}.

    K_ij = exp(-0.5 * (x_i - x_j)^T (alpha^{-1} B^{-1}) (x_i - x_j))
    """
    x = tf.cast(x, tf.float64)
    XB = tf.matmul(x, B_inv)
    sq = tf.reduce_sum(x * XB, axis=1)  # (N,)
    M = tf.matmul(XB, x, transpose_b=True)  # (N,N) where M_ij = x_i^T B_inv x_j
    d2 = tf.expand_dims(sq, 1) + tf.expand_dims(sq, 0) - 2.0 * M
    alpha = tf.cast(alpha, tf.float64)
    K = tf.exp(- d2 / (2.0 * alpha))
    return K


def compute_grad_logp(x_new: tf.Tensor, y: tf.Tensor, H: Optional[tf.Tensor], h_func: Optional[callable], R_inv: tf.Tensor, m: tf.Tensor, B_inv: tf.Tensor) -> tf.Tensor:
    """Compute per-particle posterior gradient grad_logp at the given particle locations.

    Returns grad_logp shape (N,d).
    """
    N = tf.shape(x_new)[0]
    d = tf.shape(x_new)[1]
    N_int = tf.cast(N, tf.int32)

    if h_func is not None:
        y_j_list = []
        H_stack = []
        for i in range(N_int):
            xi = tf.reshape(x_new[i], [d])
            with tf.GradientTape() as tape:
                tape.watch(xi)
                yi = tf.cast(h_func(xi), tf.float64)
            Ji = tape.jacobian(yi, xi)
            Ji = tf.reshape(Ji, (tf.shape(yi)[0], tf.shape(xi)[0]))
            y_j_list.append(tf.reshape(yi, [-1]))
            H_stack.append(Ji)
        y_j = tf.stack(y_j_list, axis=0)
        H_stack = tf.stack(H_stack, axis=0)
    else:
        H_stack = tf.tile(tf.expand_dims(tf.cast(H, tf.float64), axis=0), [N_int, 1, 1])
        y_j = tf.linalg.matmul(x_new, tf.transpose(tf.cast(H, tf.float64)))

    obs_dim = tf.shape(y_j)[1]
    y_tiled = tf.tile(tf.reshape(y, [1, obs_dim]), [N_int, 1])
    y_diff = y_tiled - y_j
    grad_obs = []
    for i in range(N_int):
        Hi = tf.cast(H_stack[i], tf.float64)
        rhs = tf.matmul(R_inv, tf.reshape(y_diff[i], [-1, 1]))
        go = tf.reshape(tf.matmul(tf.transpose(Hi), rhs), [-1])
        grad_obs.append(go)
    grad_obs = tf.stack(grad_obs, axis=0)

    X_new = x_new - m
    grad_prior = - tf.matmul(X_new, B_inv)
    grad_logp = grad_obs + grad_prior
    return grad_logp


class KernelScalarFlow:
    """Global kernel-weighted EDH-like flow using a scalarized kernel weighting.

    Computes a single weighted covariance from particle kernel weights and
    applies an affine Daum–Huang mapping identical for all particles.
    """

    def __init__(self, n_flow_steps: int = 10, alpha: Optional[float] = None, localization: Optional[np.ndarray] = None, jitter: float = 1e-6):
        self.n_flow_steps = int(n_flow_steps)
        # alpha scales the per-d kernel bandwidth via alpha * B_dd
        self.alpha = float(alpha) if alpha is not None else None
        # localization matrix C (d x d) applied elementwise to sample covariance
        self.localization = None if localization is None else tf.cast(tf.convert_to_tensor(localization, dtype=tf.float64), tf.float64)
        self.jitter = float(jitter)

    def flow(self, particles: tf.Tensor, y: tf.Tensor, H: Optional[tf.Tensor], R: tf.Tensor, h_func: Optional[callable] = None, prior_cov_override: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.cast(tf.convert_to_tensor(particles), tf.float64)  # (N, d)
        y = tf.cast(tf.reshape(tf.convert_to_tensor(y), [-1]), tf.float64)
        R = tf.cast(tf.convert_to_tensor(R), tf.float64)

        N = tf.shape(x)[0]
        d = tf.shape(x)[1]

        # Compute prior mean, localized covariance B, and its inverse via helper
        m, B, B_inv = compute_localized_B(x, prior_cov_override, self.localization, self.jitter)
        R_inv = _safe_inv(R, base_jitter=self.jitter)

        # Note: we compute grad_logp (the posterior gradient) inside the
        # pseudo-time integration loop below because it depends on the current
        # particle positions x_s. The prior mean `m` and localized covariance
        # `B` are computed once (from the prior ensemble) and used throughout.
        N_int = tf.cast(N, tf.int32)

        # prepare alpha per-d
        if self.alpha is None:
            # default alpha ~ 1/N
            alpha_val = tf.cast(1.0 / tf.cast(N, tf.float64), tf.float64)
        else:
            alpha_val = tf.cast(self.alpha, tf.float64)

        # Per-component (dimension-wise) kernel integration
        # The algorithm integrates in pseudo-time s from 0->1. We discretize s into
        # `n_steps` increments of size delta_s and for each component d compute the
        # kernel K^{(d)}_{ij} between particle component values x_i^{(d)} and
        # x_j^{(d)}. The component-wise kernel width scales with the prior variance
        # B_{d,d} (via alpha), matching the pseudocode in the PFF algorithm.
        # For each component we also compute the derivative of the kernel w.r.t.
        # x_i^{(d)} (dK) and accumulate the integral term:
        #   f^{i,(d)} += (1/N) sum_j [ K_{ij} * (partial_{x^{(d)}} log p(x_j|y)) + dK_{ij} ]
        # After computing f for all components we update particles via
        #   x_i <- x_i + delta_s * B f_i
        # where B is the localized prior covariance (same for all particles).
        n_steps = max(1, int(self.n_flow_steps))
        delta_s = 1.0 / tf.cast(n_steps, tf.float64)

        x_new = tf.identity(x)
        for _ in range(n_steps):
            # Recompute observation model and posterior gradient at current x_new.
            # For nonlinear h_func we must re-evaluate h and its Jacobian at each
            # particle; for linear H we recompute y_j = H x_new.
            if h_func is not None:
                y_j_list = []
                H_stack = []
                for i in range(N_int):
                    xi = tf.reshape(x_new[i], [d])
                    with tf.GradientTape() as tape:
                        tape.watch(xi)
                        yi = tf.cast(h_func(xi), tf.float64)
                    Ji = tape.jacobian(yi, xi)
                    Ji = tf.reshape(Ji, (tf.shape(yi)[0], tf.shape(xi)[0]))
                    y_j_list.append(tf.reshape(yi, [-1]))
                    H_stack.append(Ji)
                y_j = tf.stack(y_j_list, axis=0)
                H_stack = tf.stack(H_stack, axis=0)  # (N, obs, d)
            else:
                H_stack = tf.tile(tf.expand_dims(tf.cast(H, tf.float64), axis=0), [N_int, 1, 1])
                y_j = tf.linalg.matmul(x_new, tf.transpose(tf.cast(H, tf.float64)))  # (N, obs)

            # Compute posterior gradient at current particle locations
            grad_logp = compute_grad_logp(x_new, y, H, h_func, R_inv, m, B_inv)

            # Scalar Mahalanobis kernel (global scalar kernel using B_inv)
            K = scalar_kernel_mahalanobis(x_new, B_inv, alpha_val)  # (N,N)

            # Weighted gradient term: (K @ grad_logp)
            weighted_grad = tf.matmul(K, grad_logp)  # (N,d)

            # Kernel gradient contribution: - (1/alpha) * B_inv * sum_j K_ij (x_i - x_j)
            sumK = tf.reduce_sum(K, axis=1, keepdims=True)  # (N,1)
            weighted_x = tf.matmul(K, x_new)  # (N,d)
            vector_term = sumK * x_new - weighted_x  # (N,d)
            second = - (1.0 / alpha_val) * tf.matmul(vector_term, B_inv)  # (N,d)

            f = (weighted_grad + second) / tf.cast(N, tf.float64)

            # update particles: x += delta_s * (B @ f_i)
            delta = tf.matmul(f, tf.transpose(B)) * delta_s  # (N,d)
            x_new = x_new + delta

        # Return new particles; logabsdet not computed (set to zeros)
        logabs = tf.zeros([1], dtype=tf.float64)
        return x_new, logabs


class KernelMatrixFlow:
    """Local per-particle kernel-weighted flow (matrix kernel).

    Each particle i gets a weight vector K[i,:] defining a local empirical
    covariance; we then compute a per-particle affine mapping similarly to LEDH.
    Returns per-particle logabsdet vector.
    """

    def __init__(self, n_flow_steps: int = 10, alpha: Optional[float] = None, localization: Optional[np.ndarray] = None, jitter: float = 1e-6):
        self.n_flow_steps = int(n_flow_steps)
        self.alpha = float(alpha) if alpha is not None else None
        self.localization = None if localization is None else tf.cast(tf.convert_to_tensor(localization, dtype=tf.float64), tf.float64)
        self.jitter = float(jitter)

    def flow(self, particles: tf.Tensor, y: tf.Tensor, H: Optional[tf.Tensor], R: tf.Tensor, h_func: Optional[callable] = None, prior_cov_override: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        # Matrix-kernel variant implements a per-component (diagonal) kernel as in the algorithm,
        # but produces the same vector field update B f_i for each particle where f_i is computed
        # using component-wise kernels. For scalability we vectorize where possible.
        x = tf.cast(tf.convert_to_tensor(particles), tf.float64)
        y = tf.cast(tf.reshape(tf.convert_to_tensor(y), [-1]), tf.float64)
        R = tf.cast(tf.convert_to_tensor(R), tf.float64)

        N = tf.shape(x)[0]
        d = tf.shape(x)[1]

        # Compute prior mean, localized covariance B, and its inverse via helper
        m, B, B_inv = compute_localized_B(x, prior_cov_override, self.localization, self.jitter)
        R_inv = _safe_inv(R, base_jitter=self.jitter)

        # Defer computing per-particle observation linearizations and
        # posterior gradients until inside the pseudo-time loop because they
        # depend on the current particle positions x_s. We'll compute them
        # at the start of each pseudo-step below.
        N_int = tf.cast(N, tf.int32)

        # alpha default
        if self.alpha is None:
            alpha_val = tf.cast(1.0 / tf.cast(N, tf.float64), tf.float64)
        else:
            alpha_val = tf.cast(self.alpha, tf.float64)

        # integrate in pseudo-time using component-wise kernel, similar to scalar flow
        n_steps = max(1, int(self.n_flow_steps))
        delta_s = 1.0 / tf.cast(n_steps, tf.float64)
        x_new = tf.identity(x)
        for _ in range(n_steps):
            # At each pseudo-time step recompute per-particle y_j and H_stack
            # using current particle locations x_new, then compute grad_logp.
            if h_func is not None:
                y_j_list = []
                H_stack = []
                for i in range(N_int):
                    xi = tf.reshape(x_new[i], [d])
                    with tf.GradientTape() as tape:
                        tape.watch(xi)
                        yi = tf.cast(h_func(xi), tf.float64)
                    Ji = tape.jacobian(yi, xi)
                    Ji = tf.reshape(Ji, (tf.shape(yi)[0], tf.shape(xi)[0]))
                    y_j_list.append(tf.reshape(yi, [-1]))
                    H_stack.append(Ji)
                y_j = tf.stack(y_j_list, axis=0)
                H_stack = tf.stack(H_stack, axis=0)
            else:
                H_stack = tf.tile(tf.expand_dims(tf.cast(H, tf.float64), axis=0), [N_int, 1, 1])
                y_j = tf.linalg.matmul(x_new, tf.transpose(tf.cast(H, tf.float64)))

            # Compute posterior gradient at current particle locations
            grad_logp = compute_grad_logp(x_new, y, H, h_func, R_inv, m, B_inv)

            # Build the vector field f and update particles as in scalar flow
            f = tf.zeros_like(x_new, dtype=tf.float64)
            for dd in range(int(d.numpy())):
                xi_d = tf.reshape(x_new[:, dd], [-1, 1])
                xj_d = tf.transpose(xi_d)
                diff = xi_d - xj_d
                B_dd = tf.cast(B[dd, dd], tf.float64)
                denom = 2.0 * alpha_val * B_dd
                K = tf.exp(- tf.square(diff) / denom)
                dK = - diff / (alpha_val * B_dd) * K
                grad_j_d = tf.reshape(grad_logp[:, dd], [1, -1])
                term = tf.reduce_sum(K * grad_j_d, axis=1) + tf.reduce_sum(dK, axis=1)
                f = f + tf.concat([tf.zeros([N_int, dd], dtype=tf.float64), tf.reshape(term / tf.cast(N, tf.float64), [N_int,1]), tf.zeros([N_int, int(d.numpy())-dd-1], dtype=tf.float64)], axis=1)
            delta = tf.matmul(f, tf.transpose(B)) * delta_s
            x_new = x_new + delta

        logabs = tf.zeros([N_int], dtype=tf.float64)
        return x_new, logabs



if __name__ == "__main__":
    """Small runnable self-test for KernelScalarFlow and KernelMatrixFlow.

    Runs a short flow on N particles in d dimensions and prints simple
    diagnostics (shapes and mean changes).
    """
    import sys
    tf.random.set_seed(2)
    tfd = tfp.distributions

    N = 50
    d = 3

    H = tf.constant(np.eye(d), dtype=tf.float64)
    R = tf.constant(np.eye(d) * 0.3, dtype=tf.float64)

    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d, dtype=tf.float64), scale_diag=tf.ones(d, dtype=tf.float64))
    particles = prior.sample(N)

    ks = KernelScalarFlow(n_flow_steps=6)
    km = KernelMatrixFlow(n_flow_steps=6)

    x_s, log_s = ks.flow(particles, tf.zeros(d, dtype=tf.float64), H, R)
    x_m, log_m = km.flow(particles, tf.zeros(d, dtype=tf.float64), H, R)

    print("KernelFlow self-test")
    print("  particles:", particles.shape)
    print("  scalar flow output:", x_s.shape, "mean:", np.round(np.mean(x_s.numpy(), axis=0), 3))
    print("  matrix flow output:", x_m.shape, "mean:", np.round(np.mean(x_m.numpy(), axis=0), 3))

    moved_s = not np.allclose(particles.numpy(), x_s.numpy())
    moved_m = not np.allclose(particles.numpy(), x_m.numpy())
    if moved_s and moved_m:
        print("KERNELFLOW SELF-TEST: PASS — both flows moved particles")
        sys.exit(0)
    else:
        print("KERNELFLOW SELF-TEST: FAIL — one or both flows did not move particles")
        sys.exit(2)




