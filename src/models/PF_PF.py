"""
Particle Filtering with Invertible Particle Flow (PF-PF) - minimal implementation.

This module implements a simple particle filter that uses an invertible
Daum-Huang affine particle flow as a proposal mapping. The mapping is
invertible and we correct importance weights using the mapping Jacobian
determinant (logabsdet) as described in PF-PF style algorithms.

Notes:
- The implementation here assumes a linear-Gaussian observation model
  y = H x + noise (noise ~ N(0, R)). For nonlinear observation models
  you would need to linearize or use a different flow.
- The Daum-Huang affine flow is implemented in `IEDHFlow`.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Robust linear-algebra helpers
def _safe_cholesky(mat: tf.Tensor, base_jitter: float = 1e-6, max_tries: int = 8) -> tf.Tensor:
    """Try Cholesky on `mat` with increasing diagonal jitter until success.

    Returns the lower-triangular Cholesky factor.
    Raises ValueError if a finite factor cannot be found.
    """
    dtype = mat.dtype
    jitter = tf.cast(base_jitter, dtype)
    I = tf.eye(tf.shape(mat)[0], dtype=dtype)
    debug = os.environ.get('PF_PF_DEBUG', '0') == '1'

    # optional diagnostic: report minimum eigenvalue before attempts
    try:
        eigs = tf.linalg.eigvalsh(mat)
        min_eig = float(tf.reduce_min(eigs).numpy())
    except Exception:
        min_eig = None
    if debug:
        print(f"[PF_PF DEBUG] _safe_cholesky: initial min_eig={min_eig}, base_jitter={base_jitter}")

    for attempt_i in range(max_tries):
        attempt = mat + jitter * I
        try:
            chol = tf.linalg.cholesky(attempt)
        except Exception as e:
            if debug:
                print(f"[PF_PF DEBUG] _safe_cholesky: cholesky failed on attempt {attempt_i} with jitter={float(jitter.numpy()):.3e}: {e}")
            chol = None

        if chol is not None and tf.reduce_all(tf.math.is_finite(chol)):
            if debug and attempt_i > 0:
                print(f"[PF_PF DEBUG] _safe_cholesky: succeeded with jitter={float(jitter.numpy()):.3e} on attempt {attempt_i}")
            return chol

        jitter = jitter * 10.0

    raise ValueError(f"Cholesky failed even after adding jitter up to {float(jitter.numpy()):.3e}")


def _safe_inv(mat: tf.Tensor, base_jitter: float = 1e-6, max_tries: int = 8) -> tf.Tensor:
    """Try to invert `mat`, adding diagonal jitter if needed until inverse is finite.

    Returns the matrix inverse.
    """
    dtype = mat.dtype
    jitter = tf.cast(base_jitter, dtype)
    I = tf.eye(tf.shape(mat)[0], dtype=dtype)
    debug = os.environ.get('PF_PF_DEBUG', '0') == '1'

    # optional diagnostic: report minimum eigenvalue before attempts
    try:
        eigs = tf.linalg.eigvalsh(mat)
        min_eig = float(tf.reduce_min(eigs).numpy())
    except Exception:
        min_eig = None
    if debug:
        print(f"[PF_PF DEBUG] _safe_inv: initial min_eig={min_eig}, base_jitter={base_jitter}")

    for attempt_i in range(max_tries):
        attempt = mat + jitter * I
        try:
            inv = tf.linalg.inv(attempt)
        except Exception as e:
            if debug:
                print(f"[PF_PF DEBUG] _safe_inv: inv failed on attempt {attempt_i} with jitter={float(jitter.numpy()):.3e}: {e}")
            inv = None

        if inv is not None and tf.reduce_all(tf.math.is_finite(inv)):
            if debug and attempt_i > 0:
                print(f"[PF_PF DEBUG] _safe_inv: succeeded with jitter={float(jitter.numpy()):.3e} on attempt {attempt_i}")
            return inv

        jitter = jitter * 10.0

    raise ValueError(f"Matrix inversion failed even after adding jitter up to {float(jitter.numpy()):.3e}")

# Inlined IEDHFlow implementation (previously in separate file)
class IEDHFlow:
    """Perform affine Daum-Huang flow and return mapping + log-determinant.

    Parameters
    ----------
    n_flow_steps : int
        Number of lambda integration steps (default 20).
    jitter : float
        Small diagonal jitter added to empirical covariances for stability.
    dtype : tf.DType
        TensorFlow dtype to use (default tf.float64).
    """

    def __init__(self, n_flow_steps: int = 20, jitter: float = 1e-6, dtype=tf.float64):
        self.n_flow_steps = int(n_flow_steps)
        self.jitter = float(jitter)
        self.dtype = dtype

    def _empirical_mean_and_cov(self, particles: tf.Tensor):
        N = tf.cast(tf.shape(particles)[0], self.dtype)
        mean = tf.reduce_mean(particles, axis=0)
        centered = particles - mean
        cov = tf.matmul(centered, centered, transpose_a=True) / N
        cov += tf.eye(tf.shape(particles)[1], dtype=self.dtype) * tf.cast(self.jitter, self.dtype)
        return mean, cov

    def flow(self, particles: tf.Tensor, y: tf.Tensor, H: tf.Tensor, R: tf.Tensor, prior_cov_override: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply Daum-Huang affine flow to `particles` and return (x_mapped, logabsdet).

        Parameters
        ----------
        particles : tf.Tensor, shape [N, d]
        y : tf.Tensor, shape [obs_dim] or [obs_dim,]
        H : tf.Tensor, shape [obs_dim, d]
        R : tf.Tensor, shape [obs_dim, obs_dim]

        Returns
        -------
        x_mapped : tf.Tensor, shape [N, d]
        logabsdet : tf.Tensor scalar (dtype=self.dtype)
            Log absolute determinant of the overall mapping (same for all particles)
        """
        x = tf.convert_to_tensor(particles)
        x = tf.cast(x, dtype=self.dtype)
        y = tf.convert_to_tensor(y)
        y = tf.cast(tf.reshape(y, [-1]), dtype=self.dtype)
        H = tf.convert_to_tensor(H)
        H = tf.cast(H, dtype=self.dtype)
        R = tf.convert_to_tensor(R)
        R = tf.cast(R, dtype=self.dtype)

        # Empirical prior mean and covariance (unless overridden)
        m0, P0_emp = self._empirical_mean_and_cov(x)
        if prior_cov_override is not None:
            P0 = tf.cast(prior_cov_override, dtype=self.dtype)
        else:
            P0 = P0_emp
        P0_inv = _safe_inv(P0, base_jitter=self.jitter)
        R_inv = _safe_inv(R, base_jitter=self.jitter)

        lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]

        m_prev = m0
        P_prev = P0

        total_logabsdet = tf.constant(0.0, dtype=self.dtype)

        for lam in lambdas:
            lam = tf.cast(lam, dtype=self.dtype)

            # Compute P(lambda) = inv(P0_inv + lam H^T R^{-1} H)
            Ht_Rinv = tf.matmul(tf.transpose(H), R_inv)
            S = P0_inv + lam * tf.matmul(Ht_Rinv, H)
            # compute P_lam robustly
            P_lam = _safe_inv(S, base_jitter=self.jitter)

            # Compute m(lambda) = P(lambda) (P0_inv m0 + lam H^T R^{-1} y)
            rhs = tf.matmul(P0_inv, tf.reshape(m0, [-1, 1])) + lam * tf.matmul(Ht_Rinv, tf.reshape(y, [-1, 1]))
            m_lam = tf.reshape(tf.matmul(P_lam, rhs), [-1])

            # Compute affine mapping matrix A
            # Compute Cholesky factors robustly (add jitter if needed)
            chol_prev = _safe_cholesky(P_prev, base_jitter=self.jitter)
            chol_lam = _safe_cholesky(P_lam, base_jitter=self.jitter)

            # A = chol_lam @ inv(chol_prev)
            I = tf.eye(tf.shape(chol_prev)[0], dtype=self.dtype)
            inv_chol_prev = tf.linalg.triangular_solve(chol_prev, I, lower=True)
            A = tf.matmul(chol_lam, inv_chol_prev)

            # Update x by affine mapping
            centered = x - m_prev
            x = tf.linalg.matmul(centered, tf.transpose(A)) + m_lam

            # logabsdet contribution: log|det(A)|
            logdet_chol_lam = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(chol_lam))))
            logdet_chol_prev = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(chol_prev))))
            logabsdet_A = logdet_chol_lam - logdet_chol_prev
            total_logabsdet += tf.cast(logabsdet_A, dtype=self.dtype)

            # advance
            m_prev = m_lam
            P_prev = P_lam

        return x, total_logabsdet


class ILEDHFlow:
    """Invertible Local Exact Daum-Huang (LEDH) flow.

    Performs a per-particle affine mapping by linearizing the observation
    function at each particle (or using a provided H matrix). Returns the
    mapped particles and a per-particle log-absolute-determinant vector.
    """

    def __init__(self, n_flow_steps: int = 20, jitter: float = 1e-6, dtype=tf.float64):
        self.n_flow_steps = int(n_flow_steps)
        self.jitter = float(jitter)
        self.dtype = dtype

    def flow(self, particles: tf.Tensor, y: tf.Tensor, H: Optional[tf.Tensor], R: tf.Tensor, h_func: Optional[callable] = None, prior_cov_override: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(particles)
        x = tf.cast(x, dtype=self.dtype)
        y = tf.convert_to_tensor(y)
        y = tf.cast(tf.reshape(y, [-1]), dtype=self.dtype)
        R = tf.convert_to_tensor(R)
        R = tf.cast(R, dtype=self.dtype)

        N = tf.shape(x)[0]
        d = tf.shape(x)[1]

        # empirical prior mean and covariance (unless overridden)
        m0 = tf.reduce_mean(x, axis=0)
        centered = x - m0
        Nf = tf.cast(tf.shape(x)[0], self.dtype)
        P0_emp = tf.matmul(centered, centered, transpose_a=True) / Nf
        P0_emp += tf.eye(d, dtype=self.dtype) * tf.cast(self.jitter, dtype=self.dtype)
        if prior_cov_override is not None:
            P0 = tf.cast(prior_cov_override, dtype=self.dtype)
        else:
            P0 = P0_emp
        P0_inv = _safe_inv(P0, base_jitter=self.jitter)
        R_inv = _safe_inv(R, base_jitter=self.jitter)

        # Build per-particle Jacobians (N, obs_dim, state_dim)
        if h_func is not None:
            H_list = []
            # convert N to python int for loop
            N_py = int(N.numpy()) if hasattr(N, 'numpy') else int(N)
            for i in range(N_py):
                x_i = tf.reshape(x[i], [d])
                with tf.GradientTape() as tape:
                    tape.watch(x_i)
                    y_i = tf.convert_to_tensor(h_func(x_i), dtype=self.dtype)
                J = tape.jacobian(y_i, x_i)
                J = tf.reshape(J, (tf.shape(y_i)[0], tf.shape(x_i)[0]))
                H_list.append(J)
            H_stack = tf.stack(H_list, axis=0)
        else:
            if H is None:
                raise ValueError("Either H or h_func must be provided to LEDH flow")
            H_stack = tf.tile(tf.expand_dims(tf.cast(H, dtype=self.dtype), axis=0), [N, 1, 1])

        # Per-particle states and accumulators
        N_py = int(N.numpy()) if hasattr(N, 'numpy') else int(N)
        x_list = [tf.reshape(x[i], [-1]) for i in range(N_py)]
        m_prev = [m0 for _ in range(N_py)]
        P_prev = [P0 for _ in range(N_py)]
        logabsdet_per_particle = [tf.constant(0.0, dtype=self.dtype) for _ in range(N_py)]

        lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]
        for lam in lambdas:
            lam = tf.cast(lam, dtype=self.dtype)
            for i in range(N_py):
                Hi = tf.cast(H_stack[i], dtype=self.dtype)
                Ht_Rinv = tf.matmul(tf.transpose(Hi), R_inv)
                S = P0_inv + lam * tf.matmul(Ht_Rinv, Hi)
                P_lam = _safe_inv(S, base_jitter=self.jitter)

                rhs = tf.matmul(P0_inv, tf.reshape(m0, [-1, 1])) + lam * tf.matmul(Ht_Rinv, tf.reshape(y, [-1, 1]))
                m_lam = tf.reshape(tf.matmul(P_lam, rhs), [-1])

                # Compute Cholesky factors robustly for per-particle covariances
                chol_prev = _safe_cholesky(P_prev[i], base_jitter=self.jitter)
                chol_lam = _safe_cholesky(P_lam, base_jitter=self.jitter)
                I = tf.eye(tf.shape(chol_prev)[0], dtype=self.dtype)
                inv_chol_prev = tf.linalg.triangular_solve(chol_prev, I, lower=True)
                A = tf.matmul(chol_lam, inv_chol_prev)

                centered_i = x_list[i] - m_prev[i]
                updated = tf.matmul(A, tf.reshape(centered_i, [-1, 1]))
                updated = tf.reshape(updated, [-1]) + m_lam
                x_list[i] = updated

                logdet_chol_lam = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(chol_lam))))
                logdet_chol_prev = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(chol_prev))))
                logabsdet_A = logdet_chol_lam - logdet_chol_prev
                logabsdet_per_particle[i] += tf.cast(logabsdet_A, dtype=self.dtype)

                m_prev[i] = m_lam
                P_prev[i] = P_lam

        x_prop = tf.stack(x_list, axis=0)
        logabsdet_vec = tf.stack(logabsdet_per_particle, axis=0)
        return x_prop, logabsdet_vec

tfd = tfp.distributions


class ParticleFilterWithInvertibleFlow:
    """Particle filter that uses invertible Daum-Huang flow as proposal.

    Parameters
    ----------
    num_particles : int
    state_dim : int
    dtype : tf.DType
    """

    def __init__(self, num_particles: int, state_dim: int, dtype=tf.float64, seed: Optional[int] = None, use_local_flow: bool = False, verbose_or_not: bool = False):
        self.num_particles = int(num_particles)
        self.state_dim = int(state_dim)
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

        self.particles = None  # tf.Tensor shape (N, d)
        self.log_weights = None  # tf.Tensor shape (N,)
        self.flow = IEDHFlow(n_flow_steps=20, dtype=self.dtype)
        # local-flow helper
        self.ledh_flow = ILEDHFlow(n_flow_steps=20, dtype=self.dtype)
        # If True, use a local (per-particle) affine EDH linearization (LEDH-like)
        # Requires providing `h_func` to `update` when using local flow for
        # nonlinear observations.
        self.use_local_flow = bool(use_local_flow)
        self.verbose = bool(verbose_or_not)

        # diagnostics
        self.ess_history = []

    def initialize(self, mean: np.ndarray = None, cov: np.ndarray = None):
        if mean is None or cov is None:
            # default small gaussian
            samples = self._rng.normal(scale=1.0, size=(self.num_particles, self.state_dim))
            self.particles = tf.convert_to_tensor(samples, dtype=self.dtype)
        else:
            prior = tfd.MultivariateNormalTriL(loc=tf.convert_to_tensor(mean, dtype=self.dtype),
                                               scale_tril=tf.linalg.cholesky(tf.convert_to_tensor(cov, dtype=self.dtype)))
            self.particles = prior.sample(self.num_particles)

        self.log_weights = tf.fill([self.num_particles], tf.math.log(1.0 / tf.cast(self.num_particles, self.dtype)))
        if self.verbose:
            print(f"PF-PF: initialize() created {self.num_particles} particles; mean shape={np.shape(mean) if mean is not None else None}")

    def predict(self, dynamics_fn, process_noise_cov: Optional[np.ndarray] = None):
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        # Apply dynamics elementwise
        if self.verbose:
            print("PF-PF: predict() applying dynamics to particles")
        self.particles = tf.map_fn(dynamics_fn, self.particles, fn_output_signature=self.dtype)

        if process_noise_cov is not None:
            chol = tf.linalg.cholesky(tf.convert_to_tensor(process_noise_cov, dtype=self.dtype))
            noise = tf.random.normal([self.num_particles, self.state_dim], dtype=self.dtype)
            self.particles = self.particles + tf.linalg.matmul(noise, chol)
            if self.verbose:
                print("PF-PF: predict() added process noise")

    def update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray, resample_threshold: float = 0.5, h_func: Optional[callable] = None, prior_cov_override: Optional[np.ndarray] = None):
        """Update particles using invertible flow and correct weights.

        Steps:
        1. Apply invertible Daum-Huang flow mapping to particles -> x_proposed, logabsdet
        2. Evaluate likelihood p(y | x_proposed)
        3. Update log-weights: log_w <- log_w + loglik + logabsdet
        4. Normalize weights and optionally resample
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialized")

        y_tf = tf.convert_to_tensor(y, dtype=self.dtype)
        # H may be None when a nonlinear h_func is provided (local flow).
        H_tf = None if H is None else tf.convert_to_tensor(H, dtype=self.dtype)
        R_tf = tf.convert_to_tensor(R, dtype=self.dtype)

        if self.verbose:
            mode = 'local' if self.use_local_flow else 'global'
            print(f"PF-PF: update() start; mode={mode}; obs_dim={tf.shape(y_tf)[0].numpy() if hasattr(tf.shape(y_tf)[0], 'numpy') else tf.shape(y_tf)[0]}")

        prior_cov_override_tf = None if prior_cov_override is None else tf.convert_to_tensor(prior_cov_override, dtype=self.dtype)

        if not self.use_local_flow:
            # Apply global invertible flow (same mapping for all particles)
            x_prop, logabsdet = self.flow.flow(self.particles, y_tf, H_tf, R_tf, prior_cov_override=prior_cov_override_tf)

            # Evaluate likelihood p(y | x_prop) under y ~ N(H x, R)
            preds = tf.transpose(tf.linalg.matmul(H_tf, tf.transpose(x_prop)))  # Shape: (N, obs_dim)

            # For batch evaluation, create MultivariateNormalTriL with batch locs
            R_chol = _safe_cholesky(R_tf, base_jitter=self.flow.jitter)
            mvn = tfd.MultivariateNormalTriL(loc=preds, scale_tril=R_chol)

            y_reshaped = tf.reshape(y_tf, [-1])  # flatten to (obs_dim,)
            obs_dim = tf.shape(y_reshaped)[0]
            y_batched = tf.broadcast_to(y_reshaped[None, :], [self.num_particles, obs_dim])

            loglikes = mvn.log_prob(y_batched)

            # Update log-weights: include mapping jacobian (same scalar for all particles)
            new_log_w = self.log_weights + loglikes + tf.cast(logabsdet, dtype=self.dtype)

            # Normalize
            log_norm = tf.reduce_logsumexp(new_log_w)
            self.log_weights = new_log_w - log_norm

            if self.verbose:
                print(f"PF-PF: update() global flow; log_norm={float(log_norm.numpy())}")

            # Assign proposed particles (we used a deterministic invertible mapping)
            self.particles = x_prop

        else:
            # Local per-particle invertible EDH (LEDH-like). We require either
            # `h_func` to compute per-particle Jacobians, or fall back to H.
            if h_func is None and H is None:
                raise ValueError("Local flow requires `h_func` or `H` to be provided")

            # Delegate to the ILEDHFlow implementation which supports an
            # optional `prior_cov_override` argument for perturbed predictive covariances.
            x_prop, logabsdet_vec = self.ledh_flow.flow(self.particles, y_tf, H_tf, R_tf, h_func=h_func, prior_cov_override=prior_cov_override_tf)

            # Evaluate likelihoods for each proposed particle
            if h_func is not None:
                preds = tf.map_fn(lambda xi: tf.reshape(tf.convert_to_tensor(h_func(tf.reshape(xi, [-1])), dtype=self.dtype), [-1]), x_prop, fn_output_signature=self.dtype)
            else:
                preds = tf.transpose(tf.linalg.matmul(H_tf, tf.transpose(x_prop)))  # Shape: (N, obs_dim)

            R_chol = _safe_cholesky(R_tf, base_jitter=self.ledh_flow.jitter)
            mvn = tfd.MultivariateNormalTriL(loc=preds, scale_tril=R_chol)
            y_reshaped = tf.reshape(y_tf, [-1])
            obs_dim = tf.shape(y_reshaped)[0]
            y_batched = tf.broadcast_to(y_reshaped[None, :], [self.num_particles, obs_dim])
            loglikes = mvn.log_prob(y_batched)

            # Update log-weights: per-particle logabsdet
            new_log_w = self.log_weights + loglikes + logabsdet_vec

            # Normalize
            log_norm = tf.reduce_logsumexp(new_log_w)
            self.log_weights = new_log_w - log_norm

            if self.verbose:
                try:
                    ln = float(log_norm.numpy())
                except Exception:
                    ln = None
                print(f"PF-PF: update() local flow; log_norm={ln}")

            # Assign proposed particles
            self.particles = x_prop

        # Effective sample size
        weights = tf.exp(self.log_weights)
        ess = 1.0 / tf.reduce_sum(weights ** 2)
        try:
            ess_val = float(ess.numpy())
        except Exception:
            ess_val = float(ess)
        self.ess_history.append(ess_val)
        if self.verbose:
            print(f"PF-PF: update() finished; ESS={ess_val:.2f}")

        # Resample if needed
        if ess < resample_threshold * float(self.num_particles):
            indices = self._systematic_resample(weights)
            self.particles = tf.gather(self.particles, indices)
            self.log_weights = tf.fill([self.num_particles], tf.math.log(1.0 / tf.cast(self.num_particles, self.dtype)))

        return float(log_norm.numpy())

    def _systematic_resample(self, weights: tf.Tensor) -> tf.Tensor:
        N = tf.shape(weights)[0]
        cs = tf.cumsum(weights)
        u0 = float(self._rng.random()) / float(N)
        positions = (tf.cast(tf.range(N), dtype=self.dtype) / tf.cast(N, dtype=self.dtype)) + tf.cast(u0, dtype=self.dtype)
        positions = tf.math.floormod(positions, 1.0)
        idx = tf.searchsorted(cs, positions, side='right')
        return tf.cast(idx, dtype=tf.int32)

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        w = tf.exp(self.log_weights)
        mean = tf.reduce_sum(tf.expand_dims(w, -1) * self.particles, axis=0)
        diff = self.particles - mean
        cov = tf.matmul(tf.transpose(diff), diff * tf.reshape(w, [-1, 1]))
        return mean.numpy(), cov.numpy()


if __name__ == "__main__":
    # small demo using linear Gaussian example similar to DaumHuangFlow tests
    tf.random.set_seed(1)
    N = 1000
    d = 1
    pf = ParticleFilterWithInvertibleFlow(num_particles=N, state_dim=d, seed=2, use_local_flow=True)
    pf.initialize(mean=np.zeros((d,)), cov=np.eye(d))

    # Observation model
    H = np.array([[1.0]])
    R = np.array([[0.5]])
    true_x = np.array([2.0])
    y = true_x + np.random.normal(scale=np.sqrt(R[0,0]))

    logz = pf.update(y=y, H=H, R=R)
    mean, cov = pf.estimate()
    print("log evidence approx:", logz)
    print("mean estimate:", mean)
