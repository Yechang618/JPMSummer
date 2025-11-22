"""
Particle-Flow Particle Filters (PFPF)

Implements two particle-flow-based particle filters using the flows in `EDH.py`:
- LEDH_ParticleFlowPF : Local Exact Daum-Huang flow per particle (LEDH) used inside a PF
- EDH_ParticleFlowPF  : Global EDH flow used inside a PF

The implementation follows the algorithms described in the user's prompt and re-uses
helper functions from `models.EDH` (calculate_A_b, lin_F, mP_predict, etc.).

Notes / simplifications:
- Likelihoods and transition densities are Gaussian (uses `self.R` and `self.Q`).
- EKF prediction/update steps are approximated using linearizations via `lin_F`/`lin_H`.
- The implementation emphasizes clarity over maximum vectorization; per-particle loops
  are used for the LEDH variant.

"""

from typing import Optional, Callable
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.EDH import calculate_A_b, lin_F, lin_H, mP_predict, mP_update, safe_cholesky

tfd = tfp.distributions


class BaseParticleFlowPF:
    def __init__(self,
                 num_particles: int,
                 f: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 h: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
                 state_dim: int = None,
                 observation_dim: Optional[int] = None,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 dtype=tf.float64):
        self.num_particles = int(num_particles)
        self.state_dim = int(state_dim)
        if observation_dim is None:
            observation_dim = self.state_dim
        self.observation_dim = int(observation_dim)
        self.f = f if f is not None else (lambda x: x)
        self.h = h if h is not None else (lambda x: x)
        self.dtype = dtype

        # Covariances
        self.Q = tf.cast(Q if Q is not None else np.zeros((self.state_dim, self.state_dim)), dtype=self.dtype)
        self.R = tf.cast(R if R is not None else np.zeros((self.observation_dim, self.observation_dim)), dtype=self.dtype)

        # Particles and weights
        self.particles: Optional[tf.Tensor] = None  # shape [N, d]
        self.weights: Optional[tf.Tensor] = None    # shape [N]

        # Per-particle (or global) local covariances for EKF-like prediction
        self.P_particles = None  # shape [N, d, d] (for LEDH), or single [d,d] for EDH

        # Flow integration settings
        self.n_flow_steps = 10
        self.jitter = 1e-6

    def initialize(self, particles: Optional[tf.Tensor] = None, initial_dist: Optional[tfd.Distribution] = None):
        if particles is None and initial_dist is None:
            raise ValueError("Either particles or initial_dist must be provided")
        if particles is None:
            p = initial_dist.sample(self.num_particles)
            self.particles = tf.cast(p, dtype=self.dtype)
        else:
            self.particles = tf.cast(tf.convert_to_tensor(particles), dtype=self.dtype)

        # uniform weights
        self.weights = tf.ones([self.num_particles], dtype=self.dtype) / tf.cast(self.num_particles, self.dtype)

        # initialize per-particle covariances to empirical covariance
        mean, cov = self._empirical_mean_and_cov(self.weights, self.particles)
        self.x_hat = mean
        self.P = cov
        # default per-particle covariances: copy global cov
        self.P_particles = tf.tile(tf.expand_dims(cov, 0), [self.num_particles, 1, 1])

    def _empirical_mean_and_cov(self, weights: Optional[tf.Tensor], particles: tf.Tensor):
        """Compute empirical (possibly weighted) mean and covariance.

        Args:
            weights: None or 1-D tensor of shape [N]. If provided, a weighted mean
                     is computed. If None, an unweighted mean is used.
            particles: tensor shape [N, d]

        Returns:
            mean: 1-D tensor shape [d]
            cov: 2-D tensor shape [d, d]
        """
        particles = tf.cast(particles, dtype=self.dtype)
        if weights is None:
            mean = tf.reduce_mean(particles, axis=0)
        else:
            w = tf.cast(tf.reshape(weights, [-1, 1]), dtype=self.dtype)
            w_sum = tf.reduce_sum(w)
            # avoid division by zero
            w_sum = tf.where(w_sum == 0, tf.constant(1.0, dtype=self.dtype), w_sum)
            mean = tf.reduce_sum(w * particles, axis=0) / w_sum

        centered = particles - mean
        N = tf.cast(tf.shape(particles)[0], self.dtype)
        cov = tf.matmul(centered, centered, transpose_a=True) / N
        cov = cov + tf.eye(self.state_dim, dtype=self.dtype) * tf.cast(self.jitter, self.dtype)
        return mean, cov

    def effective_N(self):
        w = tf.cast(self.weights, self.dtype)
        return 1.0 / tf.reduce_sum(w ** 2)

    def resample(self):
        # systematic resampling
        w = self.weights.numpy().astype(float)
        N = self.num_particles
        positions = (np.arange(N) + np.random.random()) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(w)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        # reorder particles and reset weights
        p_np = self.particles.numpy()
        p_np = p_np[indexes]
        self.particles = tf.cast(p_np, dtype=self.dtype)
        # reorder P_particles if per-particle
        if len(self.P_particles.shape) == 3:
            P_np = self.P_particles.numpy()[indexes]
            self.P_particles = tf.cast(P_np, dtype=self.dtype)
        self.weights = tf.ones([N], dtype=self.dtype) / tf.cast(N, self.dtype)


class EDH_ParticleFlowPF(BaseParticleFlowPF):
    """Particle Flow Particle Filter using global EDH flow."""

    def step(self, z: tf.Tensor, resample_threshold: Optional[float] = None):
        z = tf.cast(tf.reshape(z, [-1]), dtype=self.dtype)
        N = self.num_particles
        d = self.state_dim

        # EKF-like global prediction from weighted mean
        # mean_prev, cov_prev = self._empirical_mean_and_cov(self.particles)
        x_hat = self.x_hat
        P = self.P
        F = lin_F(self.f, tf.reshape(x_hat, [d]), dtype=self.dtype)
        m_pred = tf.reshape(self.f(x_hat), [-1])
        P_pred = F @ P @ tf.transpose(F) + self.Q
        
        # propagate particles through dynamics (sample noise)
        Q_tril = tf.linalg.cholesky(self.Q + tf.eye(d, dtype=self.dtype) * 1e-12)
        dyn_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(d, dtype=self.dtype), scale_tril=Q_tril)
        eta0 = []
        for i in range(N):
            xi = tf.reshape(self.particles[i], [d])
            noise = dyn_dist.sample()
            eta0_i = tf.reshape(self.f(xi), [-1]) + noise
            eta0.append(eta0_i)
        eta0 = tf.stack(eta0)
        eta1 = tf.identity(eta0)

        # bar_eta is linearization point: g_k(x_hat,0)
        bar_eta = tf.reshape(self.f(x_hat), [-1])

        lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]
        log_theta = np.zeros(N, dtype=float)

        # for each lambda, compute A,b globally and apply to all particles
        for lam in lambdas:
            lam = tf.cast(lam, dtype=self.dtype)
            A, b_vec, H_curr = calculate_A_b(self.h, bar_eta, z, lam, P_pred, self.R, self.jitter, d, dtype=self.dtype)
            step = tf.cast(1.0 / float(self.n_flow_steps), dtype=self.dtype)
            # (clean) shapes are expected: A [d,d], bar_eta [d], b_vec [d]
            # update bar_eta as column vector to avoid broadcasting to 2-D
            bar_col = tf.reshape(bar_eta, [d, 1])
            bar_update_col = step * (tf.matmul(A, bar_col) + tf.reshape(b_vec, [d, 1]))
            bar_eta = tf.reshape(bar_col + bar_update_col, [-1])
            # update each particle
            for i in range(N):
                # ensure per-particle vector has correct length d
                eta_vec = tf.reshape(eta1[i], [-1])
                eta_update = step * (tf.matmul(A, tf.reshape(eta_vec, [-1, 1])) + tf.reshape(b_vec, [-1, 1]))
                eta_new = tf.reshape(eta_vec + tf.reshape(eta_update, [-1]), [d])
                eta1 = tf.tensor_scatter_nd_update(eta1, [[i]], [eta_new])
            # update log_theta as product of det(I + step*A) for all particles (same A)
            I = tf.eye(d, dtype=self.dtype)
            mat = I + step * A
            sign, logabsdet = tf.linalg.slogdet(mat)
            log_theta += float(logabsdet.numpy())

        # set particles to final eta1
        self.particles = tf.cast(eta1, dtype=self.dtype)

        # weight update: w 
        # Here ratio of transition densities cancels if we used same proposal; we'll approximate with obs likelihood
        obs_tril = tf.linalg.cholesky(self.R + tf.eye(self.observation_dim, dtype=self.dtype) * 1e-12)
        obs_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(self.observation_dim, dtype=self.dtype), scale_tril=obs_tril)
        log_w_old = tf.math.log(self.weights + tf.cast(1e-20, self.dtype)).numpy()
        log_w_new = np.zeros(N, dtype=float)
        for i in range(N):
            hz = tf.reshape(self.h(tf.reshape(self.particles[i], [d])), [-1])
            obs_logp = obs_dist.log_prob(tf.reshape(z - hz, [-1]))
            # include theta (same for all particles in EDH) as additive factor
            log_w_new[i] = float(log_w_old[i]) + float(obs_logp.numpy()) + float(log_theta[i])

        # normalize
        w = np.exp(log_w_new - np.max(log_w_new))
        w = w / np.sum(w)
        self.weights = tf.cast(w, dtype=self.dtype)

        # optional normalized EKF update for global mean/cov
        # Use linearized measurement at bar_eta
        H_final = H_curr
        P_upd, m_upd = mP_update(P_pred, m_pred, H_final, self.R, z, self.jitter, dtype=self.dtype)

        # store
        self.global_mean = m_upd
        self.global_cov = P_upd

        # estimate weighted mean and covariance
        w_tf = tf.reshape(self.weights, [-1, 1])
        x_hat = tf.reduce_sum(self.particles * w_tf, axis=0)
        diffs = self.particles - x_hat
        cov = tf.matmul(tf.transpose(diffs * w_tf), diffs)

        if resample_threshold is not None:
            if float(self.effective_N()) < float(resample_threshold):
                self.resample()

        return x_hat, cov


class LEDH_ParticleFlowPF(BaseParticleFlowPF):
    """Particle Flow Particle Filter using local (per-particle) LEDH flow."""

    def step(self, z: tf.Tensor, resample_threshold: Optional[float] = None):
        """Perform one PF time-step with particle flow update using LEDH.

        z : observation at current time (1-D tensor of observation_dim)
        Returns: (x_hat, P_hat)
        """
        z = tf.cast(tf.reshape(z, [-1]), dtype=self.dtype)
        N = self.num_particles
        d = self.state_dim

        # Containers for quantities
        eta0 = []      # propagated samples from dynamics (with noise)
        eta1 = []      # post-flow particles
        log_theta = np.zeros(N, dtype=float)

        # Per-particle EKF-like prediction to estimate P^i and m_{k|k-1}^i
        m_preds = []
        P_preds = []
        for i in range(N):
            xi = tf.reshape(self.particles[i], [d])
            # linearize dynamics at xi
            F_i = lin_F(self.f, tf.reshape(xi, [d]), dtype=self.dtype)
            # m_pred = f(xi)
            m_pred = tf.reshape(self.f(xi), [-1])
            # P_i prior
            P_i = tf.cast(self.P_particles[i], dtype=self.dtype)
            P_pred = F_i @ P_i @ tf.transpose(F_i) + self.Q
            m_preds.append(m_pred)
            P_preds.append(P_pred)

        # Sample eta0 (propagate particles through dynamics including process noise)
        # Use TFP multivariate normals per particle (same Q)
        Q_tril = tf.linalg.cholesky(self.Q + tf.eye(d, dtype=self.dtype) * 1e-12)
        dyn_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(d, dtype=self.dtype), scale_tril=Q_tril)

        for i in range(N):
            xi = tf.reshape(self.particles[i], [d])
            # sample noise and form eta0
            noise = dyn_dist.sample()
            eta0_i = tf.reshape(self.f(xi), [-1]) + noise
            eta0.append(eta0_i)

        # Initialize eta1 as eta0; and bar_eta (linearization point) at zero-noise dynamics
        eta1 = [tf.identity(e) for e in eta0]
        bar_eta = [tf.reshape(self.f(tf.reshape(self.particles[i], [d])), [d]) for i in range(N)]

        # flow step sizes
        lambdas = tf.linspace(0.0, 1.0, self.n_flow_steps + 1)[1:]

        # For each lambda step, recompute local linearization at bar_eta^i and update
        for lam in lambdas:
            lam = tf.cast(lam, dtype=self.dtype)
            # eps = lam - 0.0 if False else (1.0 / (self.n_flow_steps))  # small step size (approximate)
            # The code uses lam itself when building A/b (as in EDH.calculate_A_b)
            for i in range(N):
                m0_i = bar_eta[i]
                P_pred_i = P_preds[i]
                # compute A,b at this lam for this particle
                A_i, b_i, H_curr = calculate_A_b(self.h, m0_i, z, lam, P_pred_i, self.R, self.jitter, d, dtype=self.dtype)
                # update bar_eta and eta1 using Euler-like step with step size 1/n_flow_steps
                step = tf.cast(1.0 / float(self.n_flow_steps), dtype=self.dtype)
                # Coerce vectors to length d in case of unexpected shapes
                # bar_vec = tf.reshape(bar_eta[i], [-1])[:d]
                # eta_vec = tf.reshape(eta1[i], [-1])[:d]
                bar_vec = tf.reshape(bar_eta[i], [-1])
                eta_vec = tf.reshape(eta1[i], [-1])
                bar_update = step * (tf.matmul(A_i, tf.reshape(bar_vec, [-1, 1])) + tf.reshape(b_i, [-1, 1]))
                eta_update = step * (tf.matmul(A_i, tf.reshape(eta_vec, [-1, 1])) + tf.reshape(b_i, [-1, 1]))
                bar_eta[i] = tf.reshape(bar_vec + tf.reshape(bar_update, [-1]), [d])
                eta1[i] = tf.reshape(eta_vec + tf.reshape(eta_update, [-1]), [d])
                # update log_theta by adding log|det(I + step * A)|
                I = tf.eye(d, dtype=self.dtype)
                mat = I + step * A_i
                sign, logabsdet = tf.linalg.slogdet(mat)
                # convert to numpy float and accumulate
                log_theta[i] += float(logabsdet.numpy())

        # Convert lists to tensors
        eta1 = tf.stack(eta1)
        eta0 = tf.stack(eta0)
        bar_eta = tf.stack(bar_eta)

        # Update particles: set x_k^i = eta1^i
        self.particles = tf.cast(eta1, dtype=self.dtype)

        # Weight update: w_k^i = w_{k-1}^i * [ p(x_k^i|x_{k-1}^i) * p(z|x_k^i) * theta^i ] / p(eta0^i|x_{k-1}^i)
        # Under Gaussian dynamics, transition density is N(x; f(x_{k-1}), Q)
        dyn_mean = []
        for i in range(N):
            dyn_mean.append(tf.reshape(self.f(tf.reshape(self.particles[i], [d])), [-1]))
        # But we need means relative to previous x_{k-1}
        trans_log_probs_final = []
        trans_log_probs_eta0 = []
        obs_log_probs = []
        trans_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(d, dtype=self.dtype), scale_tril=Q_tril)
        obs_tril = tf.linalg.cholesky(self.R + tf.eye(self.observation_dim, dtype=self.dtype) * 1e-12)
        obs_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(self.observation_dim, dtype=self.dtype), scale_tril=obs_tril)

        for i in range(N):
            x_prev = tf.reshape(self.particles[i], [d])  # (Note: approximation; ideal is previous particle before flow)
            mean_prev = tf.reshape(self.f(x_prev), [-1])
            # transition prob of final x under N(f(x_prev), Q)
            trans_log_prob_final = trans_dist.log_prob(tf.reshape(self.particles[i] - mean_prev, [-1]))
            trans_log_prob_eta0 = trans_dist.log_prob(tf.reshape(eta0[i] - mean_prev, [-1]))
            # observation likelihood p(z | x_k^i)
            hz = tf.reshape(self.h(tf.reshape(self.particles[i], [d])), [-1])
            obs_logp = obs_dist.log_prob(tf.reshape(z - hz, [-1]))
            trans_log_probs_final.append(float(trans_log_prob_final.numpy()))
            trans_log_probs_eta0.append(float(trans_log_prob_eta0.numpy()))
            obs_log_probs.append(float(obs_logp.numpy()))

        # compute new log weights
        log_w_old = tf.math.log(self.weights + tf.cast(1e-20, self.dtype)).numpy()
        log_w_new = np.zeros(N, dtype=float)
        for i in range(N):
            log_w_new[i] = (log_w_old[i]
                            + trans_log_probs_final[i]
                            + obs_log_probs[i]
                            + log_theta[i]
                            - trans_log_probs_eta0[i])
        # normalize
        w = np.exp(log_w_new - np.max(log_w_new))
        w = w / np.sum(w)
        self.weights = tf.cast(w, dtype=self.dtype)

        # recompute per-particle covariances via EKF update (approx)
        for i in range(N):
            # use linearized update at particle
            xi = tf.reshape(self.particles[i], [d])
            H_i = lin_H(self.h, tf.reshape(xi, [d]), dtype=self.dtype)
            P_pred = P_preds[i]
            P_upd, m_upd = mP_update(P_pred, m_preds[i], H_i, self.R, z, self.jitter, dtype=self.dtype)
            # store P_particles
            self.P_particles = tf.tensor_scatter_nd_update(self.P_particles, [[i]], [P_upd])

        # Estimate mean and covariance (weighted)
        w_tf = tf.reshape(self.weights, [-1, 1])
        x_hat = tf.reduce_sum(self.particles * w_tf, axis=0)
        # weighted covariance
        diffs = self.particles - x_hat
        cov = tf.matmul(tf.transpose(diffs * w_tf), diffs)

        # optional resample
        if resample_threshold is not None:
            if float(self.effective_N()) < float(resample_threshold):
                self.resample()

        return x_hat, cov




if __name__ == "__main__":
        """Small self-test for particle-flow particle filters.

        - Runs T=5 time steps where the true state propagates by additive process
            noise with covariance Q: x_{t+1} = x_t + process_noise (N(0,Q)).
        - Observations are linear: y = H x + observation_noise.
        - Initializes LEDH and EDH particle-flow PFs with the same particle cloud
            and runs them on the same observations, then compares final estimates.
        """
        import sys

        tf.random.set_seed(2)

        N = 50
        d = 3

        H = tf.constant(np.eye(d), dtype=tf.float64)
        R = tf.constant(np.eye(d) * 0.3, dtype=tf.float64)
        # small process noise
        Q = tf.constant(np.eye(d) * 0.1, dtype=tf.float64)

        def make_h(H):
            def h(x):
                x_flat = tf.reshape(tf.convert_to_tensor(x), [-1])
                d_local = tf.shape(H)[1]
                x_vec = x_flat[:d_local]
                return tf.reshape(tf.matmul(H, tf.reshape(x_vec, [-1, 1])), [-1])
            return h

        h_fn = make_h(H)

        prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d, dtype=tf.float64), scale_diag=tf.ones(d, dtype=tf.float64))
        particles = prior.sample(N)

        pf_ledh = LEDH_ParticleFlowPF(num_particles=N, f=(lambda x: x), h=h_fn, state_dim=d, observation_dim=d, R=R, Q=Q, dtype=tf.float64)
        pf_edh = EDH_ParticleFlowPF(num_particles=N, f=(lambda x: x), h=h_fn, state_dim=d, observation_dim=d, R=R, Q=Q, dtype=tf.float64)

        pf_ledh.initialize(particles=particles)
        pf_edh.initialize(particles=particles)

        pf_ledh.n_flow_steps = 6
        pf_edh.n_flow_steps = 6

        T = 5
        true_x = tf.zeros(d, dtype=tf.float64)

        print("Running PFPF self-test T=", T)
        for t in range(T):
            # process noise: draw 1xd standard normal, multiply by Cholesky to get 1xd, then reshape
            proc_noise = tf.reshape(tf.matmul(tf.random.normal([1, d], dtype=tf.float64), tf.linalg.cholesky(Q)), [-1])
            true_x = true_x + proc_noise
            # observation with noise
            obs_noise = tf.reshape(tf.matmul(tf.random.normal([1, d], dtype=tf.float64), tf.linalg.cholesky(R)), [-1])
            y_t = tf.reshape(tf.matmul(H, tf.reshape(true_x, [-1, 1])), [-1]) + obs_noise

            ledh_hat, ledh_P = pf_ledh.step(y_t)
            edh_hat, edh_P = pf_edh.step(y_t)

            print(f"t={t+1}  y={np.round(y_t.numpy(),3)}, x = {np.round(true_x.numpy(),3)}")
            print("  LEDH mean:", np.round(ledh_hat.numpy(),3), "EDH mean:", np.round(edh_hat.numpy(),3))

        ledh_mean = ledh_hat.numpy()
        edh_mean = edh_hat.numpy()
        ledh_cov = ledh_P.numpy()
        edh_cov = edh_P.numpy()

        ok_mean = np.allclose(ledh_mean, edh_mean, atol=1e-1, rtol=1e-6)
        ok_cov = np.allclose(ledh_cov, edh_cov, atol=1e-1, rtol=1e-6)

        if ok_mean and ok_cov:
                print("PFPF SELF-TEST: PASS — LEDH and EDH particle-flow PFs produce similar final estimates")
                sys.exit(0)
        else:
                print("PFPF SELF-TEST: FAIL — outputs differ")
                sys.exit(2)
