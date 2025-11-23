"""Particle Filter implementation using TensorFlow and TensorFlow Probability.

This module provides a vectorized particle filter that uses TensorFlow ops and
TensorFlow Probability for probability calculations. It supports systematic
resampling and a simple API:

- Predict: propagate particles through transition function with process noise
- Update: weight particles by likelihood of observation, normalize, resample
- Estimate: compute weighted mean and covariance

The implementation focuses on clarity and numerical stability (explicit dtype
handling). A small demo is included in the `if __name__ == '__main__'` block.

API
---
class ParticleFilter:
    __init__(transition_fn, observation_fn, Q, R, num_particles=1000,
             initial_mean=None, initial_cov=None, dtype=tf.float64, seed=None)
    predict()
    update(observation)
    step(observation) -> (mean, cov, loglik)
    estimate() -> (mean, cov)

"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ParticleFilter:
    """A simple particle filter using TensorFlow.

    Parameters
    ----------
    transition_fn : Callable[[tf.Tensor], tf.Tensor]
        Function mapping a particle state -> next state (deterministic part).
        Should accept a tensor of shape (state_dim,) and return same shape.
    observation_fn : Callable[[tf.Tensor], tf.Tensor]
        Observation function h(x) -> y (deterministic part). Accepts shape
        (state_dim,) and returns (obs_dim,).
    Q : array-like or tf.Tensor
        Process noise covariance (state_dim x state_dim)
    R : array-like or tf.Tensor
        Observation noise covariance (obs_dim x obs_dim)
    num_particles : int
        Number of particles to use
    initial_mean : array-like or tf.Tensor, optional
        If provided together with `initial_cov`, particles are drawn from that
        Gaussian. Otherwise user must provide reasonable prior via seeds.
    initial_cov : array-like or tf.Tensor, optional
        Covariance to sample initial particles from.
    dtype : tf.DType
        Floating dtype to use (default tf.float64)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        transition_fn: Callable[[tf.Tensor], tf.Tensor],
        observation_fn: Callable[[tf.Tensor], tf.Tensor],
        Q,
        R,
        num_particles: int = 1000,
        initial_mean: Optional[tf.Tensor] = None,
        initial_cov: Optional[tf.Tensor] = None,
        dtype: tf.DType = tf.float64,
        seed: Optional[int] = None,
        verbose_or_not: bool = False,
    ) -> None:
        self.f = transition_fn
        self.h = observation_fn
        self.dtype = dtype
        self.num_particles = int(num_particles)
        self._rng = np.random.default_rng(seed)
        self.verbose = bool(verbose_or_not)

        # Convert covariances
        self.Q = tf.convert_to_tensor(Q, dtype=self.dtype)
        self.R = tf.convert_to_tensor(R, dtype=self.dtype)

        # Dimensions
        self.state_dim = int(self.Q.shape[0])
        self.obs_dim = int(self.R.shape[0])

        # Initialize particles and weights
        if initial_mean is not None and initial_cov is not None:
            mean = tf.convert_to_tensor(initial_mean, dtype=self.dtype)
            cov = tf.convert_to_tensor(initial_cov, dtype=self.dtype)
            mvn = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(cov))
            samples = mvn.sample(self.num_particles, seed=seed)
            # samples shape (num_particles, state_dim)
            self.particles = tf.cast(samples, dtype=self.dtype)
        else:
            # Default: small Gaussian around zero
            self.particles = tf.convert_to_tensor(
                self._rng.normal(scale=1.0, size=(self.num_particles, self.state_dim)),
                dtype=self.dtype,
            )

        # Initialize uniform weights (use self.dtype explicitly)
        init_log_w = tf.cast(tf.math.log(1.0 / tf.cast(self.num_particles, self.dtype)), dtype=self.dtype)
        self.log_weights = tf.fill([self.num_particles], init_log_w)

        # Precreate distributions for noise sampling
        self._Q_tril = tf.linalg.cholesky(self.Q)
        self._R_tril = tf.linalg.cholesky(self.R)

        # Diagnostic history for degeneracy analysis
        # last_ess: most recent effective sample size (tf.Tensor scalar)
        # last_weights: most recent normalized weights (tf.Tensor shape (N,))
        # ess_history: python list of ESS values (floats)
        # weights_history: python list of numpy arrays of weights at each step
        self.last_ess = None
        self.last_weights = None
        self.ess_history = []
        self.weights_history = []
        # Evaluation histories
        # Per-step RMSE of particle mean vs true state (if provided)
        self.rmse_history = []
        # Per-step observation log-likelihoods (float list)
        self.ll_history = []

    def __getattr__(self, name: str):
        """Lazily create and return diagnostic attributes if missing.

        This helps when older in-memory instances (or instances created
        before these diagnostics were added) are accessed. We create
        per-instance defaults so attributes are always available.
        """
        if name == "last_ess":
            self.last_ess = None
            return self.last_ess
        if name == "last_weights":
            self.last_weights = None
            return self.last_weights
        if name == "ess_history":
            self.ess_history = []
            return self.ess_history
        if name == "weights_history":
            self.weights_history = []
            return self.weights_history
        raise AttributeError(f"{self.__class__.__name__!s} object has no attribute {name!r}")

    def _systematic_resample(self, weights: tf.Tensor) -> tf.Tensor:
        """Systematic resampling.

        weights: normalized weights (sum to 1), shape (num_particles,)
        returns indices of resampled particles shape (num_particles,)
        """
        # Ensure float64 for stability
        weights = tf.cast(weights, dtype=self.dtype)
        N = tf.shape(weights)[0]
        # cumulative sum
        cs = tf.cumsum(weights)
        # positions: (u + i)/N
        u0 = tf.cast(self._rng.random(), dtype=self.dtype) / tf.cast(N, dtype=self.dtype)
        offsets = (tf.cast(tf.range(N), dtype=self.dtype) / tf.cast(N, dtype=self.dtype)) + u0
        # wrap offsets in (0,1)
        offsets = tf.math.floormod(offsets, 1.0)
        # searchsorted into cs
        indices = tf.searchsorted(cs, offsets, side="right")
        indices = tf.cast(indices, dtype=tf.int32)
        return indices

    def predict(self) -> None:
        """Propagate particles through transition and add process noise."""
        if self.verbose:
            print(f"ParticleFilter: predict() propagating {self.num_particles} particles")
        # Apply transition function elementwise. We vectorize via map_fn.
        # Input particles has shape (N, state_dim). map_fn will pass each row.
        particles_pred = tf.map_fn(self.f, self.particles, fn_output_signature=tf.float64)

        # Add process noise samples
        q_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(self.state_dim, dtype=self.dtype), scale_tril=self._Q_tril)
        # noise size (N, state_dim)
        noise = q_dist.sample(self.num_particles)
        noise = tf.cast(noise, dtype=self.dtype)

        self.particles = particles_pred + noise

    def update(self, observation) -> tf.Tensor:
        """Update particle weights with the given observation.

        Returns
        -------
        loglik : tf.Tensor scalar
            Log-likelihood of the observation under the particle predictive
            distribution (approx).
        """
        y = tf.convert_to_tensor(observation, dtype=self.dtype)
        # Predict observation for each particle
        preds = tf.map_fn(self.h, self.particles, fn_output_signature=tf.float64)

        # Likelihood p(y | x) for each particle under Gaussian R
        # Use MultivariateNormalTriL for numerical stability
        mvn = tfd.MultivariateNormalTriL(loc=preds, scale_tril=self._R_tril)
        # mvn.log_prob accepts batch loc shape (N, obs_dim) and returns (N,)
        log_likes = mvn.log_prob(y)

        if self.verbose:
            print("ParticleFilter: update() computing log-likelihoods and updating weights")

        # Update log-weights: prior log_weights + log_likes
        new_log_w = self.log_weights + log_likes
        # Normalize via log-sum-exp
        log_norm_const = tf.reduce_logsumexp(new_log_w)
        normalized_log_w = new_log_w - log_norm_const

        self.log_weights = normalized_log_w

        # Effective Sample Size
        weights = tf.exp(self.log_weights)
        ess = 1.0 / tf.reduce_sum(weights ** 2)

        # Save diagnostics before possible resampling
        try:
            self.last_ess = ess
            self.last_weights = weights
            # Store history as Python-friendly types (float, numpy array)
            # Use .numpy() where available (eager execution assumed in notebook)
            self.ess_history.append(float(ess.numpy()))
            self.weights_history.append(weights.numpy())
        except Exception:
            # Fallback: store tensors if .numpy() not available
            self.ess_history.append(float(ess))
            self.weights_history.append(weights)

        # Resample if ESS below threshold (N/2)
        threshold = tf.cast(self.num_particles / 2.0, dtype=self.dtype)
        resampled = False
        if ess < threshold:
            idx = self._systematic_resample(weights)
            # gather particles
            self.particles = tf.gather(self.particles, idx)
            # reset weights to uniform in log-space
            self.log_weights = tf.fill([self.num_particles], tf.math.log(1.0 / tf.cast(self.num_particles, self.dtype)))
            resampled = True

        if self.verbose:
            try:
                ess_val = float(ess.numpy())
            except Exception:
                ess_val = float(ess)
            print(f"ParticleFilter: update() done; ESS={ess_val:.2f}; resampled={resampled}")

        # Return approximate log-likelihood (log p(y))
        return log_norm_const

    def estimate(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute weighted mean and covariance of the particles.

        Returns
        -------
        mean : tf.Tensor shape (state_dim,)
        cov : tf.Tensor shape (state_dim, state_dim)
        """
        w = tf.exp(self.log_weights)  # (N,)
        # Weighted mean: sum_i w_i * x_i
        mean = tf.reduce_sum(tf.expand_dims(w, -1) * self.particles, axis=0)

        diff = self.particles - mean  # (N, state_dim)
        # Covariance: sum w_i * (diff_i @ diff_i^T)
        cov = tf.transpose(diff) @ (tf.expand_dims(w, -1) * diff)
        cov = tf.cast(cov, dtype=self.dtype)
        if self.verbose:
            print("ParticleFilter: estimate() returning mean and covariance")
        return mean, cov

    def step(self, observation, true_state=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Perform a predict-update cycle and return estimate and loglik.

        Parameters
        ----------
        observation : array-like
            Observation for current time step
        true_state : array-like or tf.Tensor, optional
            True state at this time step for RMSE calculation. If not provided,
            RMSE will not be recorded for this step.
        """
        if self.verbose:
            print("ParticleFilter: step() starting predict-update cycle")
        self.predict()
        loglik = self.update(observation)
        mean, cov = self.estimate()

        # Record per-step log-likelihood
        try:
            self.ll_history.append(float(loglik.numpy()))
        except Exception:
            try:
                self.ll_history.append(float(loglik))
            except Exception:
                self.ll_history.append(None)

        # RMSE if true state provided
        if true_state is not None:
            try:
                true_t = tf.convert_to_tensor(true_state, dtype=self.dtype)
                diff = tf.reshape(mean - true_t, [-1])
                rmse = tf.sqrt(tf.reduce_mean(tf.square(diff)))
                try:
                    self.rmse_history.append(float(rmse.numpy()))
                except Exception:
                    self.rmse_history.append(float(rmse))
            except Exception:
                self.rmse_history.append(None)

        if self.verbose:
            print("ParticleFilter: step() finished")
        return mean, cov, loglik

    def filter(self, observations, true_states=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Filter a sequence of observations.

        Runs sequential predict-update cycles over `observations` by calling
        `self.step` for each observation. Returns stacked tensors of
        means, covariances and log-likelihoods for each time step.

        Parameters
        ----------
        observations : array-like or tf.Tensor
            Sequence of observations. Can be a 1D array-like for scalar
            observations (obs_dim == 1) or shape (T, obs_dim).

        Returns
        -------
        means : tf.Tensor, shape (T, state_dim)
        covs : tf.Tensor, shape (T, state_dim, state_dim)
        logliks : tf.Tensor, shape (T,)
        """
        # Convert to numpy array for simple iteration (keeps API flexible)
        obs_arr = np.asarray(observations)

        # Handle scalar-observation case: ensure first axis is time
        if obs_arr.ndim == 0:
            obs_arr = np.expand_dims(obs_arr, axis=0)

        T = int(obs_arr.shape[0])

        means_list = []
        covs_list = []
        logliks_list = []

        for t in range(T):
            y_t = obs_arr[t]
            if true_states is not None:
                try:
                    true_t = true_states[t]
                except Exception:
                    true_t = None
                mean, cov, ll = self.step(y_t, true_state=true_t)
            else:
                mean, cov, ll = self.step(y_t)
            means_list.append(mean)
            covs_list.append(cov)
            logliks_list.append(ll)

        # Stack results along time axis
        means = tf.stack(means_list, axis=0)
        covs = tf.stack(covs_list, axis=0)
        logliks = tf.stack(logliks_list, axis=0)

        return means, covs, logliks


if __name__ == "__main__":
    # Small demo: use the same nonlinear growth model as EKF example
    import matplotlib.pyplot as plt

    T = 40
    rng = np.random.default_rng(2)

    # True system
    def sim_f(x):
        return x + 0.05 * (x ** 2)

    def sim_h(x):
        return x + 0.1 * (x ** 2)

    Q = np.array([[0.01]])
    R = np.array([[0.1]])

    x = np.zeros(T + 1)
    y = np.zeros(T)
    x[0] = 0.0
    for t in range(T):
        x[t + 1] = sim_f(x[t]) + rng.normal(scale=np.sqrt(Q[0, 0]))
        y[t] = sim_h(x[t + 1]) + rng.normal(scale=np.sqrt(R[0, 0]))

    # TensorFlow wrappers
    def f_tf(x_tf: tf.Tensor) -> tf.Tensor:
        return x_tf + 0.05 * tf.square(x_tf)

    def h_tf(x_tf: tf.Tensor) -> tf.Tensor:
        return x_tf + 0.1 * tf.square(x_tf)

    # Create PF
    pf = ParticleFilter(
        transition_fn=f_tf,
        observation_fn=h_tf,
        Q=Q,
        R=R,
        num_particles=1000,
        initial_mean=tf.constant([0.0], dtype=tf.float64),
        initial_cov=tf.constant([[1.0]], dtype=tf.float64),
        seed=2,
    )

    est_means = []
    for t in range(T):
        mean, cov, ll = pf.step(y[t])
        est_means.append(float(mean.numpy()[0]))

    est_means = np.array(est_means)
    rmse = np.sqrt(np.mean((est_means - x[1:]) ** 2))
    print(f"Particle filter RMSE: {rmse:.4f}")

    # Quick plot
    plt.figure()
    plt.plot(range(T), x[1:], label="True")
    plt.plot(range(T), y, '.', alpha=0.3, label="Obs")
    plt.plot(range(T), est_means, label="PF mean")
    plt.legend()
    plt.title("Particle Filter demo")
    plt.show()
