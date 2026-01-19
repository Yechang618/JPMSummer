"""
Stochastic Volatility Model (SVM) Filtering Experiment

Model:
    X_k = alpha * X_{k-1} + sigma * eta_k,   eta_k ~ N(0, I)
    Y_k = beta * exp(X_k / 2) * eps_k,       eps_k ~ N(0, I)

Filters:
    - EKF (using log(y^2) trick)
    - UKF (using raw y with nonlinear h(x))
    - Particle Filter (PF)
    - DH Algorithm 1 (Global EDH flow)
    - DH Algorithm 2 (Local EDH flow)
"""

import os
import sys
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.set_visible_devices([], 'GPU')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import data
from models.kalman_filters import UnscentedKalmanFilter  # <-- Added
from models.particle_filters import (
    algorithm1_DH_filter,
    algorithm2_modified_DH_filter,
    PFPF_EDH,
    PFPF_LEDH
)

# ----------------------------
# 1. SVM Data Generator
# ----------------------------

def generate_svm_data(T=200, alpha=0.95, sigma=0.2, beta=1.0, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(T + 1)
    y = np.zeros(T)
    x[0] = 0.0
    for t in range(1, T + 1):
        x[t] = alpha * x[t - 1] + sigma * rng.normal()
        y[t - 1] = beta * np.exp(x[t] / 2) * rng.normal()
    return x[1:], y  # (T,), (T,)

# ----------------------------
# 2. Observation Model for PFPF/DH & UKF
# ----------------------------

@tf.function
def gamma_svm(x):
    """Observation: y = beta * exp(x/2) * eps → we use raw y (scalar)"""
    if x.shape.ndims == 1:
        x = tf.expand_dims(x, axis=0)
    beta = 0.5
    y_pred = beta * tf.exp(x[:, 0] / 2.0)
    return tf.expand_dims(y_pred, axis=-1)  # (N, 1)

@tf.function
def dgamma_dx_svm(x):
    if x.shape.ndims == 1:
        x = tf.expand_dims(x, axis=0)
    beta = 0.5
    dy_dx = 0.5 * beta * tf.exp(x[:, 0] / 2.0)
    return tf.expand_dims(dy_dx, axis=-1)

@tf.function
def dgamma_dx_svm_global(x):
    beta = 0.5
    dy_dx = 0.5 * beta * tf.exp(x[0] / 2.0)
    return tf.reshape(dy_dx, [1, 1])

# For UKF: define h(x) as a callable that works with NumPy or TF scalars
def h_ukf(x):
    """Nonlinear observation function for UKF: h(x) = beta * exp(x/2)"""
    beta = 0.5
    return np.array([beta * np.exp(x[0] / 2.0)])

# ----------------------------
# 3. Baseline Filters (EKF, PF, UKF)
# ----------------------------

class SVMEKF:
    def __init__(self, alpha, sigma, beta, m0=0.0, P0=1.0):
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.m = m0
        self.P = P0

    def step(self, obs_log_y2):
        m_pred = self.alpha * self.m
        P_pred = (self.alpha ** 2) * self.P + self.sigma ** 2
        h_x = np.log(self.beta ** 2) + m_pred
        H = 1.0
        R = 4.0
        S = H * P_pred * H + R
        K = P_pred * H / S
        innov = obs_log_y2 - h_x
        self.m = m_pred + K * innov
        self.P = P_pred - K * S * K
        return self.m, self.P

    def filter(self, obs_log_y2):
        means, vars = [], []
        for y in obs_log_y2:
            m, v = self.step(y)
            means.append(m)
            vars.append(v)
        return np.array(means), np.array(vars)

class SVMParticleFilter:
    def __init__(self, alpha, sigma, beta, num_particles=2000, seed=0):
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.num_particles = num_particles
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.normal(0.0, 1.0, size=num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.ess_history = []

    def predict(self):
        noise = self.rng.normal(0, self.sigma, size=self.num_particles)
        self.particles = self.alpha * self.particles + noise

    def update(self, observation):
        var = (self.beta ** 2) * np.exp(self.particles)
        log_lik = -0.5 * (np.log(2 * np.pi) + np.log(var) + (observation ** 2) / var)
        self.weights *= np.exp(log_lik)
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)
        ess = 1.0 / np.sum(self.weights ** 2)
        self.ess_history.append(ess)
        if ess < self.num_particles / 2:
            u0 = self.rng.random() / self.num_particles
            positions = (u0 + np.arange(self.num_particles)) % 1
            cumsum_weights = np.cumsum(self.weights)
            indices = np.searchsorted(cumsum_weights, positions)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        mean = np.sum(self.weights * self.particles)
        var = np.sum(self.weights * (self.particles - mean)**2)
        return mean, var

    def filter(self, observations):
        means, vars = [], []
        for y in observations:
            self.predict()
            self.update(y)
            m, v = self.estimate()
            means.append(m)
            vars.append(v)
        return np.array(means), np.array(vars)

# ----------------------------
# 4. Main Experiment
# ----------------------------

def main():
    T = 200
    alpha, sigma, beta = 0.91, 0.2, 0.5
    seed = 42

    true_x, obs_y = data.generate_svm_data(T=T, alpha=alpha, sigma=sigma, beta=beta, seed=seed)
    obs_log_y2 = np.log(obs_y**2 + 1e-8)

    m0, P0 = 0.0, 1.0
    Q = np.array([[sigma**2]], dtype=np.float64)
    R = np.array([[1.0]], dtype=np.float64)  # observation noise variance
    Psi = np.array([[alpha]], dtype=np.float64)

    results = {}
    runtimes = {}
    memories = {}
    ess_history = None

    # --- EKF ---
    print("Running EKF...")
    tracemalloc.start(); start = time.time()
    ekf = SVMEKF(alpha, sigma, beta, m0, P0)
    ekf_means, _ = ekf.filter(obs_log_y2)
    end = time.time(); current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    results['EKF'] = ekf_means
    runtimes['EKF'] = end - start; memories['EKF'] = peak / 1024 / 1024

    # --- UKF ---
    print("Running UKF...")
    tracemalloc.start(); start = time.time()
    ukf = UnscentedKalmanFilter(
        f=lambda x: np.array([alpha * x[0]]),
        h=h_ukf,
        Q=Q,
        R=R,
        initial_mean=np.array([m0]),
        initial_cov=np.array([[P0]]),  # <-- Critical fix
        dtype=np.float64
    )
    ukf_means, _, _ = ukf.filter(obs_y, true_states=true_x)
    end = time.time(); current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    results['UKF'] = ukf_means[:, 0]
    runtimes['UKF'] = end - start; memories['UKF'] = peak / 1024 / 1024

    # --- PF ---
    print("Running Particle Filter...")
    tracemalloc.start(); start = time.time()
    pf = SVMParticleFilter(alpha, sigma, beta, num_particles=3000, seed=seed)
    pf_means, _ = pf.filter(obs_y)
    ess_history = pf.ess_history
    end = time.time(); current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    results['PF'] = pf_means
    runtimes['PF'] = end - start; memories['PF'] = peak / 1024 / 1024

    # Prepare data for DH/PFPF
    y_obs_f64 = obs_y.astype(np.float64).reshape(-1, 1)
    x0_mean_f64 = np.array([m0], dtype=np.float64)
    P0_f64 = np.array([[P0]], dtype=np.float64)

    # --- DH Algorithm 1 (Global) ---
    print("Running DH Algorithm 1 (Global EDH)...")
    tracemalloc.start(); start = time.time()
    dh1_est = algorithm1_DH_filter(
        y_seq=y_obs_f64,
        T=T,
        N=2000,
        Psi=Psi,
        Q=Q,
        R=R,
        gamma=gamma_svm,
        dgamma_dx=dgamma_dx_svm_global,
        x0_mean=x0_mean_f64,
        x0_cov=P0_f64,
        n_lambda=29,
        dtype=tf.float64
    )
    end = time.time(); current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    results['EDH'] = dh1_est[:, 0]
    runtimes['EDH'] = end - start; memories['EDH'] = peak / 1024 / 1024

    # --- DH Algorithm 2 (Local) ---
    print("Running DH Algorithm 2 (Local EDH)...")
    tracemalloc.start(); start = time.time()
    dh2_est = algorithm2_modified_DH_filter(
        y_seq=y_obs_f64,
        T=T,
        N=2000,
        Psi=Psi,
        Q=Q,
        R=R,
        gamma=gamma_svm,
        dgamma_dx=dgamma_dx_svm,
        x0_mean=x0_mean_f64,
        x0_cov=P0_f64,
        n_lambda=29,
        dtype=tf.float64
    )
    end = time.time(); current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    results['LEDH'] = dh2_est[:, 0]
    runtimes['LEDH'] = end - start; memories['LEDH'] = peak / 1024 / 1024

    t = np.arange(T)

    # --- Figure 1: Latent State Estimation ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, true_x, 'k-', label='True $x_t$', linewidth=2)
    for name, est in results.items():
        plt.plot(t, est, '--', label=name)
    plt.xlabel('Time'); plt.ylabel('Latent State')
    plt.title('SVM: Latent State Estimation')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/svm_latent_state.pdf", dpi=150)
    plt.show()

    # --- Figure 2: Estimated Volatility ---
    plt.figure(figsize=(10, 6))
    true_vol = beta * np.exp(true_x / 2)
    plt.plot(t, true_vol, 'k-', label='True $\sigma_t$', linewidth=2)
    for name, est in results.items():
        vol_est = beta * np.exp(est / 2)
        plt.plot(t, vol_est, '--', label=name)
    plt.xlabel('Time'); plt.ylabel('Volatility')
    plt.title('SVM: Estimated Volatility')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures/svm_volatility.pdf", dpi=150)
    plt.show()

    # --- Figure 3: Observations ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, obs_y, 'ko', markersize=2, alpha=0.7)
    plt.xlabel('Time'); plt.ylabel('$y_t$')
    plt.title('SVM Observations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures/svm_observations.pdf", dpi=150)
    plt.show()

    # --- Figure 4: ESS (PF only) ---
    if ess_history:
        plt.figure(figsize=(10, 4))
        plt.plot(t, ess_history, 'm-')
        plt.axhline(y=3000/2, color='r', linestyle='--', label='Resampling Threshold')
        plt.xlabel('Time'); plt.ylabel('ESS')
        plt.title('Effective Sample Size (PF)')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig("./figures/svm_ess.pdf", dpi=150)
        plt.show()

    # --- Performance Summary ---
    print("\n" + "="*80)
    print("SVM FILTERING PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Method':12s} | {'Time (s)':>8s} | {'RAM (MB)':>8s} | {'RMSE':>8s}")
    print("-" * 80)
    for name in ['EKF', 'UKF', 'PF', 'EDH', 'LEDH']:
        if name not in results:
            continue
        rmse = np.sqrt(np.mean((results[name] - true_x)**2))
        print(f"{name:12s} | {runtimes[name]:8.3f} | {memories[name]:8.2f} | {rmse:8.4f}")

if __name__ == "__main__":
    main()