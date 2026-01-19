# ./experiments/multidim_nongaussian_ssm.py
"""
Multi-dimensional Non-Gaussian State-Space Model Experiment

Model:
    State: x_k ∈ ℝ^d (d=20)
    Dynamics: Lorenz-96 system with Student-t process noise
        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    Observation: y_k = H(x_k) + r_k, where H(x) = [x₁², sin(x₂), ..., x_d²] + Student-t noise

Filters:
    1. Standard Particle Filter (PF)
    2. EDH Flow (Exact Daum-Huang)
    3. LEDH Flow (Localized EDH)
    4. PFPF-EDH (Particle Flow Particle Filter with EDH)
    5. PFPF-LEDH (Particle Flow Particle Filter with LEDH)
"""

import os
import sys
import time
import tracemalloc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Environment setup
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t

from data.data import MultidimNonGaussianSSM
from models.particle_filters import (
    StandardParticleFilter,
    algorithm1_DH_filter,
    algorithm2_modified_DH_filter,
    PFPF_EDH,
    PFPF_LEDH
)


# =============================================================================
# TensorFlow-native dynamics and observation model
# =============================================================================

@tf.function
def lorenz96_step_tf(x, F=8.0, dt=0.005):
    """Euler step for Lorenz-96 system using TensorFlow ops."""
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    F = tf.cast(F, x.dtype)
    dt = tf.cast(dt, x.dtype)

    orig_shape = tf.shape(x)
    if x.shape.ndims == 1:
        x = tf.expand_dims(x, axis=0)
        squeeze = True
    else:
        squeeze = False

    d = tf.shape(x)[-1]
    x_im2 = tf.roll(x, shift=2, axis=-1)
    x_im1 = tf.roll(x, shift=1, axis=-1)
    x_ip1 = tf.roll(x, shift=-1, axis=-1)

    dxdt = (x_ip1 - x_im2) * x_im1 - x + F
    x_next = x + dt * dxdt

    if squeeze:
        x_next = tf.squeeze(x_next, axis=0)
    return x_next


class Lorenz96Dynamics:
    def __init__(self, F=8.0, dt=0.005, dtype=tf.float64):
        self.F = tf.constant(F, dtype=dtype)
        self.dt = tf.constant(dt, dtype=dtype)

    @tf.function
    def __call__(self, x):
        return lorenz96_step_tf(x, F=self.F, dt=self.dt)


@tf.function
def h_obs(x):
    """Observation function: [x1^2, sin(x2), x3^2, sin(x4), ...]"""
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    if x.shape.ndims == 1:
        x = tf.expand_dims(x, axis=0)
        squeeze = True
    else:
        squeeze = False

    d = tf.shape(x)[-1]
    indices = tf.range(d, dtype=tf.int32)
    is_even = tf.equal(indices % 2, 0)

    # Compute element-wise
    squared = tf.square(x)
    sine = tf.sin(x)
    y = tf.where(is_even, squared, sine)

    if squeeze:
        y = tf.squeeze(y, axis=0)
    return y


@tf.function
def dh_dx(x):
    """Jacobian of h_obs: diagonal matrix with 2*x_i (even) or cos(x_i) (odd)."""
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    if x.shape.ndims == 1:
        x = tf.expand_dims(x, axis=0)
        squeeze = True
    else:
        squeeze = False

    d = tf.shape(x)[-1]
    indices = tf.range(d, dtype=tf.int32)
    is_even = tf.equal(indices % 2, 0)

    diag_vals = tf.where(is_even, 2.0 * x, tf.cos(x))  # (N, d)
    J = tf.linalg.diag(diag_vals)  # (N, d, d)

    if squeeze:
        J = tf.squeeze(J, axis=0)  # (d, d)
    return J

# Observation model for Lorenz-96: [x1^2, sin(x2), x3^2, ...]
def h_pf(x):
    """x: (N, d) → returns (N, d)"""
    x = np.asarray(x)
    obs = np.empty_like(x)
    for i in range(x.shape[1]):
        if i % 2 == 0:
            obs[:, i] = x[:, i] ** 2
        else:
            obs[:, i] = np.sin(x[:, i])
    return obs


# =============================================================================
# Main experiment
# =============================================================================

def main():
    T = 50
    dim = 20
    seed = 42

    print("Generating multi-dimensional non-Gaussian SSM data...")
    ssm = MultidimNonGaussianSSM(dim=dim, F=8.0, dt=0.005, df_process=3.0, df_obs=3.0, seed=seed)
    true_x, obs_y = ssm.generate_data(T=T)

    # Use TensorFlow-native dynamics
    f = Lorenz96Dynamics(F=8.0, dt=0.005, dtype=tf.float64)

    # Common parameters
    m0 = np.zeros(dim)
    P0 = np.eye(dim) * 2.0
    Q = np.eye(dim) * (0.1 ** 2)

    results = {}
    runtimes = {}
    memories = {}
    ess_histories = {}
    flow_histories = {}

    # --- Standard PF (scalar observation only) ---
    print("Running Standard Particle Filter (PF)...")
    # Observation noise covariance (diagonal, scale=0.2 → variance=0.04)
    R = np.eye(dim) * (0.2 ** 2)  # matches df_obs scale in data generation
    tracemalloc.start()
    start = time.time()
    pf = StandardParticleFilter(
        f=lambda x: ssm.lorenz96_step(x),
        h=h_pf,          
        Q=Q,
        R=R,             
        initial_mean=m0,
        initial_cov=P0,
        num_particles=5000,
        seed=seed
    )
    pf_means = pf.filter(obs_y[:, 0])
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['PF'] = pf_means
    runtimes['PF'] = end - start
    memories['PF'] = peak / 1024 / 1024
    ess_histories['PF'] = pf.ess_history

    # --- Algorithm 1: Global DH Flow (replaces EDH) ---
    print("Running Algorithm 1 (Global DH Flow)...")
    tracemalloc.start()
    start = time.time()

    # Dynamics matrix Psi (for linearized dynamics in DH filters)
    # Since Lorenz-96 is nonlinear, approximate as identity for small dt
    Psi = np.eye(dim)  # or use Jacobian of f at mean

    # Observation noise covariance
    R = np.eye(dim)  # assume R = I

    dh_means = algorithm1_DH_filter(
        y_seq=obs_y,
        T=T,
        N=3000,
        Psi=Psi,
        Q=Q,
        R=R,
        gamma=h_obs,          # TensorFlow-compatible observation function
        dgamma_dx=dh_dx,      # TensorFlow-compatible Jacobian
        x0_mean=m0,
        x0_cov=P0,
        n_lambda=29,
        dtype=tf.float64
    )
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['EDH'] = dh_means
    runtimes['EDH'] = end - start
    memories['EDH'] = peak / 1024 / 1024

    # --- Algorithm 2: Local DH Flow (replaces LEDH) ---
    print("Running Algorithm 2 (Local DH Flow)...")
    tracemalloc.start()
    start = time.time()
    dh_local_means = algorithm2_modified_DH_filter(
        y_seq=obs_y,
        T=T,
        N=3000,
        Psi=Psi,
        Q=Q,
        R=R,
        gamma=h_obs,
        dgamma_dx=dh_dx,
        x0_mean=m0,
        x0_cov=P0,
        n_lambda=29,
        dtype=tf.float64
    )
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['LEDH'] = dh_local_means
    runtimes['LEDH'] = end - start
    memories['LEDH'] = peak / 1024 / 1024

    # --- PFPF-EDH (full multivariate) ---
    print("Running PFPF-EDH...")
    tracemalloc.start()
    start = time.time()
    pfpf_edh = PFPF_EDH(
        f=f, Q=Q, h=h_obs, dh_dx=dh_dx,
        initial_mean=m0, initial_cov=P0,
        num_particles=3000, seed=seed, dtype=tf.float64,
        dy=dim  #
    )
    pfpf_edh_means = pfpf_edh.filter(obs_y)
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['PFPF-EDH'] = pfpf_edh_means
    runtimes['PFPF-EDH'] = end - start
    memories['PFPF-EDH'] = peak / 1024 / 1024
    flow_histories['PFPF-EDH'] = pfpf_edh.flow_history

    # --- PFPF-LEDH (full multivariate) ---
    print("Running PFPF-LEDH...")
    tracemalloc.start()
    start = time.time()
    pfpf_ledh = PFPF_LEDH(
        f=f, Q=Q, h=h_obs, dh_dx=dh_dx,
        initial_mean=m0, initial_cov=P0,
        num_particles=3000, seed=seed, dtype=tf.float64,
        dy=dim  #
    )
    pfpf_ledh_means = pfpf_ledh.filter(obs_y)
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['PFPF-LEDH'] = pfpf_ledh_means
    runtimes['PFPF-LEDH'] = end - start
    memories['PFPF-LEDH'] = peak / 1024 / 1024
    flow_histories['PFPF-LEDH'] = pfpf_ledh.flow_history

    t = np.arange(T)

    # --- Figure 1: 2D State Trajectory (x1 vs x2) ---
    plt.figure(figsize=(8, 8))
    # True trajectory
    plt.plot(true_x[:, 0], true_x[:, 1], 'k-', label='True', linewidth=2)

    # Estimated trajectories
    for name, est in results.items():
        plt.plot(est[:, 0], est[:, 1], '--', label=name, alpha=0.8)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('2D State Trajectory ($x_1$ vs $x_2$)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Optional: equal scaling for isotropic view
    plt.tight_layout()
    plt.savefig("./figures/state_trajectory_2d.pdf", dpi=150)
    plt.show()

    # --- Figure 2: Observations ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        axes[i].plot(t, obs_y[:, i], 'ko', markersize=3, alpha=0.7)
        axes[i].set_ylabel(f'$y_{{{i+1}}}$')
        axes[i].grid(True)
    axes[-1].set_xlabel('Time')
    fig.suptitle('Observations (First 3 Dimensions)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("./figures/observations.pdf", dpi=150)
    plt.show()

    # --- Figure 3: ESS (PF only) ---
    if 'PF' in ess_histories:
        plt.figure(figsize=(8, 4))
        plt.plot(t, ess_histories['PF'], 'm-', linewidth=1.5)
        plt.axhline(y=5000/2, color='r', linestyle='--', label='Resampling Threshold')
        plt.xlabel('Time')
        plt.ylabel('ESS')
        plt.title('Effective Sample Size (Standard PF)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./figures/ess_pf.pdf", dpi=150)
        plt.show()

    # --- Figure 4: Flow Magnitude ---
    plt.figure(figsize=(10, 4))
    if 'EDH' in flow_histories:
        plt.plot(t, flow_histories['EDH'], 'b-', label='EDH', linewidth=1.2)
    if 'PFPF-EDH' in flow_histories:
        plt.plot(t, flow_histories['PFPF-EDH'], 'c--', label='PFPF-EDH', linewidth=1.2)
    if 'LEDH' in flow_histories:
        plt.plot(t, flow_histories['LEDH'], 'g-', label='LEDH', linewidth=1.2)
    if 'PFPF-LEDH' in flow_histories:
        plt.plot(t, flow_histories['PFPF-LEDH'], 'y--', label='PFPF-LEDH', linewidth=1.2)
    plt.xlabel('Time')
    plt.ylabel('Mean Flow Magnitude')
    plt.title('Flow Magnitude Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures/flow_magnitude.pdf", dpi=150)
    plt.show()

    # --- Figure 5: RMSE Comparison ---
    rmse_values = []
    labels = ['PF', 'EDH', 'LEDH', 'PFPF-EDH', 'PFPF-LEDH']
    for name in labels:
        if name not in results:
            continue
        est = results[name]
        rmse = np.sqrt(np.mean((est - true_x)**2))
        rmse_values.append(rmse)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, rmse_values, color=['red', 'blue', 'green', 'cyan', 'orange'])
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison Across Methods')
    plt.xticks(rotation=30)
    plt.grid(axis='y')
    for bar, rmse in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{rmse:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("./figures/rmse_comparison.pdf", dpi=150)
    plt.show()

    # --- Figure 6: Runtime ---
    runtime_vals = [runtimes[name] for name in labels if name in runtimes]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, runtime_vals, color=['red', 'blue', 'green', 'cyan', 'orange'])
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.xticks(rotation=30)
    plt.grid(axis='y')
    for bar, rt in zip(bars, runtime_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{rt:.1f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("./figures/runtime_comparison.pdf", dpi=150)
    plt.show()

    # --- Performance Summary ---
    print("\n" + "="*80)
    print("MULTI-DIMENSIONAL NON-GAUSSIAN SSM PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Method':15s} | {'Time (s)':>8s} | {'RAM (MB)':>8s} | {'RMSE':>8s}")
    print("-" * 80)
    
    # Define which methods are scalar-only (only estimate dim 0)
    SCALAR_METHODS = {'PF'}
    labels = ['PF', 'EDH', 'LEDH', 'PFPF-EDH', 'PFPF-LEDH']
    for name in labels:
        if name not in results:
            continue
        est = results[name]
        
        rmse = np.sqrt(np.mean((est - true_x)**2))
        
        print(f"{name:15s} | {runtimes[name]:8.1f} | {memories[name]:8.0f} | {rmse:8.4f}")


if __name__ == "__main__":
    main()