# ./experiments/compare_range_bearing_filters.py
"""
Experiment: Compare EKF, UKF, and Particle Filters on Range-Bearing SSM.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data import RangeBearingSSM
from models.kalman_filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from models.particle_filters import (
    StandardParticleFilter,
    algorithm1_DH_filter,
    algorithm2_modified_DH_filter,
    PFPF_EDH,      
    PFPF_LEDH     
)

# Disable GPU if not needed (optional)
tf.config.set_visible_devices([], 'GPU')
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# =============================================================================
# 1. Define measurement model for EKF/UKF
# =============================================================================

def gamma(x):
    """Measurement function: [r, b] = [sqrt(x^2+y^2), atan2(y,x)]"""
    x_pos, y_pos = x[..., 0], x[..., 1]
    r = tf.sqrt(x_pos**2 + y_pos**2)
    b = tf.math.atan2(y_pos, x_pos)
    return tf.stack([r, b], axis=-1)

def dgamma_dx(x):
    x_pos, y_pos = x[0], x[1]
    r_sq = x_pos**2 + y_pos**2
    
    # Stronger regularization for near-origin
    r_safe = tf.sqrt(r_sq + 1e-4)  # was 1e-8, now 1e-4
    r_sq_safe = r_sq + 1e-4        # was 1e-8, now 1e-4
    
    dr_dx = x_pos / r_safe
    dr_dy = y_pos / r_safe
    db_dx = -y_pos / r_sq_safe
    db_dy = x_pos / r_sq_safe
    
    return tf.stack([
        [dr_dx, dr_dy, 0.0, 0.0],
        [db_dx, db_dy, 0.0, 0.0]
    ])

@tf.function
def dgamma_dx_batch(x):
    """Batch Jacobian for PFPF_LEDH (x: [N, 4])"""
    if x.shape.ndims == 1:
        return dgamma_dx(x)  # Use original for single particle
    
    x_pos = x[:, 0]  # (N,)
    y_pos = x[:, 1]  # (N,)
    r_sq = x_pos**2 + y_pos**2
    r_safe = tf.sqrt(r_sq + 1e-4)
    r_sq_safe = r_sq + 1e-4

    dr_dx = x_pos / r_safe
    dr_dy = y_pos / r_safe
    db_dx = -y_pos / r_sq_safe
    db_dy = x_pos / r_sq_safe

    # Build Jacobian matrix for each particle: [N, 2, 4]
    J = tf.stack([
        tf.stack([dr_dx, dr_dy, tf.zeros_like(dr_dx), tf.zeros_like(dr_dx)], axis=1),
        tf.stack([db_dx, db_dy, tf.zeros_like(db_dx), tf.zeros_like(db_dx)], axis=1)
    ], axis=1)  # (N, 2, 4)
    return J
# =============================================================================
# 2. Generate data
# =============================================================================

T = 100
ssm = RangeBearingSSM(
    dt=1.0,
    process_noise_std=0.1,
    range_std=0.5,
    bearing_std=0.05,
    initial_state=[0.0, 0.0, 1.0, 0.5],
    dtype=tf.float64
)

x_true_tf, y_obs_tf = ssm.sample(T, seed=42)
x_true = x_true_tf.numpy()[1:]  # (T, 4)
y_obs = y_obs_tf.numpy()        # (T, 2)

# =============================================================================
# 3. Run Kalman Filters
# =============================================================================

# Shared parameters
A = ssm.A.numpy()
Q = ssm.Q.numpy()
R = ssm.R.numpy()
x0_mean = ssm.initial_state.numpy()
P0 = np.eye(4) * 1.0

# --- EKF ---
# --- EKF ---
ekf = ExtendedKalmanFilter(
    f=lambda x: tf.linalg.matvec(A, x),  # ✅ safe for 1D tensors
    h=gamma,
    Q=Q, R=R,
    initial_mean=x0_mean,
    initial_cov=P0,
    dtype=tf.float64
)
start = time.time()
ekf_means, _, _ = ekf.filter(y_obs, true_states=x_true)
ekf_time = time.time() - start
ekf_rmse = np.sqrt(np.mean((x_true - ekf_means)**2))

# --- UKF ---
ukf = UnscentedKalmanFilter(
    f=lambda x: tf.linalg.matvec(A, x),
    h=gamma,
    Q=Q, R=R,
    initial_mean=x0_mean,
    initial_cov=P0,
    dtype=tf.float64
)
start = time.time()
ukf_means, _, _ = ukf.filter(y_obs, true_states=x_true)
ukf_time = time.time() - start
ukf_rmse = np.sqrt(np.mean((x_true - ukf_means)**2))

# =============================================================================
# 4. Run Particle Filters
# =============================================================================

N_PARTICLES = 1000
seed = 42

# Helper: linear dynamics
def f_particle(x):
    return A @ x

# --- Standard PF ---
pf = StandardParticleFilter(
    f=f_particle,
    Q=Q,
    initial_mean=x0_mean,
    initial_cov=P0,
    num_particles=N_PARTICLES,
    seed=seed
)

Psi = A          # already float64 from RangeBearingSSM
Q_mat = Q        # same
R_mat = R
x0_mean_f64 = x0_mean
P0_f64 = P0
y_obs_f64 = y_obs

# Algorithm 1 (Global)
start = time.time()
dh1_est = algorithm1_DH_filter(
    y_obs_f64, T, N_PARTICLES,
    Psi, Q_mat, R_mat,
    gamma, dgamma_dx,
    x0_mean_f64, P0_f64,
    n_lambda=10
)
dh1_time = time.time() - start
dh1_rmse = np.sqrt(np.mean((x_true - dh1_est)**2))

# Algorithm 2 (Local)
start = time.time()
dh2_est = algorithm2_modified_DH_filter(
    y_obs_f64, T, N_PARTICLES,
    Psi, Q_mat, R_mat,
    gamma, dgamma_dx,
    x0_mean_f64, P0_f64,
    n_lambda=10
)
dh2_time = time.time() - start
dh2_rmse = np.sqrt(np.mean((x_true - dh2_est)**2))

# --- PFPF-EDH ---
print("Running PFPF-EDH...")
start = time.time()
pfpf_edh = PFPF_EDH(
    f=lambda x: tf.linalg.matvec(A, x),          # Linear dynamics
    Q=Q,
    h=gamma,                    # Range-bearing observation
    dh_dx=dgamma_dx,            # Single-particle Jacobian
    R = R,
    initial_mean=x0_mean,
    initial_cov=P0,
    num_particles=N_PARTICLES,
    seed=seed,
    dy=2,                       # Observation dimension
    dtype=tf.float64
)
pfpf_edh_est = pfpf_edh.filter(y_obs)
pfpf_edh_time = time.time() - start
pfpf_edh_rmse = np.sqrt(np.mean((x_true - pfpf_edh_est)**2))

# --- PFPF-LEDH ---
print("Running PFPF-LEDH...")
start = time.time()
pfpf_ledh = PFPF_LEDH(
    f=lambda x: tf.linalg.matvec(A, x),          # Linear dynamics
    Q=Q,
    h=gamma,
    dh_dx=dgamma_dx_batch,      # Batch Jacobian
    R= R,
    initial_mean=x0_mean,
    initial_cov=P0,
    num_particles=N_PARTICLES,
    seed=seed,
    dy=2,
    dtype=tf.float64
)
pfpf_ledh_est = pfpf_ledh.filter(y_obs)
pfpf_ledh_time = time.time() - start
pfpf_ledh_rmse = np.sqrt(np.mean((x_true - pfpf_ledh_est)**2))


# =============================================================================
# 5. Results
# =============================================================================

print("\n=== RMSE (position only) ===")
ekf_means, _, _ = ekf.filter(y_obs, true_states=x_true)
ekf_means = ekf_means.numpy()  # ✅ convert to NumPy

ukf_means, _, _ = ukf.filter(y_obs, true_states=x_true)
ukf_means = ukf_means.numpy()

# DH outputs are already NumPy (from .numpy() in loop)

# Now safe to use list indexing
pos_idx = [0, 1]

print("\n=== RMSE (position only) ===")
print(f"EKF          : {np.sqrt(np.mean((x_true[:,pos_idx] - ekf_means[:,pos_idx])**2)):.4f}")
print(f"UKF          : {np.sqrt(np.mean((x_true[:,pos_idx] - ukf_means[:,pos_idx])**2)):.4f}")
print(f"EDH     : {np.sqrt(np.mean((x_true[:,pos_idx] - dh1_est[:,pos_idx])**2)):.4f}")
print(f"LEDH     : {np.sqrt(np.mean((x_true[:,pos_idx] - dh2_est[:,pos_idx])**2)):.4f}")
print(f"PFPF-EDH     : {np.sqrt(np.mean((x_true[:,pos_idx] - pfpf_edh_est[:,pos_idx])**2)):.4f}")
print(f"PFPF-LEDH    : {np.sqrt(np.mean((x_true[:,pos_idx] - pfpf_ledh_est[:,pos_idx])**2)):.4f}")
# =============================================================================
# 6. Plotting
# =============================================================================

plt.figure(figsize=(12, 5))

# Trajectory
plt.figure(figsize=(8, 8))
plt.plot(x_true[:, 0], x_true[:, 1], 'k-', label='True', linewidth=2)
plt.plot(ekf_means[:, 0], ekf_means[:, 1], 'b--', label='EKF')
plt.plot(ukf_means[:, 0], ukf_means[:, 1], 'g--', label='UKF')
plt.plot(dh1_est[:, 0], dh1_est[:, 1], 'c-.', label='EDH')
plt.plot(dh2_est[:, 0], dh2_est[:, 1], 'm-.', label='LEDH')
plt.plot(pfpf_edh_est[:, 0], pfpf_edh_est[:, 1], color='red', label='PFPF-EDH')
plt.plot(pfpf_ledh_est[:, 0], pfpf_ledh_est[:, 1], color='orange', label='PFPF-LEDH')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Trajectory Comparison (Range-Bearing)')
plt.legend(); plt.grid(True); plt.axis('equal')
plt.savefig("./figures/range_bearing_trajectory.pdf", dpi=150)
plt.show()

# Position error over time
plt.figure(figsize=(10, 6))
t = np.arange(T)
err_ekf = np.linalg.norm(x_true[:, :2] - ekf_means[:, :2], axis=1)
err_ukf = np.linalg.norm(x_true[:, :2] - ukf_means[:, :2], axis=1)
err_dh1 = np.linalg.norm(x_true[:, :2] - dh1_est[:, :2], axis=1)
err_dh2 = np.linalg.norm(x_true[:, :2] - dh2_est[:, :2], axis=1)
err_pfpf_edh = np.linalg.norm(x_true[:, :2] - pfpf_edh_est[:, :2], axis=1)
err_pfpf_ledh = np.linalg.norm(x_true[:, :2] - pfpf_ledh_est[:, :2], axis=1)

plt.plot(t, err_ekf, 'b-', label='EKF')
plt.plot(t, err_ukf, 'g-', label='UKF')
plt.plot(t, err_dh1, 'c-', label='EDH')
plt.plot(t, err_dh2, 'm-', label='LEDH')
plt.plot(t, err_pfpf_edh, 'r-', label='PFPF-EDH')
plt.plot(t, err_pfpf_ledh, color='orange', label='PFPF-LEDH')
plt.xlabel('Time step'); plt.ylabel('Position error (m)')
plt.title('Position Error Over Time')
plt.legend(); plt.grid(True)
plt.savefig("./figures/range_bearing_error.pdf", dpi=150)
plt.show()

methods = ['EKF', 'UKF', 'EDH', 'LEDH', 'PFPF-EDH', 'PFPF-LEDH']
rmse_vals = [
    np.sqrt(np.mean((x_true[:,pos_idx] - ekf_means[:,pos_idx])**2)),
    np.sqrt(np.mean((x_true[:,pos_idx] - ukf_means[:,pos_idx])**2)),
    np.sqrt(np.mean((x_true[:,pos_idx] - dh1_est[:,pos_idx])**2)),
    np.sqrt(np.mean((x_true[:,pos_idx] - dh2_est[:,pos_idx])**2)),
    np.sqrt(np.mean((x_true[:,pos_idx] - pfpf_edh_est[:,pos_idx])**2)),
    np.sqrt(np.mean((x_true[:,pos_idx] - pfpf_ledh_est[:,pos_idx])**2))
]

# RMSE Bar Chart
plt.figure(figsize=(10, 5))
bars = plt.bar(methods, rmse_vals, color=['blue', 'green', 'cyan', 'magenta', 'red', 'orange'])
plt.ylabel('Position RMSE (m)')
plt.title('Position RMSE Comparison')
plt.xticks(rotation=30)
plt.grid(axis='y')
for bar, rmse in zip(bars, rmse_vals):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{rmse:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig("./figures/range_bearing_rmse.pdf", dpi=150)
plt.show()