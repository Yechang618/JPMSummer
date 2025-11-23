"""
Replication of Section V-A: Multi-Target Acoustic Tracking

This script simulates the multi-target acoustic tracking scenario described
in the prompt and runs a baseline Bootstrap Particle Filter (BPF) with
resampling and ESS tracking. It computes OMAT (p=1) error between the true
target positions and the estimated target positions (from particle means),
records ESS per time step, and measures execution time per step.

Usage:
    python scripts/exp_section_VA_acoustic_tracking.py

Outputs:
    - Prints per-step OMAT, ESS and timings
    - Saves simple plots into `figures/` (OMAT over time, ESS over time)

Note: This is a single-file experiment runner intended for reproducible
research; it intentionally keeps dependencies small. If `scipy` is present
we use the Hungarian algorithm for OMAT; otherwise we fallback to brute-force
permutations (C small => 4 targets => 24 permutations manageable).
"""

import os
import time
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
import os as _os
# Ensure the repository root is on sys.path so `from src...` imports work
# when running this script directly (without PYTHONPATH set).
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
_repo_root = _os.path.dirname(_this_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# import EDH flow implementation
from src.models.EDH import EDH

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---------------------- Simulation parameters ----------------------
C = 4          # number of targets
T = 30         # time steps
Ns = 25        # sensors (5x5 grid)
Np = 500       # number of particles for PF baseline
Psi = 10.0
d0 = 0.1
sigma_w = 0.1  # measurement noise std (note prompt used variance 0.01 -> std 0.1)

# Per-target state dimension
state_per_target = 4
state_dim = C * state_per_target

# Motion model (constant velocity)
F_4 = np.array([[1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]], dtype=float)

# Block-diagonal state transition for all targets
F = np.kron(np.eye(C), F_4)

V_true_block = (1.0 / 20.0) * np.array([
    [1.0/3.0, 0.0, 0.5, 0.0],
    [0.0, 1.0/3.0, 0.0, 0.5],
    [0.5, 0.0, 1.0, 0.0],
    [0.0, 0.5, 0.0, 1.0],
], dtype=float)

# full process noise covariance (block diagonal)
Q_true = np.kron(np.eye(C), V_true_block)

# Initial states for each target as in the prompt
x0_true_targets = np.array([
    [12.0, 6.0, 0.001, 0.001],
    [32.0, 32.0, -0.001, -0.005],
    [20.0, 13.0, -0.1, 0.01],
    [15.0, 35.0, 0.002, 0.002],
], dtype=float)  # shape (C,4)

# Flatten to full state vector of length 4*C
x0_true = x0_true_targets.flatten()

# Sensor grid (5x5 over [0,40]x[0,40])
grid = np.linspace(0.0, 40.0, 5)
sensors = np.array([[x, y] for x in grid for y in grid])  # shape (25,2)

# ---------------------- Helper functions ----------------------

def simulate_trajectories(F_block, Q_block, x0_targets, T, rng=None):
    """Simulate C target trajectories for T time steps.
    Returns trajectories shaped (C, T, 4) and flattened full-states list length T (each is length 4C)
    """
    if rng is None:
        rng = np.random.default_rng()
    C = x0_targets.shape[0]
    trajectories = np.zeros((C, T, state_per_target), dtype=float)
    # initialize
    trajectories[:, 0, :] = x0_targets
    for t in range(1, T):
        for c in range(C):
            prev = trajectories[c, t-1]
            # apply local 4x4 motion + process noise
            new = F_4 @ prev + rng.multivariate_normal(mean=np.zeros(state_per_target), cov=V_true_block)
            trajectories[c, t, :] = new
    # produce flattened states per time step (length 4C)
    flat_states = [trajectories[:, t, :].flatten() for t in range(T)]
    return trajectories, flat_states


def simulate_measurements(flat_state, sensors, Psi=10.0, d0=0.1, sigma_w=0.1, rng=None):
    """Given flattened full state (length 4C) compute sensor amplitudes and add Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()
    C = int(len(flat_state) // state_per_target)
    positions = flat_state.reshape((C, state_per_target))[:, :2]  # (C,2)
    amps = np.zeros(len(sensors), dtype=float)
    for pos in positions:
        # squared distance to all sensors
        d2 = np.sum((sensors - pos) ** 2, axis=1)
        amps += Psi / (d2 + d0)
    noisy = amps + rng.normal(0.0, sigma_w, size=amps.shape)
    return noisy


def compute_log_likelihood(z_obs, z_pred, sigma_w):
    """Compute log likelihood of observation under independent Gaussian noise sigma_w.
    z_obs and z_pred are arrays of shape (Ns,)
    """
    resid = z_obs - z_pred
    var = sigma_w ** 2
    # ignoring constant term when comparing weights is fine, but we will include full log-prob
    ll = -0.5 * np.sum((resid ** 2) / var + np.log(2 * np.pi * var))
    return ll


def predict_particle_states(particles, F_block, Q_block, rng):
    """Propagate all particles one step: particles shape (Np, 4C)
    Returns new_particles shape (Np, 4C)
    """
    Np = particles.shape[0]
    # linear propagation x' = F x + noise ~ N(Fx, Q)
    mean = particles @ F_block.T  # but F_block is (4C x 4C); for clarity use matrix mult
    # Actually use np.dot on each particle
    new_particles = np.zeros_like(particles)
    for i in range(Np):
        new_particles[i] = F_block @ particles[i] + rng.multivariate_normal(np.zeros(state_dim), Q_block)
    return new_particles


def measurement_from_particle(flat_state, sensors, Psi=10.0, d0=0.1):
    """Predict measurement (noise-free) from a particle state vector (length 4C)."""
    C = int(len(flat_state) // state_per_target)
    positions = flat_state.reshape((C, state_per_target))[:, :2]
    amps = np.zeros(len(sensors), dtype=float)
    for pos in positions:
        d2 = np.sum((sensors - pos) ** 2, axis=1)
        amps += Psi / (d2 + d0)
    return amps


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    w = w / (np.sum(w) + 1e-300)
    return 1.0 / np.sum(w ** 2)


def systematic_resample(weights, rng):
    """Systematic resampling. weights sum to 1."""
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumw = np.cumsum(weights)
    inds = np.zeros(N, dtype=int)
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumw[j]:
            inds[i] = j
            i += 1
        else:
            j += 1
    return inds


def compute_omat(true_positions, est_positions, p=1):
    """Compute OMAT (p=1) between two sets of C 2-D positions (arrays shape (C,2)).
    Uses Hungarian if scipy available; otherwise brute-force permutations (C small).
    """
    C = true_positions.shape[0]
    cost = np.linalg.norm(true_positions[:, None, :] - est_positions[None, :, :], axis=2) ** p  # (C,C)
    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(cost)
        val = np.sum(cost[row_ind, col_ind])
        return (val / float(C)) ** (1.0 / p)
    else:
        # brute-force (C <= 4 is manageable)
        best = np.inf
        for perm in itertools.permutations(range(C)):
            s = sum(cost[i, perm[i]] for i in range(C))
            if s < best:
                best = s
        return (best / float(C)) ** (1.0 / p)

# ---------------------- Main experiment runner ----------------------

def run_experiment(seed=1, save_figures=True, show_progress=True):
    rng = np.random.default_rng(seed)

    # simulate trajectories
    trajectories, flat_states = simulate_trajectories(F_4, V_true_block, x0_true_targets, T, rng=rng)

    # generate observations for each t
    measurements = [simulate_measurements(flat_states[t], sensors, Psi=Psi, d0=d0, sigma_w=sigma_w, rng=rng) for t in range(T)]

    # Initialize particles: sample around initial true states with large uncertainty
    pos_std = 10.0
    vel_std = 1.0
    init_cov_per = np.diag([pos_std ** 2, pos_std ** 2, vel_std ** 2, vel_std ** 2])
    init_cov = np.kron(np.eye(C), init_cov_per)
    particles = rng.multivariate_normal(mean=x0_true, cov=init_cov, size=Np)  # shape (Np, 4C)
    # keep a copy of initial particles for use by EDH run (BPF will mutate `particles`)
    init_particles = particles.copy()
    weights = np.ones(Np, dtype=float) / float(Np)

    # histories
    ess_hist = []
    omat_hist = []
    time_hist = []

    # run filtering
    for t in range(T):
        z_t = measurements[t]
        t0 = time.perf_counter()

        # predict step: propagate each particle under block F and Q_true
        new_particles = np.zeros_like(particles)
        for i in range(Np):
            new_particles[i] = F @ particles[i] + rng.multivariate_normal(np.zeros(state_dim), Q_true)
        particles = new_particles

        # compute weights via likelihood p(z_t | particle)
        logw = np.zeros(Np, dtype=float)
        for i in range(Np):
            z_pred = measurement_from_particle(particles[i], sensors, Psi=Psi, d0=d0)
            logw[i] = compute_log_likelihood(z_t, z_pred, sigma_w)
        # stabilize and normalize
        maxlog = np.max(logw)
        w_unnorm = np.exp(logw - maxlog)
        weights = w_unnorm / np.sum(w_unnorm)

        ess = effective_sample_size(weights)
        ess_hist.append(ess)

        # estimate: use weighted mean state, then extract target positions
        mean_state = np.sum(particles * weights[:, None], axis=0)
        est_positions = mean_state.reshape((C, state_per_target))[:, :2]
        true_positions = flat_states[t].reshape((C, state_per_target))[:, :2]
        omat = compute_omat(true_positions, est_positions, p=1)
        omat_hist.append(omat)

        # resample if ESS below threshold
        if ess < (Np / 2.0):
            inds = systematic_resample(weights, rng)
            particles = particles[inds]
            weights = np.ones(Np, dtype=float) / float(Np)

        t1 = time.perf_counter()
        time_hist.append((t1 - t0) / 1.0)

        if show_progress:
            print(f"t={t+1}/{T}  OMAT={omat:.3f}  ESS={ess:.1f}  step_time={time_hist[-1]:.4f}s")

    # --- EDH particle-flow run using src/models/EDH.py ---
    # Build TensorFlow-compatible dynamics and observation functions
    dtype = tf.float64

    F_mat = tf.convert_to_tensor(F, dtype=dtype)
    Q_tf = tf.convert_to_tensor(Q_true, dtype=dtype)
    R_tf = tf.eye(Ns, dtype=dtype) * (sigma_w ** 2)

    def make_f_tf(Fmat):
        def f(x):
            x_tf = tf.convert_to_tensor(x, dtype=dtype)
            # handle vector (state_dim,) and batch (N, state_dim)
            if len(x_tf.shape) == 1:
                x_col = tf.reshape(x_tf, [-1, 1])
                y = tf.matmul(Fmat, x_col)
                return tf.reshape(y, [-1])
            else:
                # x is (N, state_dim) -> want (N, state_dim) after multiply by F^T
                return tf.matmul(x_tf, tf.transpose(Fmat))
        return f

    def make_h_tf(sensors, Psi=10.0, d0=0.1):
        sensors_tf = tf.convert_to_tensor(sensors, dtype=dtype)
        def h(x):
            x_tf = tf.convert_to_tensor(x, dtype=dtype)
            # x can be (state_dim,) or (N, state_dim)
            if len(x_tf.shape) == 1:
                positions = tf.reshape(x_tf, [ -1, state_per_target ])[:, :2]  # (C,2)
                # compute sensor amplitudes
                # sensors_tf shape (Ns,2), positions (C,2) -> compute pairwise d2 (Ns, C)
                diff = tf.expand_dims(sensors_tf, 1) - tf.expand_dims(positions, 0)  # (Ns, C, 2)
                d2 = tf.reduce_sum(diff ** 2, axis=-1)  # (Ns, C)
                inv = Psi / (d2 + d0)
                amps = tf.reduce_sum(inv, axis=1)
                return tf.reshape(amps, [-1])
            else:
                # batch case
                # x_tf shape (N, state_dim) -> positions (N, C, 2)
                Nbatch = tf.shape(x_tf)[0]
                resh = tf.reshape(x_tf, (Nbatch, -1, state_per_target))[:, :, :2]  # (N, C, 2)
                # sensors_tf (Ns,2) -> expand to (1, Ns, 2)
                s = tf.expand_dims(sensors_tf, 0)  # (1, Ns, 2)
                # compute d2 per batch: (N, Ns, C)
                # resh: (N, C, 2) -> expand sensors axis
                resh_exp = tf.expand_dims(resh, 1)  # (N,1,C,2)
                s_exp = tf.expand_dims(s, 2)        # (1,Ns,1,2)
                diff = s_exp - resh_exp  # (N, Ns, C, 2)
                d2 = tf.reduce_sum(diff ** 2, axis=-1)  # (N, Ns, C)
                inv = Psi / (d2 + d0)
                amps = tf.reduce_sum(inv, axis=-1)  # (N, Ns)
                return amps
        return h

    f_tf = make_f_tf(F_mat)
    h_tf = make_h_tf(sensors, Psi=Psi, d0=d0)

    # Create EDH filter instance
    edh = EDH(num_particles=Np, f=f_tf, h=h_tf, state_dim=state_dim, observation_dim=Ns, Q=Q_true, R=R_tf, dtype=dtype)

    # initialize EDH particles from the same init_particles distribution
    edh.initialize(particles=tf.convert_to_tensor(init_particles, dtype=dtype))

    omat_edh_hist = []
    time_edh_hist = []

    for t in range(T):
        z_t = measurements[t]
        t0 = time.perf_counter()
        # convert observation to TF tensor
        y_tf = tf.convert_to_tensor(z_t, dtype=dtype)
        # EDH.update performs propagation + particle flow update
        edh.update(y_tf)
        t1 = time.perf_counter()

        est_state_tf = edh.get_state_estimate()
        est_state = np.asarray(est_state_tf.numpy())
        est_positions = est_state.reshape((C, state_per_target))[:, :2]
        true_positions = flat_states[t].reshape((C, state_per_target))[:, :2]
        omat_e = compute_omat(true_positions, est_positions, p=1)
        omat_edh_hist.append(omat_e)
        time_edh_hist.append(t1 - t0)

        print(f"EDH t={t+1}/{T}  OMAT={omat_e:.3f}  step_time={(t1-t0):.4f}s")

    # EDH summary
    print('\nEDH Experiment summary:')
    print(f"Mean OMAT (EDH): {np.mean(omat_edh_hist):.4f}, Median OMAT: {np.median(omat_edh_hist):.4f}")
    print(f"Mean step time (EDH): {np.mean(time_edh_hist):.4f}s, Total time: {np.sum(time_edh_hist):.4f}s")

    # save EDH plots
    if save_figures:
        plt.figure()
        plt.plot(omat_edh_hist, marker='o')
        plt.xlabel('Time step')
        plt.ylabel('OMAT (p=1)')
        plt.title('OMAT over time (EDH particle flow)')
        plt.grid(True)
        plt.savefig(os.path.join('figures', 'expV_A_OMAT_edh.pdf'), bbox_inches='tight')

        plt.figure()
        plt.plot(time_edh_hist, marker='o')
        plt.xlabel('Time step')
        plt.ylabel('Step time (s)')
        plt.title('Per-step time (EDH)')
        plt.grid(True)
        plt.savefig(os.path.join('figures', 'expV_A_time_edh.pdf'), bbox_inches='tight')

    # summary
    print('\nExperiment summary:')
    print(f"Mean OMAT: {np.mean(omat_hist):.4f}, Median OMAT: {np.median(omat_hist):.4f}")
    print(f"Mean ESS: {np.mean(ess_hist):.1f}, Min ESS: {np.min(ess_hist):.1f}")
    print(f"Mean step time: {np.mean(time_hist):.4f}s, Total time: {np.sum(time_hist):.4f}s")

    # save plots
    if save_figures:
        os.makedirs('figures', exist_ok=True)
        plt.figure()
        plt.plot(omat_hist, marker='o')
        plt.xlabel('Time step')
        plt.ylabel('OMAT (p=1)')
        plt.title('OMAT over time (BPF baseline)')
        plt.grid(True)
        plt.savefig(os.path.join('figures', 'expV_A_OMAT_bpf.pdf'), bbox_inches='tight')

        plt.figure()
        plt.plot(ess_hist, marker='o')
        plt.hlines(Np/2.0, 0, T-1, colors='r', linestyles='--', label='resample threshold (Np/2)')
        plt.xlabel('Time step')
        plt.ylabel('ESS')
        plt.title('ESS over time (BPF baseline)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join('figures', 'expV_A_ESS_bpf.pdf'), bbox_inches='tight')

        plt.figure(figsize=(6,6))
        # show true and estimated positions at final time
        plt.scatter(true_positions[:,0], true_positions[:,1], c='k', marker='x', label='true')
        plt.scatter(est_positions[:,0], est_positions[:,1], c='r', marker='o', label='estimated (mean)')
        plt.legend()
        plt.title('Final true vs estimated target positions')
        plt.savefig(os.path.join('figures', 'expV_A_final_positions.pdf'), bbox_inches='tight')

    return {
        'omat_hist': np.array(omat_hist),
        'ess_hist': np.array(ess_hist),
        'time_hist': np.array(time_hist),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--np', dest='np', type=int, default=Np)
    parser.add_argument('--T', type=int, default=T)
    args = parser.parse_args()
    # allow overriding Np and T easily
    Np = args.np
    T = args.T
    # call main runner
    results = run_experiment(seed=args.seed)
    print('Done. Figures saved in `figures/` (if enabled).')
