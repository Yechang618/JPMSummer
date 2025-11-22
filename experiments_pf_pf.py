"""
Experiments for PF-PF, EDH, LEDH and comparisons.

This script contains three experiments described in the project notes:
 1) Multi-Target Acoustic Tracking
 2) Linear Gaussian Spatial Sensor Network
 3) Skewed-t Dynamic Model with Count Measurements

Each experiment builds a simple simulation, runs a small set of filters
(from `src.models`) and prints basic performance metrics (OMAT, MSE etc.).

Notes:
- The scripts aim to be runnable in this repository environment and to
  demonstrate the experimental setups. Parameters are chosen to be
  computationally modest while reflecting the structure described in the
  experiment descriptions.

Run:
    python experiments_pf_pf.py

"""
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math
import numpy as np
import tensorflow as tf

from scipy.optimize import linear_sum_assignment

# Import filters from project
from src.models.PF_PF import ParticleFilterWithInvertibleFlow
from src.models.ParticleFilter import ParticleFilter as BootstrapPF
from src.models.KalmanFilter import KalmanFilter
from src.models.UnscentedKalmanFilter import UnscentedKalmanFilter
from models.EDH_1 import EDH
from src.models.LEDH import LEDH
import time


def omat_distance(true_positions, est_positions):
    """Compute OMAT (Earth-mover/assignment) distance p=1 between two
    unordered point sets of same cardinality.
    true_positions, est_positions: arrays shape (C, 2)
    Returns: average matched L1 distance (here L2 Euclidean used as cost but p=1 sum).
    """
    C = true_positions.shape[0]
    cost = np.linalg.norm(true_positions[:, None, :] - est_positions[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    total = cost[row_ind, col_ind].sum()
    return total / float(C)


def experiment_acoustic_tracking(T=20, seed=1):
    """Multi-target acoustic tracking experiment.
    - C targets in 40x40 region, constant velocity
    - Ns sensors on uniform grid
    - Sensor measurement: sum of attenuated amplitudes
    """
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    # Parameters
    C = 4
    region = 40.0
    state_per_target = 4
    d = C * state_per_target
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    Psi = 10.0
    d0 = 0.1
    Ns = 25
    sigma_w = math.sqrt(0.01)  # 0.1

    # Sensor grid positions
    side = int(np.sqrt(Ns))
    xs = np.linspace(0, region, side)
    ys = np.linspace(0, region, side)
    R_s = np.array([[x,y] for x in xs for y in ys])

    # Initialize targets randomly in region with small velocities
    targets = []
    for c in range(C):
        x0 = rng.uniform(0.2*region, 0.8*region)
        y0 = rng.uniform(0.2*region, 0.8*region)
        vx0 = rng.normal(scale=0.5)
        vy0 = rng.normal(scale=0.5)
        targets.append(np.array([x0,y0,vx0,vy0]))
    targets = np.stack(targets, axis=0)  # shape (C,4)

    # Process noise (small)
    Q_single = np.diag([0.01,0.01,0.05,0.05])

    # Simulation storage
    true_states = np.zeros((T, C, 4))
    observations = np.zeros((T, Ns))

    x_c = targets.copy()
    for t in range(T):
        # propagate
        for c in range(C):
            noise = rng.multivariate_normal(np.zeros(4), Q_single)
            x_c[c] = F @ x_c[c] + noise
            true_states[t,c] = x_c[c]
        # build observation: sensor s measures sum over c of Psi/(dist^2 + d0)
        z = np.zeros(Ns)
        for s in range(Ns):
            pos_s = R_s[s]
            val = 0.0
            for c in range(C):
                pos_c = x_c[c, :2]
                dist2 = np.sum((pos_c - pos_s)**2)
                val += Psi / (dist2 + d0)
            z[s] = val + rng.normal(scale=sigma_w)
        observations[t] = z

    # Define observation function for filters: maps full state vector (d,) to sensor measurements (Ns,)
    def h_tf(x_tf: tf.Tensor) -> tf.Tensor:
        # x_tf is shape (d,)
        x_tf = tf.reshape(x_tf, [C, 4])  # (C,4)
        pos = x_tf[:, :2]  # (C,2)
        sensors = tf.constant(R_s, dtype=tf.float64)
        Psi_tf = tf.constant(Psi, dtype=tf.float64)
        d0_tf = tf.constant(d0, dtype=tf.float64)
        # compute distances squared: (C,1,2) - (1,Ns,2)
        dif = tf.expand_dims(pos, axis=1) - tf.expand_dims(sensors, axis=0)  # (C, Ns, 2)
        dist2 = tf.reduce_sum(dif**2, axis=-1)  # (C, Ns)
        contrib = Psi_tf / (dist2 + d0_tf)  # (C, Ns)
        summed = tf.reduce_sum(contrib, axis=0)  # (Ns,)
        return summed

    # Process transition for full state vector
    def f_tf(x_tf: tf.Tensor) -> tf.Tensor:
        x_tf = tf.reshape(x_tf, [C, 4])
        def step_one(xi):
            xi = tf.reshape(xi, [4,])
            return tf.linalg.matvec(tf.constant(F, dtype=tf.float64), xi)
        nexts = tf.map_fn(step_one, x_tf)
        return tf.reshape(nexts, [d,])

    # Filters
    Np = 500
    # Bootstrap PF using ParticleFilter (requires per-particle transition/observation functions)
    pf_bpf = BootstrapPF(transition_fn=f_tf, observation_fn=h_tf, Q=np.kron(np.eye(C), Q_single), R=np.eye(Ns)*0.01, num_particles=Np, initial_mean=np.zeros(d), initial_cov=np.eye(d)*50.0, dtype=tf.float64, seed=2)

    # PF-PF global (EDH-like) and local (LEDH)
    pf_pf_edh = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=3, use_local_flow=False)
    pf_pf_edh.initialize(mean=np.zeros(d), cov=np.eye(d)*50.0)
    pf_pf_ledh = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=4, use_local_flow=True)
    pf_pf_ledh.initialize(mean=np.zeros(d), cov=np.eye(d)*50.0)

    # Process noise and R for PF_PF
    R_pf = np.eye(Ns) * 0.01

    omat_edh = []
    omat_ledh = []
    omat_bpf = []

    for t in range(T):
        y = observations[t]
        # Step bootstrap PF
        pf_bpf.predict()
        pf_bpf.update(y)
        mean_bpf, cov_bpf = pf_bpf.estimate()
        mean_bpf_np = mean_bpf.numpy() if hasattr(mean_bpf, 'numpy') else np.array(mean_bpf)
        est_pos_bpf = mean_bpf_np.reshape(C,4)[:, :2]
        # PF-PF EDH: linearize observation at prior mean and pass H
        # compute prior mean from particles
        prior_mean_edh = np.mean(pf_pf_edh.particles.numpy(), axis=0)
        x0_tf = tf.convert_to_tensor(prior_mean_edh, dtype=tf.float64)
        with tf.GradientTape() as tape_edh:
            tape_edh.watch(x0_tf)
            h0 = h_tf(x0_tf)
        J_lin = tape_edh.jacobian(h0, x0_tf)
        H_lin = np.reshape(J_lin.numpy(), (Ns, d))
        logz_edh = pf_pf_edh.update(y=y, H=H_lin, R=R_pf, h_func=None)
        mean_edh = np.mean(pf_pf_edh.particles.numpy(), axis=0)
        est_pos_edh = mean_edh.reshape(C,4)[:, :2]
        # PF-PF LEDH (use h_func)
        # provide a dummy H (will not be used because h_func is provided)
        logz_ledh = pf_pf_ledh.update(y=y, H=np.zeros((Ns, d)), R=R_pf, h_func= h_tf)
        mean_ledh = tf.reduce_mean(pf_pf_ledh.particles, axis=0).numpy()
        est_pos_ledh = mean_ledh.reshape(C,4)[:, :2]

        true_pos = true_states[t,:, :2]
        omat_edh.append(omat_distance(true_pos, est_pos_edh))
        omat_ledh.append(omat_distance(true_pos, est_pos_ledh))
        omat_bpf.append(omat_distance(true_pos, est_pos_bpf))

    print("Acoustic tracking results (average OMAT over T steps):")
    print("BPF OMAT:", np.mean(omat_bpf))
    print("PF-PF (EDH) OMAT:", np.mean(omat_edh))
    print("PF-PF (LEDH) OMAT:", np.mean(omat_ledh))


def experiment_linear_gaussian(T=20, seed=2):
    """Linear Gaussian Spatial Sensor Network experiment.
    State is x in R^d where d sensors on sqrt(d)x sqrt(d) grid.
    Dynamics: x_k = alpha x_{k-1} + v_k, measurements z = x + w
    """
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    d = 64
    side = int(np.sqrt(d))
    coords = np.array([[i, j] for i in range(side) for j in range(side)], dtype=float)

    alpha = 0.9
    alpha0 = 3.0
    alpha1 = 0.01
    beta = 20.0

    Np = 200
    # Build spatial covariance Sigma
    Sigma = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            dist2 = np.sum((coords[i] - coords[j])**2)
            Sigma[i,j] = alpha0 * np.exp(-dist2 / beta) + (alpha1 if i==j else 0.0)

    Q = Sigma * 0.1  # process noise scaled

    T = T
    # initial state zero
    x = np.zeros(d)

    # Filters: Kalman (optimal) and PF-PF(EDH)
    A = alpha * np.eye(d)
    
    def f_fn(x_tf: tf.Tensor) -> tf.Tensor:
        return tf.cast(alpha, tf.float64) * x_tf

    def h_fn(x_tf: tf.Tensor) -> tf.Tensor:
        return x_tf
    
    sigma_zs = [2.0, 1.0, 0.5]
    results = {}

    for sigma_z in sigma_zs:
        R = (sigma_z**2) * np.eye(d)
        # simulate T steps
        xs = np.zeros((T, d))
        zs = np.zeros((T, d))
        xk = x.copy()
        for t in range(T):
            v = rng.multivariate_normal(np.zeros(d), Q)
            xk = alpha * xk + v
            w = rng.multivariate_normal(np.zeros(d), R)
            zt = xk + w
            xs[t] = xk
            zs[t] = zt

        alg_stats = {}

        # 1) Kalman Filter (KF) - optimal
        t0 = time.perf_counter()
        kf = KalmanFilter(transition_matrix=A, 
                          observation_matrix=np.eye(d), 
                          transition_cov=Q, 
                          observation_cov=R, 
                          initial_mean=np.zeros(d), 
                          initial_cov=np.eye(d)*1.0, 
                          verbose_or_not=True)
        fm, Fc, ll = kf.filter(zs)
        t1 = time.perf_counter()
        kf_est = fm.numpy()  # (T,d)
        mse_kf = np.mean((kf_est - xs)**2)
        alg_stats['KF'] = {'avg_mse': float(mse_kf), 'avg_ess': None, 'time_s': t1 - t0}

        # 2) Unscented Kalman Filter (UKF)
        t0 = time.perf_counter()

        ukf = UnscentedKalmanFilter(f=f_fn, 
                                    h=h_fn, 
                                    Q=tf.convert_to_tensor(Q, dtype=tf.float64), 
                                    R=tf.convert_to_tensor(R, dtype=tf.float64), 
                                    initial_mean=tf.convert_to_tensor(np.zeros(d), dtype=tf.float64), 
                                    initial_cov=tf.convert_to_tensor(np.eye(d), dtype=tf.float64),
                                    verbose_or_not=True)
        fm_u, Fc_u, ll_u = ukf.filter(zs)
        t1 = time.perf_counter()
        ukf_est = fm_u.numpy()
        mse_ukf = np.mean((ukf_est - xs)**2)
        alg_stats['UKF'] = {'avg_mse': float(mse_ukf), 'avg_ess': None, 'time_s': t1 - t0}

        # 3) Bootstrap Particle Filter (BF)
        # Np = 200
        t0 = time.perf_counter()
        bf = BootstrapPF(transition_fn=f_fn, 
                         observation_fn=h_fn, 
                         Q=Q, 
                         R=R, 
                         num_particles=Np, 
                         initial_mean=np.zeros(d), 
                         initial_cov=np.eye(d)*1.0, 
                         dtype=tf.float64, 
                         seed=7,
                         verbose_or_not=True)
        mse_list = []
        ess_list = []
        for t in range(T):
            bf.predict()
            bf.update(zs[t])
            mean_bf, cov_bf = bf.estimate()
            mean_bf_np = mean_bf.numpy() if hasattr(mean_bf, 'numpy') else np.array(mean_bf)
            mse_list.append(np.mean((mean_bf_np - xs[t])**2))
            # ESS: use last_ess if available
            try:
                ess_list.append(float(bf.last_ess.numpy()))
            except Exception:
                ess_list.append(float(bf.ess_history[-1]) if len(bf.ess_history) else float(Np))
        t1 = time.perf_counter()
        alg_stats['BF'] = {'avg_mse': float(np.mean(mse_list)), 'avg_ess': float(np.mean(ess_list)), 'time_s': t1 - t0}

        # 4) PF-PF (global invertible EDH / IEDH)
        t0 = time.perf_counter()
        pfpf_iedh = ParticleFilterWithInvertibleFlow(num_particles=Np, 
                                                     state_dim=d, 
                                                     seed=5, 
                                                     use_local_flow=False, 
                                                     verbose_or_not=True)
        pfpf_iedh.initialize(mean=np.zeros(d), cov=np.eye(d)*1.0)
        mse_list = []
        ess_list = []
        for t in range(T):
            # Predict step for PF-PF (apply linear dynamics and process noise)
            pfpf_iedh.predict(dynamics_fn = f_fn, process_noise_cov=Q)
            pfpf_iedh.update(y=zs[t], H=np.eye(d), R=R)
            mean = np.mean(pfpf_iedh.particles, axis=0)
            mse_list.append(np.mean((mean - xs[t])**2))
            ess_list.append(float(pfpf_iedh.ess_history[-1]) if len(pfpf_iedh.ess_history) else float(Np))
        t1 = time.perf_counter()
        alg_stats['PF-PF(IEDH)'] = {'avg_mse': float(np.mean(mse_list)), 'avg_ess': float(np.mean(ess_list)), 'time_s': t1 - t0}

        # 5) PF-PF (local LEDH / ILEDH)
        t0 = time.perf_counter()
        pfpf_iledh = ParticleFilterWithInvertibleFlow(num_particles=Np, 
                                                      state_dim=d, 
                                                      seed=6, 
                                                      use_local_flow=True, 
                                                      verbose_or_not=True)
        pfpf_iledh.initialize(mean=np.zeros(d), cov=np.eye(d)*1.0)
        mse_list = []
        ess_list = []
        for t in range(T):
            # Predict step for local PF-PF
            pfpf_iledh.predict(dynamics_fn = f_fn, process_noise_cov=Q)
            pfpf_iledh.update(y=zs[t], H=np.eye(d), R=R)
            mean = np.mean(pfpf_iledh.particles, axis=0)
            mse_list.append(np.mean((mean - xs[t])**2))
            ess_list.append(float(pfpf_iledh.ess_history[-1]) if len(pfpf_iledh.ess_history) else float(Np))
        t1 = time.perf_counter()
        alg_stats['PF-PF(ILEDH)'] = {'avg_mse': float(np.mean(mse_list)), 'avg_ess': float(np.mean(ess_list)), 'time_s': t1 - t0}

        # 6) EDH (standalone deterministic affine flow)
        t0 = time.perf_counter()
        edh = EDH(num_particles=Np, state_dim=d, dtype=tf.float64)
        # initialize with samples from prior (mean zero, cov I)
        init_particles = rng.multivariate_normal(np.zeros(d), np.eye(d) * 1.0, size=Np)
        edh.initialize(particles=tf.convert_to_tensor(init_particles, dtype=tf.float64))
        mse_list = []
        for t in range(T):
            edh.predict(lambda x: tf.cast(alpha, tf.float64) * x, process_noise_cov=Q)
            edh.update(y=zs[t], H=np.eye(d), R=R)
            mean_edh = edh.get_state_estimate().numpy()
            mse_list.append(np.mean((mean_edh - xs[t])**2))
        t1 = time.perf_counter()
        alg_stats['EDH'] = {'avg_mse': float(np.mean(mse_list)), 'avg_ess': None, 'time_s': t1 - t0}

        # 7) LEDH (standalone local flow)
        t0 = time.perf_counter()
        ledh = LEDH(num_particles=Np, state_dim=d, dtype=tf.float64)
        ledh.initialize(particles=tf.convert_to_tensor(init_particles, dtype=tf.float64))
        mse_list = []
        for t in range(T):
            ledh.predict(lambda x: tf.cast(alpha, tf.float64) * x, process_noise_cov=Q)
            # for linear observation H=I we can pass H directly
            ledh.update(y=zs[t], H=np.eye(d), R=R)
            mean_ledh = ledh.get_state_estimate().numpy()
            mse_list.append(np.mean((mean_ledh - xs[t])**2))
        t1 = time.perf_counter()
        alg_stats['LEDH'] = {'avg_mse': float(np.mean(mse_list)), 'avg_ess': None, 'time_s': t1 - t0}

        results[sigma_z] = alg_stats

    # Print results in readable form
    print("Linear Gaussian spatial network results (Avg. MSE, Avg. ESS, Time [s]):")
    for s, stats in results.items():
        print(f"sigma_z={s}:")
        for alg, vals in stats.items():
            avg_mse = vals['avg_mse']
            avg_ess = vals['avg_ess']
            tsec = vals['time_s']
            ess_str = f"{avg_ess:.2f}" if avg_ess is not None else "N/A"
            print(f"  {alg:15s} MSE={avg_mse:.4e}, AvgESS={ess_str}, Time={tsec:.3f}s")


def experiment_skewed_t_counts(T=30, seed=3):
    """Skewed-t dynamic model with Poisson count measurements (simplified).
    We simulate heavy-tailed transitions via multivariate t and Poisson obs.
    """
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    d = 144
    alpha = 0.95

    # State transition: sample from multivariate t approx via scale*normal/sqrt
    df = 5.0
    scale = 0.5
    # covariance for t
    Sigma = np.eye(d) * 0.5

    m1 = 1.0
    m2 = 1.0/3.0

    # initial state
    x = np.zeros(d)

    # simple PF-PF EDH baseline
    Np = 2000
    pfpf = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=6, use_local_flow=False)
    pfpf.initialize(mean=np.zeros(d), cov=np.eye(d)*1.0)

    # We simulate and perform filtering
    mse_list = []
    for t in range(T):
        # sample heavy-tailed increment via student-t
        z = rng.standard_t(df, size=d)
        v = scale * z
        x = alpha * x + v
        lam = m1 * np.exp(m2 * x)
        # sample Poisson counts
        y = rng.poisson(lam)

        # For PF-PF we need an observation mapping from x -> counts rate
        def h_counts_tf(x_tf: tf.Tensor) -> tf.Tensor:
            # returns vector of rates (d,)
            return tf.exp(tf.cast(m2, tf.float64) * x_tf) * tf.cast(m1, tf.float64)

        # Use a diagonal R approximating Poisson variance (variance ~ mean)
        R = np.diag(np.maximum(lam, 1.0))

        # update filter (we pass observation y as real vector, and use h_func for mapping)
        pfpf.update(y=y.astype(float), H=None, R=R, h_func=h_counts_tf)
        mean = np.mean(pfpf.particles, axis=0)
        mse_list.append(np.mean((mean - x)**2))

    print("Skewed-t + count obs experiment: average MSE:", np.mean(mse_list))


def experiment_kernel_flow_smoke(T=5, seed=123, Np=200, d=16):
    """Small smoke experiment comparing KernelScalar and KernelMatrix flows.

    Runs a simple linear-Gaussian dynamics with identity observation and
    compares PF-PF configured with kernel-scalar (global) and kernel-matrix (local) flows.
    """
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    # Simple dynamics: x_k = 0.9 x_{k-1} + v
    alpha = 0.9
    A = alpha * np.eye(d)
    def f_fn(x_tf: tf.Tensor) -> tf.Tensor:
        return tf.cast(alpha, tf.float64) * x_tf
    def h_fn(x_tf: tf.Tensor) -> tf.Tensor:
        return x_tf

    # Covariances
    Q = np.eye(d) * 0.1
    R = np.eye(d) * 1.0

    # simulate
    xs = np.zeros((T, d))
    zs = np.zeros((T, d))
    xk = np.zeros(d)
    for t in range(T):
        v = rng.multivariate_normal(np.zeros(d), Q)
        xk = alpha * xk + v
        w = rng.multivariate_normal(np.zeros(d), R)
        zs[t] = xk + w
        xs[t] = xk

    # PF-PF with kernel-scalar (global)
    pf_kernel_scalar = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=11, use_local_flow=False, verbose_or_not=True, flow_type='kernel-scalar')
    pf_kernel_scalar.initialize(mean=np.zeros(d), cov=np.eye(d)*1.0)

    # PF-PF with kernel-matrix (local)
    pf_kernel_matrix = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=12, use_local_flow=True, verbose_or_not=True, flow_type='kernel-matrix')
    pf_kernel_matrix.initialize(mean=np.zeros(d), cov=np.eye(d)*1.0)

    stats = {'kernel-scalar': {'mse': [], 'ess': []}, 'kernel-matrix': {'mse': [], 'ess': []}}

    for t in range(T):
        zt = zs[t]
        # predict
        pf_kernel_scalar.predict(dynamics_fn=f_fn, process_noise_cov=Q)
        pf_kernel_matrix.predict(dynamics_fn=f_fn, process_noise_cov=Q)

        pf_kernel_scalar.update(y=zt, H=np.eye(d), R=R)
        pf_kernel_matrix.update(y=zt, H=np.eye(d), R=R)

        m1 = np.mean(pf_kernel_scalar.particles.numpy(), axis=0)
        m2 = np.mean(pf_kernel_matrix.particles.numpy(), axis=0)
        stats['kernel-scalar']['mse'].append(np.mean((m1 - xs[t])**2))
        stats['kernel-matrix']['mse'].append(np.mean((m2 - xs[t])**2))
        stats['kernel-scalar']['ess'].append(float(pf_kernel_scalar.ess_history[-1]))
        stats['kernel-matrix']['ess'].append(float(pf_kernel_matrix.ess_history[-1]))

    print("Kernel-flow smoke experiment results:")
    for k in stats:
        print(f"  {k}: avg MSE={np.mean(stats[k]['mse']):.4e}, avg ESS={np.mean(stats[k]['ess']):.1f}")


def figure2_kernel_divergence(savepath='results/figure2_kernel_divergence.png'):
    """Reproduce Figure 2: compare divergence (repelling forces) for scalar vs matrix kernel.

    Creates two scenarios: (a) similar convergence rates and (b) observed component converges faster.
    Plots divergence vectors for scalar and matrix kernels.
    """
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    def scalar_kernel_divergence(x_particles, x_eval, A):
        # divergence wrt x_i: -A^T (x_i - x) K where K = exp(-0.5*(x_i-x)^T A (x_i-x))
        diffs = x_particles - x_eval[None, :]
        Kvals = np.exp(-0.5 * np.sum(diffs @ A * diffs, axis=1))
        divs = - (A.T @ diffs.T).T * Kvals[:, None]
        return divs, Kvals

    def matrix_kernel_divergence(x_particles, x_eval, alpha, prior_std):
        # per-component kernel: K^{(a)} = exp(- (x_i^a - x^a)^2 / (2 alpha sigma_a^2))
        diffs = x_particles - x_eval[None, :]
        sig2 = (prior_std ** 2)
        Kcomp = np.exp(- (diffs ** 2) / (2.0 * alpha * sig2[None, :]))
        # divergence per component: - (x_i^a - x^a)/(alpha sigma_a^2) * K^{(a)}
        divs = - (diffs / (alpha * sig2[None, :])) * Kcomp
        # full vector divergence is component-wise (diagonal kernel)
        return divs, Kcomp.mean(axis=1)

    # scenario settings
    scenarios = [
        {'name': 'similar', 'prior_std': np.array([1.0, 1.0]), 'alpha': 1.0},
        {'name': 'observed_fast', 'prior_std': np.array([2.0, 0.2]), 'alpha': 1.0}
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, sc in enumerate(scenarios):
        # particles arranged around two clusters
        x_particles = np.array([[ -1.0, -0.2], [0.0, 0.0], [0.5, 1.0], [1.2, 0.8]])
        x_eval = np.array([0.2, 0.2])
        # scalar kernel A choose inverse prior covariance scaled
        B = np.diag(sc['prior_std'] ** 2)
        A = np.linalg.inv(sc['alpha'] * B)

        sdiv, sK = scalar_kernel_divergence(x_particles, x_eval, A)
        mdiv, mK = matrix_kernel_divergence(x_particles, x_eval, sc['alpha'], sc['prior_std'])

        ax1 = axes[i, 0]
        ax1.scatter(x_particles[:,0], x_particles[:,1], c='k')
        ax1.scatter([x_eval[0]], [x_eval[1]], c='gray')
        for p, v in zip(x_particles, sdiv):
            ax1.arrow(p[0], p[1], v[0], v[1], head_width=0.05, color='C0')
        ax1.set_title(f"Scalar kernel - {sc['name']}")

        ax2 = axes[i, 1]
        ax2.scatter(x_particles[:,0], x_particles[:,1], c='k')
        ax2.scatter([x_eval[0]], [x_eval[1]], c='gray')
        for p, v in zip(x_particles, mdiv):
            ax2.arrow(p[0], p[1], v[0], v[1], head_width=0.05, color='C1')
        ax2.set_title(f"Matrix kernel - {sc['name']}")

    plt.tight_layout()
    fig.savefig(savepath)
    print(f"Saved Figure 2 reproduction to {savepath}")


def figure3_posterior_marginals_lorenz(d=40, Np=200, obs_indices=None, savepath='results/figure3_posterior.png', seed=42, flow_kwargs=None, save_csv: bool = True):
    """Reproduce Figure 3-like posterior marginals using Lorenz96 (small-scale).

    This function simulates Lorenz96, runs one assimilation step at t=T (default T=20)
    and applies kernel-scalar and kernel-matrix flows to compare posterior marginals
    for two components (a chosen unobserved and observed index).
    """
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    # Lorenz-96 dynamics (standard formulation)
    def lorenz96_step(x, F=8.0, dt=0.01):
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        N = len(x)
        dx = np.zeros(N)
        for i in range(N):
            dx[i] = (x[(i+1)%N] - x[(i-2)%N]) * x[(i-1)%N] - x[i] + F
        return x + dt * dx

    # simulate until T steps
    T = 20
    x = np.zeros(d)
    Q = np.eye(d) * 0.1
    R = np.eye(len(obs_indices) if obs_indices is not None else d) * 1.0

    # choose observation indices if not provided: observe every other variable
    if obs_indices is None:
        obs_indices = np.arange(0, d, 2)

    # propagate to time T
    for t in range(T):
        x = lorenz96_step(x)
    prior_particles = rng.multivariate_normal(x, np.eye(d)*1.0, size=Np)

    # observation at T: observe obs_indices with Gaussian noise
    y = prior_particles.mean(axis=0)[obs_indices] + rng.normal(scale=1.0, size=len(obs_indices))

    # Prepare H matrix that selects observed indices
    H = np.zeros((len(obs_indices), d))
    for i, idx in enumerate(obs_indices):
        H[i, idx] = 1.0

    # Apply kernel-scalar and kernel-matrix flows using PF-PF wrapper
    flow_kwargs = {} if flow_kwargs is None else dict(flow_kwargs)
    pf_scalar = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=1, use_local_flow=False, flow_type='kernel-scalar', flow_kwargs=flow_kwargs)
    pf_scalar.particles = tf.convert_to_tensor(prior_particles, dtype=tf.float64)
    pf_scalar.log_weights = tf.fill([Np], tf.cast(tf.math.log(1.0/float(Np)), dtype=tf.float64))

    pf_matrix = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=2, use_local_flow=True, flow_type='kernel-matrix', flow_kwargs=flow_kwargs)
    pf_matrix.particles = tf.convert_to_tensor(prior_particles.copy(), dtype=tf.float64)
    pf_matrix.log_weights = tf.fill([Np], tf.cast(tf.math.log(1.0/float(Np)), dtype=tf.float64))

    # single update
    pf_scalar.update(y=y, H=H, R=R)
    pf_matrix.update(y=y, H=H, R=R)

    # choose components to plot: pick one unobserved and one observed nearby
    obs_comp = obs_indices[0]
    unobs_comp = (obs_comp - 1) % d

    prior_vals_obs = prior_particles[:, obs_comp]
    prior_vals_unobs = prior_particles[:, unobs_comp]
    post_scalar = pf_scalar.particles.numpy()
    post_matrix = pf_matrix.particles.numpy()

    # KDEs for prior and posterior marginals
    kde_prior_obs = gaussian_kde(prior_vals_obs)
    kde_prior_unobs = gaussian_kde(prior_vals_unobs)
    kde_post_scalar_obs = gaussian_kde(post_scalar[:, obs_comp])
    kde_post_scalar_unobs = gaussian_kde(post_scalar[:, unobs_comp])
    kde_post_matrix_obs = gaussian_kde(post_matrix[:, obs_comp])
    kde_post_matrix_unobs = gaussian_kde(post_matrix[:, unobs_comp])

    xs = np.linspace(np.min(prior_vals_obs)-1, np.max(prior_vals_obs)+1, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0,0].plot(xs, kde_prior_unobs(xs), label='prior')
    axes[0,0].plot(xs, kde_post_scalar_unobs(xs), label='post scalar')
    axes[0,0].plot(xs, kde_post_matrix_unobs(xs), label='post matrix')
    axes[0,0].scatter(prior_vals_unobs, np.zeros_like(prior_vals_unobs), c='k', s=10)
    axes[0,0].set_title(f'Unobserved component {unobs_comp}')
    axes[0,0].legend()

    axes[0,1].plot(xs, kde_prior_obs(xs), label='prior')
    axes[0,1].plot(xs, kde_post_scalar_obs(xs), label='post scalar')
    axes[0,1].plot(xs, kde_post_matrix_obs(xs), label='post matrix')
    axes[0,1].scatter(prior_vals_obs, np.zeros_like(prior_vals_obs), c='k', s=10)
    axes[0,1].set_title(f'Observed component {obs_comp}')
    axes[0,1].legend()

    plt.tight_layout()
    fig.savefig(savepath)
    print(f"Saved Figure 3 reproduction to {savepath}")
    # Optionally save CSVs for diagnostics
    if save_csv:
        csv_dir = os.path.splitext(savepath)[0] + '_csv'
        os.makedirs(csv_dir, exist_ok=True)
        np.savetxt(os.path.join(csv_dir, 'prior_vals_obs.csv'), prior_vals_obs, delimiter=',')
        np.savetxt(os.path.join(csv_dir, 'prior_vals_unobs.csv'), prior_vals_unobs, delimiter=',')
        np.savetxt(os.path.join(csv_dir, 'post_scalar_obs.csv'), post_scalar[:, obs_comp], delimiter=',')
        np.savetxt(os.path.join(csv_dir, 'post_scalar_unobs.csv'), post_scalar[:, unobs_comp], delimiter=',')
        np.savetxt(os.path.join(csv_dir, 'post_matrix_obs.csv'), post_matrix[:, obs_comp], delimiter=',')
        np.savetxt(os.path.join(csv_dir, 'post_matrix_unobs.csv'), post_matrix[:, unobs_comp], delimiter=',')
        # Save kernel diagnostics if available
        try:
            with open(os.path.join(csv_dir, 'flow_kwargs.txt'), 'w') as fh:
                fh.write(str(flow_kwargs))
        except Exception:
            pass
        print(f"Saved diagnostic CSVs to {csv_dir}")
    


def experiment_linear_gaussian_part2(T=10, trials=100, Np=10000, sigma_p_list=None, sigma_z=1.0, seed=10, smoke=False):
    """Sensitivity experiment: Perturbed predictive covariance eigenvalues.

    For each trial and time step we compute the Kalman-predictive covariance
    P_pred and perturb its eigenvalues by iid LogNormal(0, sigma_p^2) factors:
        	ilde{P} = V diag( xi_i * lambda_i ) V^T,
    where xi_i ~ LogNormal(0, sigma_p^2).

    We pass `tilde_P` into the PF-PF flow via the `prior_cov_override` argument
    so the flow uses the perturbed predictive covariance instead of the empirical
    particle prior covariance. The function reports average MSE across trials
    for each sigma_p in `sigma_p_list`.

    Defaults are conservative for quick runs; set `smoke=True` to force small
    `trials` and `Np` for fast verification.
    """
    if sigma_p_list is None:
        sigma_p_list = [0.0, 0.05, 0.1, 0.2]

    if smoke:
        trials = min(trials, 3)
        Np = min(Np, 200)

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    # reuse the linear Gaussian setup from experiment_linear_gaussian
    d = 64
    side = int(np.sqrt(d))

    alpha = 0.9
    alpha0 = 3.0
    alpha1 = 0.01
    beta = 20.0

    # spatial covariance Sigma and process noise Q (same as before)
    coords = np.array([[i, j] for i in range(side) for j in range(side)], dtype=float)
    Sigma = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            dist2 = np.sum((coords[i] - coords[j])**2)
            Sigma[i, j] = alpha0 * np.exp(-dist2 / beta) + (alpha1 if i == j else 0.0)
    Q = Sigma * 0.1

    A = alpha * np.eye(d)
    H = np.eye(d)
    R = (sigma_z ** 2) * np.eye(d)

    results = {sp: [] for sp in sigma_p_list}
    times = {sp: [] for sp in sigma_p_list}

    # initial covariance (used by Kalman recursion)
    P0 = np.eye(d) * 1.0

    for sp in sigma_p_list:
        print(f"Running sigma_p={sp} (trials={trials}, Np={Np})")
        for trial in range(trials):
            t0_sp = time.perf_counter()
            # simulate a short trajectory
            x = np.zeros(d)
            xs = np.zeros((T, d))
            zs = np.zeros((T, d))
            for t in range(T):
                v = rng.multivariate_normal(np.zeros(d), Q)
                x = alpha * x + v
                w = rng.multivariate_normal(np.zeros(d), R)
                zt = x + w
                xs[t] = x
                zs[t] = zt

            # setup the PF-PF filter
            pfpf = ParticleFilterWithInvertibleFlow(num_particles=Np, state_dim=d, seed=123 + trial, use_local_flow=False, verbose_or_not=False)
            pfpf.initialize(mean=np.zeros(d), cov=np.eye(d) * 1.0)

            # Kalman recursion variables to compute predictive covariances
            P_prev = P0.copy()

            mse_list = []
            for t in range(T):
                # Predict
                P_pred = A @ P_prev @ A.T + Q

                # Eigendecompose P_pred and perturb eigenvalues
                vals, vecs = np.linalg.eigh(P_pred)
                # sample lognormal multipliers with mean 0
                xi = rng.lognormal(mean=0.0, sigma=sp, size=d)
                pert_vals = vals * xi
                tilde_P = (vecs * pert_vals) @ vecs.T

                # PF-PF predict/update using perturbed predictive covariance
                pfpf.predict(dynamics_fn=lambda x_tf: tf.cast(alpha, tf.float64) * x_tf, process_noise_cov=Q)
                pfpf.update(y=zs[t], H=H, R=R, prior_cov_override=tilde_P)

                mean = np.mean(pfpf.particles, axis=0)
                mse_list.append(np.mean((mean - xs[t]) ** 2))

                # Kalman update to produce P_prev for next time step
                S = P_pred + R
                K = P_pred @ np.linalg.inv(S)
                P_upd = (np.eye(d) - K) @ P_pred
                P_prev = P_upd

            t1_sp = time.perf_counter()
            results[sp].append(float(np.mean(mse_list)))
            times[sp].append(t1_sp - t0_sp)

        # summarize for this sigma_p
        print(f"sigma_p={sp}: avg MSE over trials = {np.mean(results[sp]):.4e}, avg time={np.mean(times[sp]):.3f}s")

    # Final summary
    print("Sensitivity experiment summary (perturbed predictive covariance):")
    for sp in sigma_p_list:
        print(f"  sigma_p={sp}: mean MSE={np.mean(results[sp]):.4e}, time_s={np.mean(times[sp]):.3f}")

def main():
    print("Running experiments (may take a while)...")
    # experiment_acoustic_tracking()
    experiment_linear_gaussian(T=10)
    # experiment_skewed_t_counts()


if __name__ == '__main__':
    main()
