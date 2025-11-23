import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from src.models.EDH import EDH
from src.models.KernelFlow import KernelScalarFlow, KernelMatrixFlow
from src.models.PFPF import EDH_ParticleFlowPF, LEDH_ParticleFlowPF

tfd = tfp.distributions

os.makedirs('figures', exist_ok=True)

def run_edh_diagnostics():
    tf.random.set_seed(1)
    np.random.seed(1)
    N = 200
    d = 3
    R = tf.constant(np.eye(d) * 0.5, dtype=tf.float64)
    Q = tf.constant(np.eye(d) * 0.1, dtype=tf.float64)
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(d, dtype=tf.float64), scale_diag=tf.ones(d, dtype=tf.float64))
    particles = prior.sample(N)

    edh = EDH(num_particles=N, f=None, h=lambda x: x, state_dim=d, observation_dim=d, R=R, Q=Q, dtype=tf.float64)
    edh.n_flow_steps = 6
    edh.initialize(particles=particles)

    # single observation (zero) to move particles
    y = tf.zeros(d, dtype=tf.float64)
    edh.update(y=y)

    # collect diagnostics
    A_hist = np.array([float(x.numpy()) for x in edh.flow_A_norm_history])
    b_hist = np.array([float(x.numpy()) for x in edh.flow_b_norm_history])
    disp_hist = np.array([float(x.numpy()) for x in edh.flow_disp_norm_history])
    cond_hist = np.array([float(x.numpy()) for x in edh.jacobian_cond_history])

    plt.figure()
    plt.plot(A_hist, marker='o')
    plt.title('EDH: A norm per lambda step')
    plt.grid(True)
    plt.savefig('figures/edh_A_norm.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(b_hist, marker='o')
    plt.title('EDH: b norm per lambda step')
    plt.grid(True)
    plt.savefig('figures/edh_b_norm.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(disp_hist, marker='o')
    plt.title('EDH: avg particle displacement per lambda step')
    plt.grid(True)
    plt.savefig('figures/edh_disp_norm.pdf', bbox_inches='tight')

    plt.figure()
    plt.semilogy(cond_hist, marker='o')
    plt.title('EDH: Jacobian condition number per lambda step')
    plt.grid(True)
    plt.savefig('figures/edh_jac_cond.pdf', bbox_inches='tight')

    print('EDH diagnostics saved to figures/')


def run_kernel_diagnostics():
    tf.random.set_seed(2)
    N = 200
    d = 3
    R = np.eye(d) * 0.5
    Q = np.eye(d) * 0.1
    prior = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=N)

    ks = KernelScalarFlow(n_flow_steps=6)
    km = KernelMatrixFlow(n_flow_steps=6)

    # single observation
    y = np.zeros(d)

    x_s, _ = ks.flow(prior, y, lambda x: x, R, np.eye(d))
    s_disp = np.array([float(x.numpy()) for x in ks.flow_disp_norm_history])
    s_jac = np.array([float(x.numpy()) for x in ks.jacobian_cond_history])

    plt.figure()
    plt.plot(s_disp, marker='o')
    plt.title('KernelScalarFlow: avg displacement per step')
    plt.grid(True)
    plt.savefig('figures/ks_disp.pdf', bbox_inches='tight')

    plt.figure()
    plt.semilogy(s_jac, marker='o')
    plt.title('KernelScalarFlow: proxy Jacobian condition per step')
    plt.grid(True)
    plt.savefig('figures/ks_jac_cond.pdf', bbox_inches='tight')

    x_m, _ = km.flow(prior, y, lambda x: x, R, np.eye(d))
    m_disp = np.array([float(x.numpy()) for x in km.flow_disp_norm_history])
    m_jac = np.array([float(x.numpy()) for x in km.jacobian_cond_history])

    plt.figure()
    plt.plot(m_disp, marker='o')
    plt.title('KernelMatrixFlow: avg displacement per step')
    plt.grid(True)
    plt.savefig('figures/km_disp.pdf', bbox_inches='tight')

    plt.figure()
    plt.semilogy(m_jac, marker='o')
    plt.title('KernelMatrixFlow: proxy Jacobian condition per step')
    plt.grid(True)
    plt.savefig('figures/km_jac_cond.pdf', bbox_inches='tight')

    print('Kernel flow diagnostics saved to figures/')


def run_pfpf_diagnostics():
    tf.random.set_seed(3)
    N = 200
    d = 3
    R = np.eye(d) * 0.5
    Q = np.eye(d) * 0.1
    prior = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=N)

    edh_pf = EDH_ParticleFlowPF(num_particles=N, f=None, h=lambda x: x, state_dim=d, observation_dim=d, Q=Q, R=R, dtype=tf.float64)
    ledh_pf = LEDH_ParticleFlowPF(num_particles=N, f=None, h=lambda x: x, state_dim=d, observation_dim=d, Q=Q, R=R, dtype=tf.float64)

    edh_pf.initialize(particles=prior)
    ledh_pf.initialize(particles=prior)

    # small sequence of observations
    T = 3
    ys = [np.zeros(d) for _ in range(T)]

    print('Running EDH_ParticleFlowPF filter for', T, 'steps')
    edh_pf.filter(ys)
    edh_A = np.array([float(x.numpy()) for x in edh_pf.flow_A_norm_history])
    edh_disp = np.array([float(x.numpy()) for x in edh_pf.flow_disp_norm_history])
    edh_jac = np.array([float(x.numpy()) for x in edh_pf.jacobian_cond_history])

    plt.figure()
    plt.plot(edh_A, marker='o')
    plt.title('EDH_PF: avg A norm per lambda step (aggregated across calls)')
    plt.grid(True)
    plt.savefig('figures/edh_pf_A_norm.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(edh_disp, marker='o')
    plt.title('EDH_PF: avg displacement per lambda step (aggregated across calls)')
    plt.grid(True)
    plt.savefig('figures/edh_pf_disp.pdf', bbox_inches='tight')

    plt.figure()
    plt.semilogy(edh_jac, marker='o')
    plt.title('EDH_PF: jacobian cond per lambda step (aggregated across calls)')
    plt.grid(True)
    plt.savefig('figures/edh_pf_jac_cond.pdf', bbox_inches='tight')

    print('PFPF diagnostics saved to figures/')


if __name__ == '__main__':
    run_edh_diagnostics()
    run_kernel_diagnostics()
    run_pfpf_diagnostics()
    print('All diagnostics saved in `figures/`')
