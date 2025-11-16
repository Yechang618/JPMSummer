"""
Particle Flow Filter implementation using TensorFlow and TensorFlow Probability.

This implementation is based on the particle flow concept where particles evolve
continuously from the prior to the posterior distribution using a transport equation.
The filter uses TensorFlow for efficient vectorized operations and TensorFlow Probability
for probability distributions and sampling.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

class ParticleFlowFilter:
    """
    Particle Flow Filter implementation using continuous transport of particles.
    
    This filter evolves particles continuously from prior to posterior using a flow-based
    approach, which can help avoid particle degeneracy issues common in traditional
    particle filters.
    
    Attributes:
        n_particles (int): Number of particles to use
        state_dim (int): Dimension of the state space
        dtype (tf.dtype): Data type for computations (default: tf.float32)
        particles (tf.Tensor): Current particle states of shape [n_particles, state_dim]
        log_weights (tf.Tensor): Log weights of particles of shape [n_particles]
    """
    
    def __init__(self, n_particles, state_dim, dtype=tf.float64):
        """
        Initialize the Particle Flow Filter.
        
        Args:
            n_particles (int): Number of particles to use
            state_dim (int): Dimension of the state space
            dtype (tf.dtype, optional): Data type for computations. Defaults to tf.float32.
        """
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.dtype = dtype
        
        # Initialize particles and weights
        self.particles = None
        self.log_weights = None
        
        # Flow parameters
        self.n_flow_steps = 10  # Number of integration steps for the flow
        self.learning_rate = 0.1  # Step size for the flow
        
    def initialize(self, initial_distribution):
        """
        Initialize particles from an initial distribution.
        
        Args:
            initial_distribution (tfd.Distribution): Initial state distribution
        """
        self.particles = initial_distribution.sample(self.n_particles)
        self.log_weights = tf.zeros([self.n_particles], dtype=self.dtype)
        
    def predict(self, dynamics_fn, process_noise_cov):
        """
        Predict step: propagate particles through system dynamics.
        
        Args:
            dynamics_fn (callable): Function that takes current state and returns next state
            process_noise_cov (tf.Tensor): Process noise covariance matrix
        """
        # Propagate particles through dynamics
        self.particles = dynamics_fn(self.particles)
        
        # Add process noise
        noise = tf.random.normal([self.n_particles, self.state_dim], 
                               dtype=self.dtype)
        noise_scaled = tf.linalg.matmul(noise, 
                                      tf.cast(tf.linalg.cholesky(process_noise_cov),dtype=self.dtype))
        self.particles += noise_scaled
        
    def _compute_flow(self, particles, observation, observation_fn, 
                     observation_noise_cov, lambda_t):
        """
        Compute the particle flow for a single integration step.
        
        Args:
            particles (tf.Tensor): Current particle states
            observation (tf.Tensor): Current observation
            observation_fn (callable): Function mapping states to observations
            observation_noise_cov (tf.Tensor): Observation noise covariance
            lambda_t (float): Flow time parameter between 0 and 1
            
        Returns:
            tf.Tensor: Flow vector for each particle
        """
        with tf.GradientTape() as tape:
            tape.watch(particles)
            
            # Compute innovation (difference between predicted and actual observation)
            predicted_obs = observation_fn(particles)
            innovation = observation - predicted_obs
            
            # Compute log-likelihood
            obs_dist = tfd.MultivariateNormalTriL(
                loc=predicted_obs,
                scale_tril=tf.linalg.cholesky(observation_noise_cov)
            )
            log_likelihood = obs_dist.log_prob(observation)
            
        # Compute gradient of log-likelihood
        grad = tape.gradient(log_likelihood, particles)
        
        # Scale gradient by lambda parameter
        flow = lambda_t * grad
        return flow
        
    def update(self, observation, observation_fn, observation_noise_cov):
        """
        Update step: evolve particles using flow to incorporate observation.
        
        Args:
            observation (tf.Tensor): Current observation
            observation_fn (callable): Function mapping states to observations
            observation_noise_cov (tf.Tensor): Observation noise covariance matrix
        """
        dt = 1.0 / self.n_flow_steps
        
        # Integrate flow
        current_particles = self.particles
        for i in range(self.n_flow_steps):
            lambda_t = (i + 1) * dt
            
            # Compute flow
            flow = self._compute_flow(
                current_particles,
                observation,
                observation_fn,
                observation_noise_cov,
                lambda_t
            )
            
            # Update particles using Euler integration
            current_particles = current_particles + self.learning_rate * flow * dt
            
        self.particles = current_particles
        
    def get_state_estimate(self):
        """
        Compute the current state estimate as the mean of the particles.
        
        Returns:
            tf.Tensor: Mean state estimate
        """
        return tf.reduce_mean(self.particles, axis=0)
    
    def get_state_covariance(self):
        """
        Compute the current state covariance estimate from the particles.
        
        Returns:
            tf.Tensor: Covariance matrix estimate
        """
        mean = self.get_state_estimate()
        centered = self.particles - mean
        return tf.matmul(centered, centered, transpose_a=True) / self.n_particles

if __name__ == "__main__":
    # Simple 1D example
    tf.random.set_seed(42)
    
    # Create filter
    n_particles = 1000
    state_dim = 1
    pf = ParticleFlowFilter(n_particles, state_dim)
    
    # Initialize from prior
    initial_dist = tfd.Normal(loc=0., scale=1.)
    pf.initialize(initial_dist)
    
    # Define linear dynamics and observation functions
    def dynamics_fn(x):
        return 0.9 * x
    
    def observation_fn(x):
        return x
    
    # True state and observation
    true_state = tf.constant([[2.0]])
    process_noise_cov = tf.constant([[0.1]])
    observation_noise_cov = tf.constant([[0.2]])
    
    # Generate observation
    observation = true_state + tf.random.normal([1, 1], stddev=np.sqrt(0.2))
    
    # Predict and update
    pf.predict(dynamics_fn, process_noise_cov)
    pf.update(observation, observation_fn, observation_noise_cov)
    
    # Get estimate
    state_estimate = pf.get_state_estimate()
    state_cov = pf.get_state_covariance()
    
    print(f"True state: {true_state.numpy()}")
    print(f"Observation: {observation.numpy()}")
    print(f"Estimated state: {state_estimate.numpy()}")
    print(f"Estimated state covariance: {state_cov.numpy()}")