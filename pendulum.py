import numpy as np
import matplotlib.pyplot as plt
import random
from dataModels import PendulumParams, Config
import os
import pandas as pd

class Pendulum:
    """
    Simulates a pendulum environment for reinforcement learning.

    The pendulum consists of a massless rod of a certain length with a bob at the end.
    The base of the pendulum can be moved horizontally by applying a force. The goal is
    to balance the pendulum in the upright position by moving the base.
    """
    def __init__(self, config : Config) -> None:
        """
        Initializes the Pendulum environment with the provided configuration.

        Parameters
        ----------
        config : Config
            Configuration parameters for the pendulum environment.
        """
        self.config = config
        self.length = config.pendulum.length # Length of the pendulum
        self.mass_base = config.pendulum.mass_base # Mass of the base
        self.mass_bob = config.pendulum.mass_bob # Mass of the pendulum 
        self.x_lim = config.pendulum.termination.x_limit # x limit for the pendulum (in meters). Set to None for no limit.
        self.angle_lim = None if config.pendulum.termination.angle_limit is None else config.pendulum.termination.angle_limit * np.pi/180 # Angle limit for the pendulum (in degrees). Set to None for no limit.
        self.time_limit = config.pendulum.termination.time_limit # Time limit for the episode in seconds. Set to None for no limit.
        self.state = self.reset() # State vector [x_pos, y_pos, x_vel, y_vel, angle, ang_vel]
        self.terminated = False # Termination flag
        self.termination_penalty = config.pendulum.termination.termination_penalty # Penalty for terminating the episode early
        self.time_step = config.pendulum.time_step # Time step for the simulation
        self.g = config.pendulum.gravity # Acceleration due to gravity (m/s^2)
        self.time = 0 # Current time in the episode

        # Logging 
        self.episode_log = [] # Log of the episode. It will contain pairs of (state, force) for each time step in the episode.
        self.logging_enable = config.pendulum.logging.enable # Enable logging
        self.log_file_path = config.pendulum.logging.file_path # File path to save log to

    def write_log_to_file(self):
        """
        Writes the episode log to a file specified in the configuration.

        If the log file already exists, the new episode data is appended.
        Otherwise, a new file is created with headers.
        """
        log_file_exists = os.path.exists(self.log_file_path)

        if not log_file_exists:
            # File does not exist, create it and write the header and the first episode
            with open(self.log_file_path, 'w') as f:
                # Write the header
                f.write('ep,state,action,reward\n')
                
                # Write the state-action pairs for the first episode (ep=0)
                for state, action, reward in self.episode_log:
                    # Convert the state to a string and write to file
                    f.write(f'0,"{state}",{action},{reward}\n')
        else:
            # File exists, read the last episode number
            df = pd.read_csv(self.log_file_path)

            # Get the maximum episode number in the current file
            max_ep = df['ep'].max()

            # Increment the episode number for the new episode
            new_ep = max_ep + 1

            # Open the file in append mode and write the new episode data
            with open(self.log_file_path, 'a') as f:
                # Write the state-action pairs for the new episode
                for state, action, reward in self.episode_log:
                    # Convert the state to a string and write to file
                    f.write(f'{new_ep},"{state}",{action},{reward}\n')

        self.episode_log = [] # Clear the episode log
        

    def derivate_state(self, state, force):
        """
        Computes the derivative of the state vector given the current state and applied force.

        Parameters
        ----------
        state : np.ndarray
            The current state vector [x_pos, y_pos, x_vel, y_vel, angle, ang_vel].
        force : np.ndarray
            The force vector applied to the base [force_x, force_y].

        Returns
        -------
        state_dot : np.ndarray
            The derivative of the state vector.
        """
        vel = [state[2], state[3]] # x_vel, y_vel
        angle = state[4] # Angle
        ang_vel = state[5] # Angular velocity
        u = np.array([np.cos(angle), np.sin(angle)]) # Unit vector in the direction of the pendulum
        fb = np.dot(force, u)*u + np.array([0, -self.g]) # Force acting on the bob
        base_acc = (force)/self.mass_base # Acceleration of the base (independent of the bob)
        ang_acc = np.cross(force-fb, self.length*u)/(self.mass_bob*self.length**2) - self.config.pendulum.damping_coefficient*ang_vel # Angular acceleration of the bob
        state_dot = np.array([vel[0], vel[1], base_acc[0], base_acc[1], ang_vel, ang_acc]) # Derivative of the state vector
        return state_dot

        
    def update(self, force):
        """
        Updates the state of the pendulum using the Runge-Kutta 4th order method.

        Parameters
        ----------
        force : np.ndarray
            The force vector applied to the base [force_x, force_y].
        """
        k1 = self.derivate_state(self.state, force)*self.time_step
        k2 = self.derivate_state(self.state + k1/2, force)*self.time_step
        k3 = self.derivate_state(self.state + k2/2, force)*self.time_step
        k4 = self.derivate_state(self.state + k3, force)*self.time_step
        self.state += (k1 + 2*k2 + 2*k3 + k4)/6
        
        self.state[4] = self.state[4] % (2*np.pi) # Keep angle between 0 and 2pi

    def get_state_representation(self, state):
        """
        Converts the state vector into a representation suitable for the agent.

        The representation includes normalized values and angle components as sin and cos.

        Parameters
        ----------
        state : np.ndarray
            The current state vector [x_pos, y_pos, x_vel, y_vel, angle, ang_vel].

        Returns
        -------
        state_repr : np.ndarray
            The processed state vector suitable for the agent [x_pos_norm, x_vel_norm, cos(angle), sin(angle), ang_vel_norm].
        """
        x_pos = state[0]
        x_vel = state[2]
        angle = state[4]
        angle_vel = state[5]

        # Normalize the x position if a limit is set
        x_pos_norm = x_pos / self.x_lim if self.x_lim is not None else x_pos

        # Normalize the x-velocity (assuming max x-velocity is 7 m/s)
        max_x_vel = 7
        x_vel_norm = x_vel / max_x_vel

        # Normalize the angle velocity (assuming max angle velocity is 7 rad/s)
        max_angle_vel = 7
        angle_vel_norm = angle_vel / max_angle_vel

        # Return the state vector with the normalized values and the angle represented as sin and cos
        state = np.array([x_pos_norm, x_vel_norm, np.cos(angle), np.sin(angle), angle_vel_norm])
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns
        -------
        state : np.ndarray
            The reset state vector.
        """
        deterministic = self.config.pendulum.deterministic_initial_state
        down = self.config.pendulum.initial_state=='down'
        if self.config.pendulum.initial_state == 'random':
            down = random.choice([True, False])

        noise = np.zeros(6) if deterministic else np.random.normal(0, 0.1, 6) 
        init_angle = -np.pi/2 if down else np.pi/2
        self.state = np.array([0, 0, 0, 0, init_angle, 0]) + noise
        self.terminated = False
        self.time = 0

        return self.state
    
    def reward(self, force=None):
        """
        Calculates the reward for the current state-action pair.
        The reward is computed based on the deviation from the goal state (upright and stationary at the center).

        Parameters
        ----------
        force : np.ndarray, optional
            The force vector applied to the base. Default is None.

        Returns
        -------
        total_reward : float
            The calculated reward value.
        """

        # Constants
        angle = self.state[4] - np.pi/2  # Deviation from upright
        angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        angular_velocity_norm = self.state[5] / 7  # Normalize angular velocity to [-1, 1] (assuming max angular velocity is 7 rad/s)
        x_position = self.state[0] / self.x_lim if self.x_lim is not None else self.state[0] # Normalize x position to [-1, 1] if a limit is set

        # Compute the combined deviation from the goal state
        # The goal state is upright and stationary at the center
        deviation = np.sqrt((angle/np.pi)**2 + angular_velocity_norm**2 + x_position**2)

        # Gaussian reward centered at deviation = 0
        sigma = 0.3  # Standard deviation of the Gaussian 
        stability = np.exp(- (deviation**2) / (2 * sigma**2))

        force_penalty = 0 if force is None else -0.01 * ((force[0]/self.config.model.force_magnitude) ** 2)  # Penalize excessive force

        total_reward = stability + force_penalty # Total reward
        return total_reward
  

    def step(self, force):
        """
        Executes one time step in the environment with the given action.

        Parameters
        ----------
        force : np.ndarray
            The force vector applied to the base [force_x, force_y].

        Returns
        -------
        state : np.ndarray
            The updated state vector after applying the action.
        reward : float
            The reward obtained after taking the action.
        done : bool
            Flag indicating if the episode has terminated.
        """
        if self.logging_enable:
            self.episode_log.append((self.state.copy(), force[0], self.reward(force))) # Log the state-action pair

        self.update(force)
        reward = self.reward(force) if self.config.model.type == 'DDPG' else self.reward()
        self.time += self.time_step

        # Check termination:
        abs_pos = np.abs(self.state[0])
        abs_angle = np.abs(self.state[4]-np.pi/2)

        if self.angle_lim is not None and abs_angle > self.angle_lim:
            self.terminated = True
            reward = self.termination_penalty
            
        if self.x_lim is not None and abs_pos > self.x_lim:
            self.terminated = True
            reward = self.termination_penalty

        if self.time_limit is not None and self.time > self.time_limit:
            self.terminated = True

        if self.terminated and self.logging_enable:
            self.write_log_to_file()

        return self.state, reward, self.terminated # New-state, reward, done
