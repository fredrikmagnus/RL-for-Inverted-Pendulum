import numpy as np
import matplotlib.pyplot as plt
import random
from dataModels import PendulumParams
import os
import pandas as pd

class Pendulum:
    def __init__(self, config : PendulumParams) -> None:
        self.config = config
        self.length = config.length # Length of the pendulum
        self.mass_base = config.mass_base # Mass of the base
        self.mass_bob = config.mass_bob # Mass of the pendulum 
        self.x_lim = config.termination.x_limit # x limit for the pendulum (in meters). Set to None for no limit.
        self.angle_lim = None if config.termination.angle_limit is None else config.termination.angle_limit * np.pi/180 # Angle limit for the pendulum (in degrees). Set to None for no limit.
        self.time_limit = config.termination.time_limit # Time limit for the episode in seconds. Set to None for no limit.
        self.state = self.reset() # State vector [x_pos, y_pos, x_vel, y_vel, angle, ang_vel]
        self.terminated = False # Termination flag
        self.termination_penalty = config.termination.termination_penalty # Penalty for terminating the episode early
        self.time_step = config.time_step # Time step for the simulation
        self.g = config.gravity # Acceleration due to gravity (m/s^2)
        self.time = 0 # Current time in the episode

        self.prev_force = 0 # Previous force applied to the base

        # Logging 
        self.episode_log = [] # Log of the episode. It will contain pairs of (state, force) for each time step in the episode.
        self.logging = config.logging.enable # Enable logging
        self.log_file_path = config.logging.file_path

    def write_log_to_file(self):
        """
        Writes the episode log to a file.
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
        Returns the derivative of the state vector given the current state and the force acting on the base.
        """
        vel = [state[2], state[3]] # x_vel, y_vel
        angle = state[4] # Angle
        ang_vel = state[5] # Angular velocity
        u = np.array([np.cos(angle), np.sin(angle)]) # Unit vector in the direction of the pendulum
        fb = np.dot(force, u)*u + np.array([0, -self.mass_bob*self.g]) # Force acting on the bob
        base_acc = (force)/self.mass_base # Acceleration of the base (independent of the bob)
        ang_acc = np.cross(force-fb, self.length*u)/(self.mass_bob*self.length**2) - self.config.damping_coefficient*ang_vel # Angular acceleration of the bob
        state_dot = np.array([vel[0], vel[1], base_acc[0], base_acc[1], ang_vel, ang_acc]) # Derivative of the state vector
        return state_dot

        
    def update(self, force):
        """
        Updates the state of the pendulum given a force acting on the base.

        Args:
            force List[float, float]: Force vector acting on the base of the pendulum.
        """
        k1 = self.derivate_state(self.state, force)*self.time_step
        k2 = self.derivate_state(self.state + k1/2, force)*self.time_step
        k3 = self.derivate_state(self.state + k2/2, force)*self.time_step
        k4 = self.derivate_state(self.state + k3, force)*self.time_step
        self.state += (k1 + 2*k2 + 2*k3 + k4)/6
        
        self.state[4] = self.state[4] % (2*np.pi) # Keep angle between 0 and 2pi

    def get_state_representation(self, state):
        # state = self.state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
        x_pos = state[0]
        x_vel = state[2]
        angle = state[4]
        angle_vel = state[5]

        x_pos = x_pos / self.x_lim if self.x_lim is not None else x_pos # Normalize x_pos

        max_x_vel = 7
        x_vel = x_vel / max_x_vel # Normalize x_vel

        max_angle_vel = 7
        angle_vel = angle_vel / max_angle_vel # Normalize angle_vel

        # state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]) # Convert angle to sin and cos signal
        state = np.array([x_pos, x_vel, np.cos(angle), np.sin(angle), angle_vel]) # Convert angle to sin and cos signal
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state

        

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        deterministic = self.config.deterministic_initial_state
        down = self.config.initial_state=='down'
        if self.config.initial_state == 'random':
            down = random.choice([True, False])

        noise = np.zeros(6) if deterministic else np.random.normal(0, 0.1, 6) 
        init_angle = -np.pi/2 if down else np.pi/2
        self.state = np.array([0, 0, 0, 0, init_angle, 0]) + noise
        self.terminated = False
        self.time = 0

        return self.state
    
    def reward(self, force=None):
        # Constants
        angle = self.state[4] - np.pi/2  # Deviation from upright
        angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        angular_velocity = self.state[5]
        x_position = self.state[0] / self.x_lim if self.x_lim is not None else self.state[0]
        # x_velocity = self.state[2]
        angular_velocity_norm = angular_velocity / 7  # Normalize angular velocity to [-1, 1]
        # Reward components
        # upright_reward = 0 #np.cos(angle)  # +1 when upright, -1 when inverted
        # velocity_penalty = 0 #-0.01 * (angular_velocity ** 2)  # Penalize high angular velocity
        # position_penalty = 0 #-0.1 * (x_position ** 2)  # Penalize being far from center
        force_penalty = -0.01 * ((force[0]/25) ** 2)  # Penalize excessive force

        # Compute the combined deviation from the goal state
        deviation = np.sqrt((angle/np.pi)**2 + angular_velocity_norm**2 + x_position**2)

        # Gaussian reward centered at deviation = 0 (upright and stationary)
        sigma = 0.3  # Controls the width of the peak; adjust as needed
        stability = np.exp(- (deviation**2) / (2 * sigma**2))

        # Total reward
        total_reward = stability + force_penalty # upright_reward + velocity_penalty + position_penalty + force_penalty + stability

        # Adjust reward to be positive
        # total_reward += 1  # Shift rewards to be mostly positive

        return total_reward

        # Reward function:
        # First we have a reward for increasing the height of the bob. 
        # This is determined by the angle theta, assuming theta = 0 is the vertical position.
        # It is R_max at the top and decreases exponentially as the angle increases towards pi (downwards).
        # R(theta) = R_max * exp(-k * abs(x)) * sin((pi - abs(x)) / 2), let k = 1

        # Now we add a scaling factor for the reward based on the x position of the base. 
        # This scaling factor is S_max at the center and decreases exponentially towards S_min when x = x_lim.
        # S(x) = (S_max - S_min) * exp(-k * abs(x)) * sin(pi/2 - pi/(2*x_lim) * abs(x)) + S_min 

        # # 1) Convert the angle to the range [-pi, pi] with 0 being the vertical position
        # angle = self.state[4] - np.pi/2 # make 0 the vertical position
        # # Convert angle to range -pi to pi
        # if angle > np.pi:
        #     angle -= 2*np.pi
        # x = self.state[0] # x position of the base
        # angular_vel = self.state[5] # Angular velocity

        # # State vector [x_pos, y_pos, x_vel, y_vel, angle, ang_vel]
        # # 2) Calculate the reward for the angle
        # R_max = 1
        # n = 1.5
        # # k = 0.5
        # # angle_reward = R_max * np.exp(-k * np.abs(angle)) * np.sin((np.pi - np.abs(angle)) / 2)
        # angle_reward = R_max * (1 - (np.abs(angle) / np.pi) ** n) # Reward for being upright

        

        # # 3) Calculate the scaling for the x position
        # S_max = 1
        # S_min = 0.2
        # k = 1
        # x_pos_scaling = (S_max - S_min) * np.exp(-k * np.abs(x)) * np.sin(np.pi/2 - np.pi/(2*self.x_lim) * np.abs(x)) + S_min


        # # 4) Penalize high angular velocity.
        # # We add a penalty for high angular velocity using a scaling of the reward based on the angular velocity.
        # # This scaling is 1 at 0 angular velocity and decreases exponentially towards 0 as the angular velocity increases.
        # angular_vel_scaling = np.exp(-np.abs(self.state[5])/5)

        # R_goal = 0.25 if np.abs(angle) < 0.25 and np.abs(angular_vel) < 0.5 else 0 # Reward for being close to the top and having low angular velocity

        # Penalty_force = 0 # if force is None else -0.005 * np.linalg.norm(force) # Penalty for using force
        
        # # 5) Calculate the total reward
        # return angle_reward * x_pos_scaling * angular_vel_scaling + R_goal + Penalty_force


        # R_theta = -1/np.pi**2 * angle**2 # Penalty for being far from the top
        # R_x = -2/self.x_lim**2 * x**2 # Penalty for being far from the center
        # R_goal = 2 if np.abs(angle) < 0.25 else 0 # Reward for being close to the top
        # R_up = 1 if np.abs(angle) < np.pi/2 else 0 # Reward for being upright
        # R_time = 0.3 # Reward for not ending the episode early

        # R_vel = -1/100 * angular_vel**2 # Penalty for high angular velocity

        # return R_theta + R_x + R_goal + R_time + R_vel + R_up

    def step(self, force):
        """
        Updates the state of the pendulum given a force acting on the base.
        
        Args:
            force List[float, float]: Force vector acting on the base of the pendulum.
        """

        if self.logging:
            self.episode_log.append((self.state.copy(), force[0], self.reward(force))) # Log the state-action pair

        self.update(force)
        reward = self.reward(force)
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

        if self.terminated and self.logging:
            self.write_log_to_file()

        return self.state, reward, self.terminated # New-state, reward, done
