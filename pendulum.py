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

        # Logging 
        self.episode_log = [] # Log of the episode. It will contain pairs of (state, force) for each time step in the episode.
        self.logging = config.logging.enable # Enable logging
        self.log_overwrite = config.logging.overwrite # Overwrite existing log file or append to it
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
                f.write('ep,state,action\n')
                
                # Write the state-action pairs for the first episode (ep=0)
                for state, action in self.episode_log:
                    # Convert the state to a string and write to file
                    f.write(f'0,"{state}",{action}\n')
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
                for state, action in self.episode_log:
                    # Convert the state to a string and write to file
                    f.write(f'{new_ep},"{state}",{action}\n')

        self.episode_log = [] # Clear the episode log
        

    def derivate_state(self, state, force):
        """
        Returns the derivative of the state vector given the current state and the force acting on the base.
        """
        vel = [state[2], state[3]] # x_vel, y_vel
        angle = state[4] # Angle
        ang_vel = state[5] # Angular velocity
        u = np.array([np.cos(angle), np.sin(angle)]) # Unit vector in the direction of the pendulum
        fb = np.dot(force, u)*u + np.array([0, -self.g]) # Force acting on the bob
        base_acc = (force)/self.mass_base # Acceleration of the base (independent of the bob)
        ang_acc = np.cross(force-fb, self.length*u)/(self.mass_bob*self.length**2) # Angular acceleration of the bob
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

        

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        deterministic = self.config.deterministic_initial_state
        down = self.config.initial_state=='down'

        noise = np.zeros(6) if deterministic else np.random.normal(0, 0.1, 6) 
        init_angle = -np.pi/2 if down else np.pi/2
        self.state = np.array([0, 0, 0, 0, init_angle, 0]) + noise
        self.terminated = False
        self.time = 0

        return self.state
    
    def reward(self):

        # Reward function:
        # First we have a reward for increasing the height of the bob. 
        # This is determined by the angle theta, assuming theta = 0 is the vertical position.
        # It is R_max at the top and decreases exponentially as the angle increases towards pi (downwards).
        # R(theta) = R_max * exp(-k * abs(x)) * sin((pi - abs(x)) / 2), let k = 1

        # Now we add a scaling factor for the reward based on the x position of the base. 
        # This scaling factor is S_max at the center and decreases exponentially towards S_min when x = x_lim.
        # S(x) = (S_max - S_min) * exp(-k * abs(x)) * sin(pi/2 - pi/(2*x_lim) * abs(x)) + S_min 

        # Lastly we want to reward the agent for getting to the top fast. 
        # To do this we add a scaling factor which depends on the time. It starts at some high value and decreases towards 1 as the time increases.
        # For this we can use a sigmoid function.
        # f(x) = A + (1 - A) / (1 + exp(-k * (t - t_0))), where A is the scaling factor at t=0, t_0 is the time at which the scaling factor is (A-1)/2, and k is a constant determining the steepness of the sigmoid (4 looks good).

        # 1) Convert the angle to the range [-pi, pi] with 0 being the vertical position
        angle = self.state[4] - np.pi/2 # make 0 the vertical position
        # Convert angle to range -pi to pi
        if angle > np.pi:
            angle -= 2*np.pi
        
        # 2) Calculate the reward for the angle
        R_max = 1
        k = 1
        angle_reward = R_max * np.exp(-k * np.abs(angle)) * np.sin((np.pi - np.abs(angle)) / 2)

        # 3) Calculate the scaling for the x position
        S_max = 1.5
        S_min = 0.75
        x = self.state[0] # x position of the base
        x_pos_scaling = (S_max - S_min) * np.exp(-k * np.abs(x)) * np.sin(np.pi/2 - np.pi/(2*self.x_lim) * np.abs(x)) + S_min

        # 4) Calculate the scaling for the time
        # A = 1.5
        # t_0 = 2
        # k = 4
        # time_scaling = A + (1 - A) / (1 + np.exp(-k * (self.time - t_0))) # Time is not an input, so on second thought this might be detrimental

        # 5) Penalize high angular velocity.
        # We add a penalty for high angular velocity using a scaling of the reward based on the angular velocity.
        # This scaling is 1 at 0 angular velocity and decreases exponentially towards 0 as the angular velocity increases.
        angular_vel_scaling = np.exp(-np.abs(self.state[5])/5)

        # 5) Calculate the total reward
        return angle_reward * x_pos_scaling * angular_vel_scaling #* time_scaling


        # sin_angle = np.sin(self.state[4])
        
        # return 1 + sin_angle
        # return 1 if np.abs(angle) < np.pi/6 else 0

    def step(self, force):
        """
        Updates the state of the pendulum given a force acting on the base.
        
        Args:
            force List[float, float]: Force vector acting on the base of the pendulum.
        """

        if self.logging:
            self.episode_log.append((self.state.copy(), force)) # Log the state-action pair

        self.update(force)
        reward = self.reward()
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

           
    def animate(self, agent, continuous=False):
        import pygame as pg

        pg.init() #initialize pygame

        #FPS
        # FPS = 60 # frames per second setting
        fpsClock = pg.time.Clock()

        dt = 0.05 # 1 / FPS # Seconds Per Frame = 1 / Frames Per Second


        # Set up the drawing window
        width = 700
        height = 600
        screen_dim = [width, height]
        screen = pg.display.set_mode(screen_dim)

        agent.eps = 0.0
        agent.noise_std = 0.0

        running = True
        while running:
            # Fill the background with white
            screen.fill((255, 255, 255))

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            
            state = agent.get_state_representation(self.state)
            force, _ = agent.act(state)

            pos = np.array([self.state[0], self.state[1]])
            angle = self.state[4]
            
            p_base = np.array([0, height])+60*np.array([1, -1])*(pos + np.array([width/120, height/120])) # base of pendulum (flip y-axis)
            p_end = np.array([0, height])+60*np.array([1, -1])*((pos + np.array([width/120, height/120])) + self.length*np.array([np.cos(angle), np.sin(angle)])) # end of pendulum (flip y-axis)
            pg.draw.line(screen, (0,0,0), p_base, p_end , 2)

            self.update(force)

            pg.display.flip() # update screen?
            fpsClock.tick(1/dt)
            
        # Done! Time to quit.
        pg.quit()


