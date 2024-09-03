import numpy as np
import matplotlib.pyplot as plt
import random
from dataModels2 import PendulumParams

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

        return self.state
    
    def reward(self):
        # rmax_theta = 1
        # rmin_theta = .5
        # theta0 = np.pi/2
        # dtheta = self.angle_lim
        # thetamin = np.pi/2-dtheta
        # a_theta = (rmax_theta-rmin_theta)/(theta0**2 - 2*thetamin*theta0 + thetamin**2)
        # b_theta = -2*a_theta*thetamin
        # c_theta = a_theta*thetamin**2 + rmin_theta
        # theta = self.state[4]
        # r_theta = a_theta*theta**2 + b_theta*theta + c_theta if theta<=np.pi/2 else a_theta*(theta-2*dtheta)**2 + b_theta*(theta-2*dtheta) + c_theta

        # rmax_x = .3
        # rmin_x = -.2
        # dx = self.x_lim
        # a1=(rmax_x-rmin_x)/dx
        # a2=-a1
        # b=rmax_x
        # x = self.state[0]
        # r_pos =  a1*x+b if x<= 0 else a2*x+b

        # return r_theta+r_pos
        # return 1

        # x = self.state[0]
        # x_lim = self.x_lim
        # reward = 1 - abs(x) / x_lim
        # return max(reward, 0)

        # # Reward for swinging up
        # angle = self.state[4]
        # R1 = np.sin(angle) # Reward for upright position
        # x_pos = self.state[0]
        # R2 = -0.5*np.abs(x_pos/self.x_lim) # Reward for being close to center
        # return R1 + R2 +1 # Total reward
        
        sin_angle = np.sin(self.state[4])
        
        return 1 + sin_angle
        # return 1 if np.abs(angle) < np.pi/6 else 0

    def step(self, force):
        """
        Updates the state of the pendulum given a force acting on the base.
        
        Args:
            force List[float, float]: Force vector acting on the base of the pendulum.
        """

        self.update(force)

        # Check termination:
        abs_pos = np.abs(self.state[0])
        abs_angle = np.abs(self.state[4]-np.pi/2)
        if self.angle_lim is not None and abs_angle > self.angle_lim:
            self.terminated = True
        if self.x_lim is not None and abs_pos > self.x_lim:
            self.terminated = True

        reward = self.reward() if not self.terminated else self.termination_penalty

        return self.state, reward, self.terminated # New-state, reward, done
    
    # def step(self, action, dt):
    #     """ Returns: state, reward, done """
    #     # We assume for now that the action is either 0 (left) or 1 (right)
    #     if action == 0: action = -1 # Convert action 0 (left) to -1
    #     angle = self.state[4]
    #     min_force = 20
    #     max_force = 25
    #     force = lambda theta: min_force+np.abs(theta-np.pi/2)/self.angle_lim * (max_force-min_force) # Force is higher given higher angle error
    #     self.update(np.array([action*force(angle), 0]))
        
    #     # Check termination:
    #     abs_pos = np.abs(self.state[0])
    #     abs_angle = np.abs(self.state[4]-np.pi/2)
    #     if abs_pos > self.x_lim or abs_angle > self.angle_lim: # Terminate if outside limits
    #         self.terminated = True
    #     # if abs_pos > self.x_lim: # Terminates if outside x limits
    #     #     self.terminated = True
    #     return self.state, self.reward(), self.terminated # New-state, reward, done
    
    # def step_continuous(self, action, dt):
    #     """ Returns: state, reward, done """
    #     # We assume for now that the action is a continuous value between -1 and 1
    #     angle = self.state[4]
    #     min_force = -30
    #     max_force = 30
    #     # Interpolate force between -30 and 30 given the action
    #     force = min_force + (action[0][0]+1)/2 * (max_force-min_force)
    #     self.update(np.array([force, 0]), 9.8, dt)

    #     # Check termination:
    #     abs_pos = np.abs(self.state[0])
    #     abs_angle = np.abs(self.state[4]-np.pi/2)
    #     # if abs_pos > self.x_lim or abs_angle > self.angle_lim: # Terminate if outside limits
    #     #     self.terminated = True
    #     if abs_pos > self.x_lim: # Terminates if outside x limits
    #         self.terminated = True
    #     return self.state, self.reward(), self.terminated # New-state, reward, done
        
    
    def animate(self, agent, continuous=False):
        import pygame as pg

        pg.init() #initialize pygame

        #FPS
        # FPS = 60 # frames per second setting
        fpsClock = pg.time.Clock()

        dt = 0.025 # 1 / FPS # Seconds Per Frame = 1 / Frames Per Second


        # Set up the drawing window
        width = 700
        height = 600
        screen_dim = [width, height]
        screen = pg.display.set_mode(screen_dim)

        agent.eps = 0.0

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


