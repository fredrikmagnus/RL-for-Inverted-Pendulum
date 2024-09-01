import numpy as np
import matplotlib.pyplot as plt
import random

class Pendulum:
    def __init__(self, length, m1, m2) -> None:
        self.length = length # Length of the pendulum
        self.m1 = m1 # Mass of the base
        self.m2 = m2 # Mass of the pendulum
        self.x_lim = 7 # min/max x_pos
        self.angle_lim = np.pi/6 # min/max angle
        self.state = None # State vector (x_pos, y_pos, x_vel, y_vel, angle, ang_vel)
        self.terminated = False

    def derivate_state(self, state, force, g):
        vel = [state[2], state[3]]
        angle = state[4]
        ang_vel = state[5]
        u = np.array([np.cos(angle), np.sin(angle)])
        fb = np.dot(force, u)*u + np.array([0, -g])
        base_acc = (force)/self.m1
        ang_acc = np.cross(force-fb, self.length*u)/(self.m2*self.length**2)
        state_dot = np.array([vel[0], vel[1], base_acc[0], base_acc[1], ang_vel, ang_acc])
        return state_dot
        
    def update(self, force, g, dt):
        # self.state += self.derivate_state(self.state, force, g)*dt # Euler integration
        # Runge-Kutta 4 integration
        k1 = self.derivate_state(self.state, force, g)*dt
        k2 = self.derivate_state(self.state + k1/2, force, g)*dt
        k3 = self.derivate_state(self.state + k2/2, force, g)*dt
        k4 = self.derivate_state(self.state + k3, force, g)*dt
        self.state += (k1 + 2*k2 + 2*k3 + k4)/6
        
        self.state[4] = self.state[4] % (2*np.pi) # Keep angle between 0 and 2pi

    def reset(self, deterministic=False):
        if deterministic:
            self.state = np.array([0, 0, 0, 0, np.pi/2, 0])
            self.terminated = False
        else:
            # Sets a random initial state which is returned
            # For this example the necessary state-variables is x-pos, x-vel, angle and angle-vel
            x_vel_lim = 2 # The velocity can be between +- 2m/s
            angle_vel_lim = 0.3 # min/max angle vel

            x_pos = random.uniform(-self.x_lim/3, self.x_lim/3)
            x_vel = random.uniform(-x_vel_lim, x_vel_lim)
            angle = random.uniform(-self.angle_lim/3 + np.pi/2, self.angle_lim/3 + np.pi/2) # Start at upright position
            # angle = -np.pi/2 # Start at bottom
            angle_vel = random.uniform(-angle_vel_lim, angle_vel_lim)

            self.state = np.array([x_pos, 0, x_vel, 0, angle, angle_vel]) #(x_pos, y_pos, x_vel, y_vel, angle, ang_vel)
            self.terminated = False
        
        return self.state
    
    # def reward(self):
    #     # Returns a reward based on the current state
    #     return 1
    
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

        x = self.state[0]
        x_lim = self.x_lim
        reward = 1 - abs(x) / x_lim
        return max(reward, 0)

        # # Reward for swinging up
        # angle = self.state[4]
        # R1 = np.sin(angle) # Reward for upright position
        # x_pos = self.state[0]
        # R2 = -0.5*np.abs(x_pos/self.x_lim) # Reward for being close to center
        # return R1 + R2 +1 # Total reward
    
    def step(self, action, dt):
        """ Returns: state, reward, done """
        # We assume for now that the action is either 0 (left) or 1 (right)
        if action == 0: action = -1 # Convert action 0 (left) to -1
        angle = self.state[4]
        min_force = 20
        max_force = 25
        force = lambda theta: min_force+np.abs(theta-np.pi/2)/self.angle_lim * (max_force-min_force) # Force is higher given higher angle error
        self.update(np.array([action*force(angle), 0]), 9.8, dt)
        
        # Check termination:
        abs_pos = np.abs(self.state[0])
        abs_angle = np.abs(self.state[4]-np.pi/2)
        if abs_pos > self.x_lim or abs_angle > self.angle_lim: # Terminate if outside limits
            self.terminated = True
        # if abs_pos > self.x_lim: # Terminates if outside x limits
        #     self.terminated = True
        return self.state, self.reward(), self.terminated # New-state, reward, done
    
    def animate(self, agent):
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
            action = agent.act(state)
            if action == 0: action = -1 # Convert action 0 (left) to -1
            angle = self.state[4]
            min_force = 20
            max_force = 25
            force = lambda theta: min_force+np.abs(theta-np.pi/2)/self.angle_lim * (max_force-min_force) # Force is higher given higher angle error
            
            pos = np.array([self.state[0], self.state[1]])
            angle = self.state[4]
            p_base = np.array([0, height])+60*np.array([1, -1])*(pos + np.array([width/120, height/120])) # base of pendulum (flip y-axis)
            p_end = np.array([0, height])+60*np.array([1, -1])*((pos + np.array([width/120, height/120])) + self.length*np.array([np.cos(angle), np.sin(angle)])) # end of pendulum (flip y-axis)
            pg.draw.line(screen, (0,0,0), p_base, p_end , 2)

            self.update(np.array([action*force(angle), 0]), 9.8, dt)

            pg.display.flip() # update screen?
            fpsClock.tick(1/dt)
            
        # Done! Time to quit.
        pg.quit()


