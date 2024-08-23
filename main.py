
from pendulum import Pendulum
from models import REINFORCEAgent, DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import custom_object_scope
import os

from dataModels import Config, read_data_from_yaml

config = read_data_from_yaml('InputParameters.yaml', Config)


# ===== Run parameters ===== #
run_type = config.run.type # 'train' or 'test'
dt = config.run.dt # Time step in seconds
num_episodes = config.run.num_episodes # Number of episodes to run
episode_time = config.run.episode_time # Maximum episode time in seconds
batch_size = config.run.batch_size # Mini-batch size for training
episode_steps = int(episode_time/dt) # Number of time steps in an episode

# =====  Model parameters ===== #
model_type = config.model.type # 'REINFORCE' or 'DQN'
discount_factor = config.model.discount_factor # Discount factor for future rewards
learning_rate = config.model.learning_rate # Learning rate for the optimizer
hidden_layer_sizes = config.model.layer_sizes # List specifying the number of neurons in each layer
# -- DQN parameters -- #
epsilon_init = config.model.DQN_params.epsilon_init # Initial epsilon value (for epsilon-greedy policy)
epsilon_min = config.model.DQN_params.epsilon_min # Minimum epsilon value
epsilon_decay = config.model.DQN_params.epsilon_decay # Epsilon decay rate

# ===== Pendulum environment parameters ===== #
length = config.pendulum_env.length # Length of the pendulum (m)
mass_base = config.pendulum_env.mass_base # Mass of the base (kg)
mass_bob = config.pendulum_env.mass_bob # Mass of the bob (kg)

# Initialise the pendulum environment
env = Pendulum(length, mass_base, mass_bob)

#  Initialise the agent
if model_type == 'REINFORCE':
    agent = REINFORCEAgent((5,), 2, hidden_layer_sizes, discount_factor, learning_rate)
    # Load initial weights if specified
    if config.model.init_from_weights.enable:
        weights_path = os.path.join('weights', config.model.init_from_weights.file_name)
        agent.load_weights(weights_path)

elif model_type == 'DQN':
    agent = DQNAgent((5,), 2, hidden_layer_sizes, discount_factor, learning_rate, epsilon_init, epsilon_min, epsilon_decay)
    # Load initial weights if specified
    if config.model.init_from_weights.enable:
        weights_path = os.path.join('weights', config.model.init_from_weights.file_name)
        agent.load_weights(weights_path)




if run_type == 'train': # Run the training loop
    if model_type == 'REINFORCE':
        for e in range(num_episodes):
            state = env.reset() # Reset the environment
            state = agent.get_state_representation(state) # Get state representation
            
            ep = [] # List of tuples (state, action, reward)
            for time_step in range(episode_steps): # Play episode
                action = agent.act(state) # Get action from agent
                next_state, reward, done = env.step(action, dt) # Take action in environment
                next_state = agent.get_state_representation(next_state) # Get state representation
                reward = reward if not done else -5 # Penalize termination
                ep.append((state, action, reward)) # Add state, action, reward to episode history
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {time_step}")
                    break
                state = next_state # Update state
            
            # Compute G for each time step in episode
            R = [reward for state, action, reward in ep]
            G = 0
            for i, r in enumerate(R[::-1]):
                i = len(R)-i-1 # Reverse index
                G = r+agent.gamma*G # Discounted reward
                ep[i] = (ep[i][0], ep[i][1], ep[i][2], G, i) # Replace reward with G and add time step

            for state, action, reward, G, time_step in ep: # Add episode to memory
                agent.remember(state, action, reward, G, time_step)

            if len(agent.memory) > batch_size:
                for i in range(len(agent.memory)//batch_size):
                    agent.replay(batch_size)

            if config.model.save_weights.enable: # Save weights
                if e%config.model.save_weights.save_frequency == 0 and e != 0:
                    agent.save(os.path.join('weights', config.model.save_weights.file_name))

    elif model_type == 'DQN':
        for e in range(num_episodes):
            state = env.reset() # Reset the environment
            state = agent.get_state_representation(state) # Get state representation
            
            for time_step in range(episode_steps):
                action = agent.act(state)
                next_state, reward, done = env.step(action, dt)
                next_state = agent.get_state_representation(next_state)
                reward = reward if not done else -5
                agent.remember(state, action, reward, next_state, done) # TODO: Adjust angle to be close to 0 instead of pi/2, maybe this will help
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {time_step}, eps: {agent.eps:.2}")
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                state = next_state
            if config.model.save_weights.enable:
                agent.save(os.path.join('weights', config.model.save_weights.file_name))

elif run_type == 'test': # Run test
    env.reset()
    env.animate(agent)

