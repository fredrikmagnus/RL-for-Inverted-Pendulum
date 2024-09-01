from pendulum import Pendulum
from models import REINFORCEAgent, DQNAgent, ActorCritic
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

elif model_type == 'ActorCritic':
    agent = ActorCritic(config)
    # Load initial weights if specified
    if config.model.init_from_weights.enable:
        weights_path = os.path.join('weights', config.model.init_from_weights.file_name)
        agent.load_weights(weights_path)

def run_episode(env, agent):
    states = []
    policies = []
    values = []

    state = env.reset(deterministic=True)
    state = agent.get_state_representation(state)

    done = False
    while not done:
        
        
        # Get the policy and value from the agent's model
        policy, value = agent.model.predict(state, verbose=0)
        
        # Store the policy and value
        policies.append(policy[0])  # Store the policy (action probabilities)
        values.append(value[0])  # Store the value prediction
        
        # Choose an action based on the policy
        action = np.random.choice(np.arange(len(policy[0])), p=policy[0])
        
        # Take the action in the environment
        next_state, reward, done = env.step(action, dt=dt)
        
        # Store the current state
        states.append(next_state.copy())
        next_state = agent.get_state_representation(next_state)
        
        # Update the current state
        state = next_state

        # Break the loop if the episode is done
        if done:
            break
    
    return states, policies, values

# states, policies, values = run_episode(env, agent)



# angle = [state[4]*1.80/np.pi for state in states]
# x_pos = [state[0] for state in states]

# # Plot the angle and value prediction over time
# plt.figure()
# plt.plot(angle, label='Angle')
# plt.plot(np.sin(angle), label='Angle sin')
# plt.plot(x_pos, label='X position')
# plt.plot(values, label='Value')
# plt.xlabel('Time step')
# plt.ylabel('Angle/Value')
# plt.legend()
# plt.show()




# Plot the actor critic value function as a function of the angle
def plot_value_function(agent):
    angle = np.linspace(0, np.pi, 100)
    value = np.zeros_like(angle)
    for i, a in enumerate(angle):
        state = np.array([0, 0, np.cos(a), np.sin(a), 0])
        state = np.reshape(state, (1, 5))
        if model_type == 'ActorCritic':
            # _, value[i] = agent.model.predict(state, verbose=0)
            value[i] = agent.critic.predict(state, verbose=0)
        elif model_type == 'DQN':
            value[i] = np.max(agent.model.predict(state, verbose=0))
            print(value[i])
    plt.plot(angle, value)
    plt.xlabel('Angle (rad)')
    plt.ylabel('Value')
    plt.title('Value function')
    plt.show()

plot_value_function(agent)