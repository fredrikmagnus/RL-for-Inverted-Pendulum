
from pendulum2 import Pendulum
from models2 import REINFORCEAgent, DQNAgent, DDPGAgent
from dataModels2 import Config, read_data_from_yaml



config = read_data_from_yaml('InputParameters2.yaml', Config)

run_params = config.run
pendulum_params = config.pendulum
model_params = config.model

env = Pendulum(config=pendulum_params)

if model_params.type == 'REINFORCE':
    agent = REINFORCEAgent(config=model_params)
elif model_params.type == 'DQN':
    agent = DQNAgent(config=model_params)
elif model_params.type == 'DDPG':
    agent = DDPGAgent(config=model_params)

if run_params.type == 'train':
    agent.train(env, config)
elif run_params.type == 'test':
    env.reset()
    env.animate(agent)

# # ===== Run parameters ===== #
# run_type = config.run.type # 'train' or 'test'
# dt = config.run.dt # Time step in seconds
# num_episodes = config.run.num_episodes # Number of episodes to run
# episode_time = config.run.episode_time # Maximum episode time in seconds
# batch_size = config.run.batch_size # Mini-batch size for training
# episode_steps = int(episode_time/dt) # Number of time steps in an episode

# # =====  Model parameters ===== #
# model_type = config.model.type # 'REINFORCE' or 'DQN'
# discount_factor = config.model.discount_factor # Discount factor for future rewards
# learning_rate = config.model.learning_rate # Learning rate for the optimizer
# hidden_layer_sizes = config.model.layer_sizes # List specifying the number of neurons in each layer
# # -- DQN parameters -- #
# epsilon_init = config.model.DQN_params.epsilon_init # Initial epsilon value (for epsilon-greedy policy)
# epsilon_min = config.model.DQN_params.epsilon_min # Minimum epsilon value
# epsilon_decay = config.model.DQN_params.epsilon_decay # Epsilon decay rate

# # ===== Pendulum environment parameters ===== #
# length = config.pendulum_env.length # Length of the pendulum (m)
# mass_base = config.pendulum_env.mass_base # Mass of the base (kg)
# mass_bob = config.pendulum_env.mass_bob # Mass of the bob (kg)

# # Initialise the pendulum environment
# env = Pendulum(length, mass_base, mass_bob)
# continous = False
# #  Initialise the agent
# if model_type == 'REINFORCE':
#     agent = REINFORCEAgent((5,), 2, hidden_layer_sizes, discount_factor, learning_rate)
#     # Load initial weights if specified
#     if config.model.init_from_weights.enable:
#         weights_path = os.path.join('weights', config.model.init_from_weights.file_name)
#         agent.load_weights(weights_path)

# elif model_type == 'DQN':
#     agent = DQNAgent((5,), 2, hidden_layer_sizes, discount_factor, learning_rate, epsilon_init, epsilon_min, epsilon_decay)
#     # Load initial weights if specified
#     if config.model.init_from_weights.enable:
#         weights_path = os.path.join('weights', config.model.init_from_weights.file_name)
#         agent.load_weights(weights_path)

# elif model_type == 'DDPG':
#     agent = DDPGAgent((5,), 1, learning_rate*0.01, learning_rate, discount_factor, 0.95, 512, 16)
#     continous = True
#     # Load initial weights if specified
#     if config.model.init_from_weights.enable:
#         weights_path = os.path.join('weights', config.model.init_from_weights.file_name)
#         agent.load_weights(weights_path)


# if run_type == 'train': # Run the training loop
#     agent.train(env, config)

# elif run_type == 'test': # Run test
#     env.reset(deterministic=False, down=True)
#     env.animate(agent, continuous=continous)





