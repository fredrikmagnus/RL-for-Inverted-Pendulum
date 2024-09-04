
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
    env.animate(agent)






