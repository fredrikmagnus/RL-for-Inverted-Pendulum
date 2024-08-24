from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Optional, Literal
import ruamel.yaml

class InitFromWeights(BaseModel):
    enable: bool = Field(False, description="Enable loading initial weights from a file.")
    file_name: Optional[str] = Field(None, description="Filename for loading initial weights.")

class SaveWeights(BaseModel):
    enable: bool = Field(False, description="Enable saving model weights to a file.")
    file_name: Optional[str] = Field(None, description="Filename for saving model weights.")
    save_frequency: Optional[int] = Field(50, description="Save weights every n episodes")

class DQNParams(BaseModel):
    epsilon_init: float = Field(1, description="Initial epsilon value (for epsilon-greedy policy).")
    epsilon_min: float = Field(0.1, description="Minimum epsilon value.")
    epsilon_decay: float = Field(0.995, description="Epsilon decay rate.")

class ModelParams(BaseModel):
    type: str = Field("REINFORCE", description="Type of model to use, either 'REINFORCE', 'DQN' or 'ActorCritic'.")
    discount_factor: float = Field(0.99, description="Discount factor for future rewards.")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")
    layer_sizes: List[int] = Field([20, 20], description="List specifying the number of neurons in each layer.")
    termination_penalty: float = Field(-5, description="Penalty for termination.")

    init_from_weights: InitFromWeights = InitFromWeights()
    save_weights: SaveWeights = SaveWeights()
    
    DQN_params: DQNParams = DQNParams()
    
    @validator('type')
    def validate_model_type(cls, v):
        if v not in ['REINFORCE', 'DQN', 'ActorCritic']:
            raise ValueError('type must be either "REINFORCE" or "DQN"')
        return v

class RunParams(BaseModel):
    type: str = Field("train", description="Run mode: 'train' or 'test'.")
    num_episodes: int = Field(1000, description="Number of episodes to run.")
    episode_time: float = Field(25, description="Maximum episode time in seconds.")
    dt: float = Field(0.025, description="Time step in seconds.")
    batch_size: int = Field(32, description="Mini-batch size for training.")
    
    @validator('type')
    def validate_run_type(cls, v):
        if v not in ['train', 'test']:
            raise ValueError('type must be either "train" or "test"')
        return v
    
class PendulumEnvParams(BaseModel):
    length: float = Field(1.0, description="Length of the pendulum.")
    mass_base: float = Field(1.0, description="Mass of the base.")
    mass_bob: float = Field(1.0, description="Mass of the bob.")

class Config(BaseModel):
    run: RunParams = RunParams()
    model: ModelParams = ModelParams()
    pendulum_env: PendulumEnvParams = PendulumEnvParams()

def read_data_from_yaml(full_file_path, data_class):
        with open(full_file_path, 'r') as stream:
            yaml = ruamel.yaml.YAML(typ='safe', pure=True)
            yaml_str = yaml.load(stream)

        return data_class(**yaml_str)