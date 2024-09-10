from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Literal
import ruamel.yaml

class TerminationParams(BaseModel):
    angle_limit: Optional[float] = Field(None, description="Angle limit for the pendulum in degrees. Set to None for no limit.")
    x_limit: Optional[float] = Field(None, description="X limit for the pendulum in meters. Set to None for no limit.")
    time_limit: Optional[float] = Field(None, description="Time limit for the episode in seconds. Set to None for no limit.")
    termination_penalty: float = Field(-20, description="Penalty for terminating the episode early.")


class LoggingParams(BaseModel):
    enable: bool = Field(True, description="Enable logging.")
    file_path: str = Field("logs/DDPG.log", description="File path to save log to.")

class PendulumParams(BaseModel):
    deterministic_initial_state: bool = Field(False, description="Use a deterministic initial state or add noise.")
    initial_state: Literal['up', 'down', 'random'] = Field('up', description="Initial state of the pendulum (up, down or random).")
    time_step: float = Field(0.025, description="Time step for the simulation.")
    mass_bob: float = Field(1.0, description="Mass of the bob.")
    mass_base: float = Field(1.0, description="Mass of the base.")
    length: float = Field(1.0, description="Length of the pendulum.")
    damping_coefficient: float = Field(0.1, description="Damping coefficient for the pendulum.")
    gravity: float = Field(9.81, description="Acceleration due to gravity.")
    termination: TerminationParams = TerminationParams()
    logging: LoggingParams = LoggingParams()

class InitFromWeights(BaseModel):
    enable: bool = Field(False, description="Enable loading initial weights from a file.")
    file_path: Optional[str] = Field(None, description="Filename for loading initial weights.")

class SaveWeights(BaseModel):
    enable: bool = Field(False, description="Enable saving model weights to a file.")
    file_path: Optional[str] = Field(None, description="Filename for saving model weights.")
    save_frequency: Optional[int] = Field(50, description="Save weights every n episodes.")

class IOParameters(BaseModel):
    init_from_weights: InitFromWeights = InitFromWeights()
    save_weights: SaveWeights = SaveWeights()

class EpsilonParams(BaseModel):
    epsilon_init: float = Field(1, description="Initial epsilon value (for epsilon-greedy policy).")
    epsilon_min: float = Field(0.1, description="Minimum epsilon value.")
    epsilon_decay: float = Field(0.9, description="Epsilon decay rate.")

class DQNParams(BaseModel):
    hidden_layer_sizes: List[int] = Field([16, 16], description="Number of neurons in each layer.")
    learning_rate: float = Field(1e-3, description="Learning rate for the optimizer.")
    num_actions: int = Field(2, description="Number of actions in the action space. (2 for left and right, 3 for left, right and no action)")
    polyak: float = Field(0.95, description="Polyak averaging parameter for target network updates.")
    epsilon: EpsilonParams = EpsilonParams()

class REINFORCEParams(BaseModel):
    hidden_layer_sizes: List[int] = Field([16, 16], description="Number of neurons in each layer.")
    learning_rate: float = Field(1e-3, description="Learning rate for the optimizer.")
    num_actions: int = Field(2, description="Number of actions in the action space. (2 for left and right, 3 for left, right and no action)")
    
class NoiseParams(BaseModel):
    enable: bool = Field(True, description="Enable addition of noise to action for exploration.")
    theta: float = Field(0.15, description="Theta parameter for the Ornstein-Uhlenbeck noise process.")
    sigma: float = Field(0.2, description="Sigma parameter for the Ornstein-Uhlenbeck noise process.")
    noise_scale_initial: float = Field(1, description="Initial scale for noise.")
    noise_decay: float = Field(1, description="Decay rate for noise scale parameter.")

class DDPGActorParams(BaseModel):
    hidden_layer_sizes: List[int] = Field([16, 16], description="Number of neurons in each layer.")
    learning_rate: float = Field(1e-5, description="Learning rate for the optimizer.")
    polyak: float = Field(0.95, description="Polyak averaging parameter for target network updates.")

    ornstein_uhlenbeck_noise: NoiseParams = NoiseParams()

class DDPGCriticParams(BaseModel):
    state_input_layer_sizes: List[int] = Field([8], description="Number of neurons in layers receiving the state input.")
    action_input_layer_sizes: List[int] = Field([8], description="Number of neurons in layers receiving the action input.")
    combined_layer_sizes: List[int] = Field([16, 8, 8], description="Number of neurons in each layer for combined state and action input.")
    learning_rate: float = Field(1e-3, description="Learning rate for the optimizer.")
    polyak: float = Field(0.95, description="Polyak averaging parameter for target network updates.")

class DDPGParams(BaseModel):
    actor: DDPGActorParams = DDPGActorParams()
    critic: DDPGCriticParams = DDPGCriticParams()

class ModelParams(BaseModel):
    type: Literal['DQN', 'REINFORCE', 'DDPG'] = Field("DDPG", description="Type of model to use.")
    memory_size: int = Field(1000, description="Size of the replay memory buffer.")
    batch_size: int = Field(16, description="Mini-batch size for training.")
    discount_factor: float = Field(0.95, description="Discount factor for future rewards.")
    force_magnitude: float = Field(20, description="Magnitude of the force applied to the base.")

    IO_parameters: IOParameters = IOParameters()
    DQN: DQNParams = DQNParams()
    REINFORCE: REINFORCEParams = REINFORCEParams()
    DDPG: DDPGParams = DDPGParams()

class RunParams(BaseModel):
    type: Literal['train', 'test'] = Field("test", description="Run mode: 'train' or 'test'.")
    num_episodes: int = Field(512, description="Number of episodes to run.")
    batch_size: int = Field(16, description="Mini-batch size for training.")

class AnimateFromLogParams(BaseModel):
    enable: bool = Field(False, description="Enable animation from file, else run simulation and animate.")
    episode: int = Field(0, description="Log index of episode to plot.")
    file_path: str = Field("", description="File path to load log from.")

class AnimateParams(BaseModel):
    x_lim: float = Field(2, description="x limit for the animation (in meters).")
    y_lim: float = Field(2, description="y limit for the animation (in meters).")
    save_path: Optional[str] = Field(None, description="Path to save the animation. None for no save.")
    from_log: AnimateFromLogParams = AnimateFromLogParams()

class Config(BaseModel):
    run: RunParams = RunParams()
    animate: AnimateParams = AnimateParams()
    pendulum: PendulumParams = PendulumParams()
    model: ModelParams = ModelParams()

def read_data_from_yaml(full_file_path, data_class):
    with open(full_file_path, 'r') as stream:
        yaml = ruamel.yaml.YAML(typ='safe', pure=True)
        yaml_str = yaml.load(stream)
    return data_class(**yaml_str)
