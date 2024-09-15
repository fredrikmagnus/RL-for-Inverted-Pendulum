import numpy as np
import random
import tensorflow as tf
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Concatenate, Input
from keras.optimizers import Adam
from collections import deque
from keras.losses import MeanSquaredError
import os
from dataModels import ModelParams, DQNParams, REINFORCEParams, DDPGParams, Config, RunParams, DDPGActorParams, DDPGCriticParams
from pendulum import Pendulum

class DQNAgent:
    def __init__(self, config : ModelParams) -> None:
        self.state_shape = (5,) # State shape (x_pos, x_vel, sin(angle), cos(angle), angle_vel)
        self.action_size = config.DQN.num_actions # Number of actions (2 for left and right, 3 for left, right and no action)

        self.memory = deque(maxlen = config.memory_size) # Circular buffer used to store transitions to learn from.
        self.gamma = config.discount_factor # Discount rate
        self.eps_init = config.DQN.epsilon.epsilon_init # Epsilon for epsilon-greedy policy
        self.epsilon_min = config.DQN.epsilon.epsilon_min # Minimum epsilon value
        self.epsilon_decay = config.DQN.epsilon.epsilon_decay # Epsilon decay rate
        self.force_magnitude = config.force_magnitude # Magnitude of the force applied to the base

        self.polyak = config.DQN.polyak # Polyak averaging parameter for target network updates
        self.model = self.build_model(config.DQN)
        self.target_model = self.build_model(config.DQN)
        if config.IO_parameters.init_from_weights.enable:
            self.load_weights(config.IO_parameters.init_from_weights.file_path)
        else:
            self.target_model.set_weights(self.model.get_weights())


    def build_model(self, config : DQNParams):
        model = Sequential()
        model.add(Dense(config.hidden_layer_sizes[0], activation='relu', input_shape = self.state_shape)) # Input layer with 5 neurons (state representation)
        for layer_size in config.hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu')) # Hidden layers with specified number of neurons
        model.add(Dense(self.action_size, activation='linear')) # Output layer with 2 neurons (one for each action)
        model.compile(loss='mse', optimizer=Adam(learning_rate=config.learning_rate)) # Compile model with mse loss and Adam optimizer
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.eps_init:
            action = random.randrange(self.action_size) # Random action (0:left or 1:right)
        else:
            action_values = self.model.predict(state, verbose=0)
            action = np.argmax(action_values[0]) # Greedy action (0:left or 1:right)
        
        force = -self.force_magnitude if action == 0 else self.force_magnitude # Convert action to force value 
        if self.action_size == 3:
            force = 0 if action == 2 else force

        return np.array([force, 0]), action # Return force vector and action taken
        
    
    def update(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        print("Running replay...")

        # Initialize arrays for states, targets_f
        states = np.zeros((batch_size, self.state_shape[0]))
        targets_f = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.target_model.predict(next_state, verbose=0)[0])

            # Model prediction of future return
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target  # Update the estimate of future return for the action taken

            # Update states and targets_f arrays
            states[i] = state
            targets_f[i] = target_f[0]

        # One gradient step for entire minibatch
        self.model.fit(states, targets_f, epochs=1, verbose=0)

        if self.eps_init > self.epsilon_min:
            self.eps_init *= self.epsilon_decay
        
        self.update_target(self.target_model, self.model, self.polyak)

    def get_state_representation(self, state):
        state = state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
        state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]) # Convert angle to sin and cos
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state
    
    def train(self, env : Pendulum, config : Config):
        num_episodes = config.run.num_episodes # Number of episodes to run
        # episode_time = config.run.episode_time # Maximum episode time in seconds
        batch_size = config.model.batch_size # Mini-batch size for training
        # episode_steps = int(episode_time/env.time_step) # Number of time steps in an episode

        for e in range(num_episodes):
            state = env.reset() # Reset the environment
            state = self.get_state_representation(state) # Get state representation
            
            time_step = 0
            done = False
            while not done: # Play episode
                force, action = self.act(state) # Get action from agent
                next_state, reward, done = env.step(force) # Take action in environment
                next_state = self.get_state_representation(next_state) # Get state representation
                self.remember(state, action, reward, next_state, done) # Add transition to memory
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {time_step}, eps: {self.eps_init:.2}")
                    break
                if len(self.memory) > batch_size: # Train agent
                    self.update(batch_size)
                state = next_state
                time_step += 1
            if config.model.IO_parameters.save_weights.enable:
                if e%config.model.IO_parameters.save_weights.save_frequency == 0 and e != 0:
                    self.save(config.model.IO_parameters.save_weights.file_path)


    def update_target(self, target_model, model, polyak):
        # Get the weights of both models
        target_weights = target_model.get_weights()
        weights = model.get_weights()
        
        # Update target network weights
        new_weights = []
        for target_weight, weight in zip(target_weights, weights):
            # Perform the soft update
            updated_weight = polyak * target_weight + (1 - polyak) * weight
            new_weights.append(updated_weight)
        
        # Set the new weights to the target model
        target_model.set_weights(new_weights)

    def load_weights(self, filepath):
        base, extension = os.path.splitext(filepath)
        self.model.load_weights(filepath)
        self.target_model.load_weights(base + '_target' + extension)
    
    def save(self, filepath):
        base, extension = os.path.splitext(filepath)
        self.model.save(filepath)
        self.target_model.save(base + '_target' + extension)



class REINFORCEAgent:
    def __init__(self, config : ModelParams) -> None:
        self.state_shape = (5,) # State shape (x_pos, x_vel, sin(angle), cos(angle), angle_vel)
        self.action_size = config.REINFORCE.num_actions # Number of actions (2 for left and right, 3 for left, right and no action)
        self.hidden_layer_sizes = config.REINFORCE.hidden_layer_sizes # List specifying the number of neurons in each layer
        self.learning_rate = config.REINFORCE.learning_rate # Learning rate for the optimizer
        self.memory = deque(maxlen = config.memory_size) # Used to store transitions to learn from.
        self.force_magnitude = config.force_magnitude # Magnitude of the force applied to the base
        self.gamma = config.discount_factor # Discount rate
        self.model = self.build_model()
        if config.IO_parameters.init_from_weights.enable:
            self.load_weights(config.IO_parameters.init_from_weights.file_path)

        self.baseline = 0 # Baseline for the reward (running average of episodic reward)

    def custom_loss(self, y_true, y_pred):
        # We now redefine the loss-function to avoid having to store the rewards in a class.
        # We can do this by incorporating the reward and gamma^t in the y_true vector.
        # Earlier we had y_true as a vector of the one-hot encoding of the selected actions. 
        # y_true was then multiplied by G*gamma^t. We now multiply G*gamma^t with y_true before passing it to the loss function.
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1-1e-7) 
        log_probs = y_true * tf.math.log(y_pred_clipped) # Log(pi(s)) * G*gamma^t (one-hot-encoded for the selected action)

        return -tf.reduce_sum(log_probs)

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_layer_sizes[0], activation='relu', input_shape = self.state_shape))
        for layer_size in self.hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=self.custom_loss, optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, G, time_step):
        self.memory.append((state, action, reward, G, time_step))

    def act(self, state):
        act_probs = self.model.predict(state, verbose=0)[0] # Get action probabilities
        action = np.random.choice(np.arange(len(act_probs)), p=act_probs) # Sample action from action probabilities
        force = -self.force_magnitude if action == 0 else self.force_magnitude # Convert action to force value
        if self.action_size == 3:
            force = 0 if action == 2 else force # Do nothing if action is 2
        return np.array([force, 0]), action # Return force vector and action taken
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size) # Sample a mini-batch from memory
        print("Running replay...")
        states = np.zeros((len(minibatch), self.state_shape[0]))  # Create array to store states
        y_true = np.zeros((len(minibatch), self.action_size)) # Create array to store y_true
        for i, (state, action, reward, G, time_step) in enumerate(minibatch): # Loop over mini-batch
            states[i] = state # Add state to states
            y_true[i][action] = G*tf.math.pow(self.gamma, time_step) # Multiply G*gamma^t with one-hot-encoded action
        self.model.fit(states, y_true, verbose = 0) # Update weights

    def get_state_representation(self, state):
        state = state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
        state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]) # Convert angle to sin and cos signal
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state
    
    def train(self, env : Pendulum, config : Config):
        num_episodes = config.run.num_episodes # Number of episodes to run
        batch_size = config.model.batch_size # Mini-batch size for training

        for e in range(num_episodes): # Loop over episodes
            state = env.reset() # Reset the environment
            state = self.get_state_representation(state) # Get state representation
            
            ep = [] # List of tuples (state, action, reward)
            time_step = 0
            done = False
            R = 0
            while not done: # Play episode
                force, action = self.act(state) # Get action from agent
                next_state, reward, done = env.step(force) # Take action in environment
                R += reward # Add reward to total reward
                next_state = self.get_state_representation(next_state) # Get state representation
                ep.append((state, action, reward)) # Add state, action, reward to episode history
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {R}, advantage: {R-self.baseline}, steps: {time_step}")
                    break
                state = next_state # Update state
                time_step += 1 # Increment time step
            
            # Compute G for each time step in episode
            R = [reward for state, action, reward in ep]
            G = 0
            for i, r in enumerate(R[::-1]):
                i = len(R)-i-1 # Reverse index
                G = r+self.gamma*G # Discounted reward
                ep[i] = (ep[i][0], ep[i][1], ep[i][2], G-self.baseline, i) # Replace reward with G-baseline and add time step

            # self.baseline += (G-self.baseline)/(e+1) # Update baseline with running average of episodic reward
            # self.baseline = 0.9*self.baseline + 0.1*G # Update baseline with exponential moving average of episodic reward

            for state, action, reward, G, time_step in ep: # Add episode to memory
                self.remember(state, action, reward, G, time_step)

            if len(self.memory) > batch_size:
                for i in range(len(self.memory)//batch_size):
                    self.replay(batch_size)

            if config.model.IO_parameters.save_weights.enable:
                if e%config.model.IO_parameters.save_weights.save_frequency == 0 and e != 0:
                    self.save(config.model.IO_parameters.save_weights.file_path)
            

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def save(self, filepath):
        self.model.save_weights(filepath)



class DDPGAgent:
    def __init__(self, config : ModelParams) -> None:
        self.state_shape = (5,) # State shape (x_pos, x_vel, sin(angle), cos(angle), angle_vel)
        self.action_size = 1
        self.memory = deque(maxlen=config.memory_size)
        self.batch_size = config.batch_size
        self.gamma = config.discount_factor

        self.polyak_critic = config.DDPG.critic.polyak # Polyak averaging parameter for critic target network
        self.polyak_actor = config.DDPG.actor.polyak # Polyak averaging parameter for actor target network


        self.force_magnitude = config.force_magnitude

        self.actor_model, self.critic_model = self.build_models(config=config.DDPG)
        self.target_actor_model, self.target_critic_model = self.build_models(config=config.DDPG)
        if config.IO_parameters.init_from_weights.enable:
            self.load_weights(config.IO_parameters.init_from_weights.file_path)
        else:
            self.target_actor_model.set_weights(self.actor_model.get_weights())
            self.target_critic_model.set_weights(self.critic_model.get_weights())

        # Initialize noise for exploration
        self.enable_noise = config.DDPG.actor.ornstein_uhlenbeck_noise.enable
        self.noise_scaling = config.DDPG.actor.ornstein_uhlenbeck_noise.noise_scale_initial
        self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1), sigma=config.DDPG.actor.ornstein_uhlenbeck_noise.sigma, theta=config.DDPG.actor.ornstein_uhlenbeck_noise.theta, dt=0.05)

    def build_models(self, config : DDPGParams):
        # Build the actor network
        state_input = Input(shape=self.state_shape)
        h = Dense(config.actor.hidden_layer_sizes[0], activation='relu')(state_input)
        for layer_size in config.actor.hidden_layer_sizes[1:]:
            h = Dense(layer_size, activation='relu')(h)

        # Initialise weights between -3e-3 and 3e-3 to avoid vanishing gradients due to tanh activation
        output_init = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)
        output = Dense(self.action_size, activation='tanh', kernel_initializer=output_init)(h)

        actor_model = Model(inputs=state_input, outputs=output)
        actor_model.compile(optimizer=Adam(config.actor.learning_rate))


        # Build the critic network
        # State input layers
        state_input = Input(shape=self.state_shape)
        state_h = Dense(config.critic.state_input_layer_sizes[0], activation='relu')(state_input)
        for layer_size in config.critic.state_input_layer_sizes[1:]:
            state_h = Dense(layer_size, activation='relu')(state_h)
        
        # Action input layers
        action_input = Input(shape=(self.action_size,))
        action_h = Dense(config.critic.action_input_layer_sizes[0], activation='relu')(action_input)
        for layer_size in config.critic.action_input_layer_sizes[1:]:
            action_h = Dense(layer_size, activation='relu')(action_h)

        # Combine state and action inputs
        concat = Concatenate()([state_h, action_h])
        h = Dense(config.critic.combined_layer_sizes[0], activation='relu')(concat)
        for layer_size in config.critic.combined_layer_sizes[1:]:
            h = Dense(layer_size, activation='relu')(h)

        # Output layer
        output = Dense(1, activation='linear')(h)
        critic_model = Model(inputs=[state_input, action_input], outputs=output)
        critic_model.compile(optimizer=Adam(config.critic.learning_rate), loss='mse')

        return actor_model, critic_model
    
    def get_state_representation(self, state):
        state = state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
        state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]) # Convert angle to sin and cos signal
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        action = self.actor_model.predict(state, verbose=0)
        
        if self.enable_noise:
            noise = self.noise_scaling*self.noise()
            action = action + noise # Add noise for exploration
        
        action = np.clip(action, -1, 1)

        force = action.squeeze()*self.force_magnitude # Convert action to force value
        return np.array([force, 0]), action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        print("Running replay...")
        # Sample a minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Separate minibatch into separate arrays for batch processing
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # Ensure correct shape of next_states
        states = np.squeeze(states)  # Remove extra dimension if it exists
        next_states = np.squeeze(next_states)  # Remove extra dimension if it exists
        actions = np.squeeze(actions)  # Remove extra dimension if it exists

        # Compute target Q-values for the minibatch
        target_actions = self.target_actor_model.predict(next_states, verbose=0)
        future_qs = self.target_critic_model.predict([next_states, target_actions], verbose=0).flatten()
        target_qs = rewards + self.gamma * future_qs * (1 - dones)  # Handle terminal states with (1 - dones)

        # Train critic using the entire mini-batch
        self.critic_model.train_on_batch([states, actions], target_qs)

        # Train actor using the entire mini-batch
        with tf.GradientTape(persistent=True) as tape:
            # Generate actions from the actor network for the entire mini-batch
            actions_for_training = self.actor_model(states, training=True)

            # Compute Q-values for the generated actions
            critic_value = self.critic_model([states, actions_for_training], training=True)

        # Compute the gradient of the Q-value with respect to the actions
        action_grads = tape.gradient(critic_value, actions_for_training)

        # Compute the gradient of the actor's weights using the chain rule
        actor_grads = tape.gradient(actions_for_training, self.actor_model.trainable_variables, output_gradients=-action_grads)

        # print(f"actor_grads: {actor_grads}")


        # Apply the gradients to the actor's weights
        # weights_prev = self.actor_model.get_weights()

        self.actor_model.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # actions_for_training_new = self.actor_model(states, training=True)

        # weights_new = self.actor_model.get_weights()




        # Delete the persistent tape manually
        del tape

        # Update target networks
        self.update_target(self.target_actor_model, self.actor_model, self.polyak_actor)
        self.update_target(self.target_critic_model, self.critic_model, self.polyak_critic)

    def train(self, env : Pendulum, config : Config):
        num_episodes = config.run.num_episodes # Number of episodes to run
        
        for e in range(num_episodes):
            state = env.reset()
            state = env.get_state_representation(state)
            episode_reward = 0
            time_step = 0
            done = False

            while not done:
                force, action = self.act(state)
                next_state, reward, done = env.step(force)
                next_state = env.get_state_representation(next_state)
                episode_reward += reward
                self.remember(state, action, reward, next_state, done)
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {episode_reward}, steps: {time_step}, noise: {self.noise_scaling:.2}")
                    break
                self.update()
                state = next_state
                time_step += 1

            if config.model.IO_parameters.save_weights.enable:
                if e%config.model.IO_parameters.save_weights.save_frequency == 0 and e != 0:
                    self.save(config.model.IO_parameters.save_weights.file_path)

            self.noise_scaling *= config.model.DDPG.actor.ornstein_uhlenbeck_noise.noise_decay

    def update_target(self, target_model, model, polyak):
        # Get the weights of both models
        target_weights = target_model.get_weights()
        weights = model.get_weights()
        
        # Update target network weights
        new_weights = []
        for target_weight, weight in zip(target_weights, weights):
            # Perform the soft update
            updated_weight = polyak * target_weight + (1 - polyak) * weight
            new_weights.append(updated_weight)
        
        # Set the new weights to the target model
        target_model.set_weights(new_weights)

    def load_weights(self, filepath):
        base, extension = os.path.splitext(filepath)
        self.actor_model.load_weights(base + '_actor' + extension)
        self.critic_model.load_weights(base + '_critic' + extension)

        self.target_actor_model.load_weights(base + '_target_actor' + extension)
        self.target_critic_model.load_weights(base + '_target_critic' + extension)
    
    def save(self, filepath):
        base, extension = os.path.splitext(filepath)
        self.actor_model.save(base + '_actor' + extension)
        self.critic_model.save(base + '_critic' + extension)

        self.target_actor_model.save(base + '_target_actor' + extension)
        self.target_critic_model.save(base + '_target_critic' + extension)


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for exploration in DDPG (as suggested in https://arxiv.org/pdf/1509.02971)
    """
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_prev = x0 if x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
