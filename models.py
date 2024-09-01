import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Concatenate, Input
from keras.optimizers import Adam
from collections import deque
from keras.losses import MeanSquaredError
import os

class DQNAgent:
    def __init__(self, state_shape, action_size, hidden_layer_sizes, discount_factor, learning_rate, epsilon_init, epsilon_min, epsilon_decay) -> None:
        self.state_shape = state_shape
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes # List specifying the number of neurons in each layer
        self.learning_rate = learning_rate
        self.memory = deque(maxlen = 1000) # Circular buffer with maxlen 1000 (approximately). Used to store transitions to learn from.
        self.gamma = discount_factor # Discount rate
        self.eps = epsilon_init # Epsilon for epsilon-greedy policy
        self.epsilon_min = epsilon_min # Minimum epsilon value
        self.epsilon_decay = epsilon_decay # Epsilon decay rate
        self.model = self.build_model(state_shape, action_size, hidden_layer_sizes, learning_rate)

    def build_model(self, state_shape, action_size, hidden_layer_sizes, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape = state_shape))
        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        print("Running replay...")
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.max(self.model.predict(next_state, verbose=0)[0]) # Bellman equation to compute better estimate of expected future return
            target_f = self.model.predict(state, verbose=0) # Model prediction of future return
            target_f[0][action] = target # Update the estimate of future return for the action taken based on updated better estimate
            self.model.fit(state, target_f, epochs=1, verbose=0) # Update weights using mse
        if self.eps > self.epsilon_min:
            self.eps *= self.epsilon_decay

    def get_state_representation(self, state):
        state = state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
        state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]) # Convert angle to sin and cos
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state
    
    def train(self, env, config):
        dt = config.run.dt # Time step in seconds
        num_episodes = config.run.num_episodes # Number of episodes to run
        episode_time = config.run.episode_time # Maximum episode time in seconds
        batch_size = config.run.batch_size # Mini-batch size for training
        episode_steps = int(episode_time/dt) # Number of time steps in an episode

        for e in range(num_episodes):
            state = env.reset() # Reset the environment
            state = self.get_state_representation(state) # Get state representation
            
            for time_step in range(episode_steps):
                action = self.act(state) # Get action from agent
                next_state, reward, done = env.step(action, dt) # Take action in environment
                next_state = self.get_state_representation(next_state) # Get state representation
                reward = reward if not done else config.model.termination_penalty # Penalize termination
                self.remember(state, action, reward, next_state, done) # Add transition to memory
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {time_step}, eps: {self.eps:.2}")
                    break
                if len(self.memory) > batch_size: # Train agent
                    self.replay(batch_size)
                state = next_state
            if config.model.save_weights.enable:
                self.save(os.path.join('weights', config.model.save_weights.file_name))


    def load_weights(self, filepath):
        self.model = load_model(filepath)
    
    def save(self, filepath):
        self.model.save(filepath)



class REINFORCEAgent:
    def __init__(self, state_shape, action_size, hidden_layer_sizes, discount_factor, learning_rate) -> None:
        self.state_shape = state_shape
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes # List specifying the number of neurons in each layer
        self.learning_rate = learning_rate
        self.memory = deque(maxlen = 1000) # Used to store transitions to learn from.
        self.gamma = discount_factor # Discount rate
        self.model = self.build_model(state_shape, action_size, hidden_layer_sizes, learning_rate)

    def custom_loss(self, y_true, y_pred):
        # We now redefine the loss-function to avoid having to store the rewards in a class.
        # We can do this by incorporating the reward and gamma^t in the y_true vector.
        # Earlier we had y_true as a vector of the one-hot encoding of the selected actions. 
        # y_true was then multiplied by G*gamma^t. We now multiply G*gamma^t with y_true before passing it to the loss function.
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1-1e-7) 
        log_probs = y_true * tf.math.log(y_pred_clipped) # Log(pi(s)) * G*gamma^t (one-hot-encoded for the selected action)

        return -tf.reduce_sum(log_probs)

    def build_model(self, state_shape, action_size, hidden_layer_sizes, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape = state_shape))
        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(action_size, activation='softmax'))
        model.compile(loss=self.custom_loss, optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def remember(self, state, action, reward, G, time_step):
        self.memory.append((state, action, reward, G, time_step))

    def act(self, state):
        act_probs = self.model.predict(state, verbose=0)[0]
        return np.random.choice(np.arange(len(act_probs)), p=act_probs)
    
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
    
    def train(self, env, config):
        dt = config.run.dt # Time step in seconds
        num_episodes = config.run.num_episodes # Number of episodes to run
        episode_time = config.run.episode_time # Maximum episode time in seconds
        batch_size = config.run.batch_size # Mini-batch size for training
        episode_steps = int(episode_time/dt) # Number of time steps in an episode

        for e in range(num_episodes): # Loop over episodes
            state = env.reset() # Reset the environment
            state = self.get_state_representation(state) # Get state representation
            
            ep = [] # List of tuples (state, action, reward)
            for time_step in range(episode_steps): # Play episode
                action = self.act(state) # Get action from agent
                next_state, reward, done = env.step(action, dt) # Take action in environment
                next_state = self.get_state_representation(next_state) # Get state representation
                reward = reward if not done else config.model.termination_penalty # Penalize termination
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
                G = r+self.gamma*G # Discounted reward
                ep[i] = (ep[i][0], ep[i][1], ep[i][2], G, i) # Replace reward with G and add time step

            for state, action, reward, G, time_step in ep: # Add episode to memory
                self.remember(state, action, reward, G, time_step)

            if len(self.memory) > batch_size:
                for i in range(len(self.memory)//batch_size):
                    self.replay(batch_size)

            if config.model.save_weights.enable: # Save weights
                if e%config.model.save_weights.save_frequency == 0 and e != 0:
                    self.save(os.path.join('weights', config.model.save_weights.file_name))
            

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def save(self, filepath):
        self.model.save_weights(filepath)



class DDPGAgent:
    def __init__(self, state_shape, action_size, actor_lr, critic_lr, gamma, rho, buffer_size, batch_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.rho = rho
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Actor Network
        self.actor_model = self.build_actor(state_shape, action_size, actor_lr)
        self.target_actor_model = self.build_actor(state_shape, action_size, actor_lr)
        self.target_actor_model.set_weights(self.actor_model.get_weights())

        # Critic Network
        self.critic_model = self.build_critic(state_shape, action_size, critic_lr)
        self.target_critic_model = self.build_critic(state_shape, action_size, critic_lr)
        self.target_critic_model.set_weights(self.critic_model.get_weights())

        # Noise process
        self.noise_variance = 1
        self.noise = lambda: np.random.normal(loc=0, scale=1, size=self.action_size)

    def build_actor(self, state_shape, action_size, learning_rate):
        # Build the actor network
        state_input = Input(shape=state_shape)
        h = Dense(24, activation='relu')(state_input)
        h = Dense(24, activation='relu')(h)
        output = Dense(action_size, activation='tanh')(h)
        model = Model(inputs=state_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate), loss='mse')
        return model

    def build_critic(self, state_shape, action_size, learning_rate):
        # Build the critic network
        state_input = Input(shape=state_shape)
        state_h = Dense(24, activation='relu')(state_input)

        action_input = Input(shape=(action_size,))
        action_h = Dense(24, activation='relu')(action_input)

        concat = Concatenate()([state_h, action_h])
        h = Dense(48, activation='relu')(concat)
        output = Dense(1, activation='linear')(h)
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate), loss='mse')
        return model
    
    def get_state_representation(self, state):
        state = state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
        state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]) # Convert angle to sin and cos signal
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, noise=True):
        action = self.actor_model.predict(state, verbose=0)
        if noise:
            action = action + self.noise() # Add noise for exploration

        return np.clip(action, -1, 1) # Ensure action is within bounds

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

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
        actor_grads = tape.gradient(actions_for_training, self.actor_model.trainable_variables) #, output_gradients=-action_grads)

        # Manually multiply dQ/da with da/dw to get the final gradient to update the actor
        # Since we're maximizing Q-value, we use the negative of dQ/da
        final_grads = [-ag * g for ag, g in zip(action_grads, actor_grads)]

        # Apply the gradients to the actor's weights
        self.actor_model.optimizer.apply_gradients(zip(final_grads, self.actor_model.trainable_variables))

        # Delete the persistent tape manually
        del tape

        # Update target networks
        self.update_target(self.target_actor_model, self.actor_model)
        self.update_target(self.target_critic_model, self.critic_model)

    def train(self, env, config):
        dt = config.run.dt # Time step in seconds
        num_episodes = config.run.num_episodes # Number of episodes to run
        episode_time = config.run.episode_time # Maximum episode time in seconds
        batch_size = config.run.batch_size # Mini-batch size for training
        episode_steps = int(episode_time/dt) # Number of time steps in an episode

        for e in range(num_episodes):
            state = env.reset(deterministic=True)
            state = self.get_state_representation(state)
            for time_step in range(episode_steps):
                action = self.act(state)
                next_state, reward, done = env.step_continuous(action, dt)
                next_state = self.get_state_representation(next_state)
                reward = reward if not done else config.model.termination_penalty
                self.remember(state, action, reward, next_state, done)
                if done:
                    print(f"episode: {e}/{num_episodes}, score: {time_step}")
                    break
                self.replay()
                state = next_state
            if config.model.save_weights.enable:
                if e%config.model.save_weights.save_frequency == 0 and e != 0:
                    self.save(os.path.join('weights', config.model.save_weights.file_name))

    def update_target(self, target_model, model):
        # Get the weights of both models
        target_weights = target_model.get_weights()
        weights = model.get_weights()
        
        # Update target network weights
        new_weights = []
        for target_weight, weight in zip(target_weights, weights):
            # Perform the soft update
            updated_weight = self.rho * target_weight + (1 - self.rho) * weight
            new_weights.append(updated_weight)
        
        # Set the new weights to the target model
        target_model.set_weights(new_weights)

    def load_weights(self, filepath):
        base, extension = os.path.splitext(filepath)
        self.actor_model.load_weights(base + '_actor' + extension)
        self.critic_model.load_weights(base + '_critic' + extension)
    
    def save(self, filepath):
        base, extension = os.path.splitext(filepath)
        self.actor_model.save(base + '_actor' + extension)
        self.critic_model.save(base + '_critic' + extension)
