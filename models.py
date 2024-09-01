import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input, Dense
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





class ActorCritic:
    def __init__(self, config) -> None:
        self.state_shape = (5,)
        self.action_size = 2
        self.hidden_layer_sizes = config.model.layer_sizes
        self.learning_rate = config.model.learning_rate
        # self.learning_rate_actor = config.model.learning_rate
        # self.learning_rate_critic = config.model.learning_rate
        self.memory = deque(maxlen = 300)
        self.gamma = config.model.discount_factor
        self.critic = self.build_critic(self.state_shape, self.hidden_layer_sizes, self.learning_rate)
        self.actor = self.build_actor(self.state_shape, self.action_size, self.hidden_layer_sizes, self.learning_rate*1e-1)
         
        # Eligibility traces
        self.eligibility_traces = True
        self.trace_decay = 0.9
        self.actor_eligibility_trace = [tf.zeros_like(weight) for weight in self.actor.trainable_weights]
        self.critic_eligibility_trace = [tf.zeros_like(weight) for weight in self.critic.trainable_weights]

    def build_critic(self, state_shape, hidden_layer_sizes, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape = state_shape))
        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def build_actor(self, state_shape, action_size, hidden_layer_sizes, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape = state_shape))
        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(action_size, activation='softmax'))
        model.compile(loss=self.custom_actor_loss, optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def custom_actor_loss(self, y_true, y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1-1e-7) 
        log_probs = y_true * tf.math.log(y_pred_clipped)
        return -tf.reduce_sum(log_probs)
    
    def act(self, state):
        act_probs = self.actor.predict(state, verbose=0)[0]
        return np.random.choice(np.arange(len(act_probs)), p=act_probs)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        print("Running replay...")
        mini_batch = random.sample(self.memory, batch_size)

        # Prepare batches
        states = np.zeros((batch_size, self.state_shape[0]))
        next_states = np.zeros((batch_size, self.state_shape[0]))
        targets = np.zeros((batch_size, 1))
        advantages = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
            states[i] = state
            next_states[i] = next_state
            reward = reward if not done else 0
            
            # Compute the target for the critic
            target = reward + self.gamma * self.critic.predict(next_state, verbose=0)
            targets[i] = target
            
            # Compute the advantage for the actor
            advantage = target - self.critic.predict(state, verbose=0)
            advantages[i][action] = advantage

        # Update the critic using the batch of states and targets
        self.critic.fit(states, targets, verbose=0)
        
        # Update the actor using the batch of states and advantages
        self.actor.fit(states, advantages, verbose=0)
    
    

    def get_state_representation(self, state):
        state = state[[0, 2, 4, 5]]
        state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]])
        state = np.reshape(state, (1,5))
        return state
    
    def train(self, env, config):
        dt = config.run.dt
        num_episodes = config.run.num_episodes
        episode_time = config.run.episode_time
        episode_steps = int(episode_time/dt)

        mini_batch_enable = False

        for e in range(num_episodes):
            state = env.reset(deterministic=False)
            state = self.get_state_representation(state)
            I = 1 # Importance sampling ratio

            for time_step in range(episode_steps):
                # action = self.act(state)
                policy = self.actor.predict(state, verbose=0)
                action = np.random.choice(np.arange(len(policy[0])), p=policy[0])

                next_state, reward, done = env.step(action, dt)
                # print(f"angle: {next_state[4]*180/np.pi}, state: {next_state}, done: {done}")
                next_state = self.get_state_representation(next_state)
                reward = reward if not done else config.model.termination_penalty

                if mini_batch_enable:
                    self.remember(state, action, reward, next_state, done)
                    self.replay(config.run.batch_size)   
                elif self.eligibility_traces:

                    # Compute the target for the critic
                    target = tf.convert_to_tensor(reward + self.gamma * self.critic(next_state, training=False), dtype=tf.float32)

                    # Compute the advantage for the actor
                    current_state_value = tf.convert_to_tensor(self.critic(state, training=False), dtype=tf.float32)
                    advantage = target - current_state_value  # Advantage

                    y_true = np.zeros((1, self.action_size))
                    y_true[0][action] = I * advantage 
                    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

                    # Compute gradient of the critic
                    with tf.GradientTape() as tape:
                        # Forward pass using the call method
                        current_state_value = self.critic(state, training=True)
                        critic_loss = MeanSquaredError()(target, current_state_value) # Compute the loss for the critic
                    critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights) # Compute the gradient of the loss w.r.t. the critic weights
                    print(f"critic_loss: {critic_loss}, grads: {critic_grads}")
                    # self.critic_eligibility_trace = self.gamma * self.trace_decay * self.critic_eligibility_trace + critic_grads # Update the eligibility trace for the critic
                    # Update the critic eligibility trace, but only if the gradients are not None
                    self.critic_eligibility_trace = [
                        self.gamma * self.trace_decay * trace + advantage*grad if grad is not None else trace
                        for trace, grad in zip(self.critic_eligibility_trace, critic_grads)
                    ]
                    self.critic.optimizer.apply_gradients(zip(self.critic_eligibility_trace, self.critic.trainable_weights)) # Update the critic weights
                    # Update the actor using the advantage
                    # Compute gradient of the actor
                    with tf.GradientTape() as tape:
                        actor_loss = self.custom_actor_loss(y_true, policy)
                    actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
                    print(f"actor_loss: {actor_loss}, grads: {actor_grads}")
                    # self.actor_eligibility_trace = self.gamma * self.trace_decay * self.actor_eligibility_trace + actor_grads
                    # Update the actor eligibility trace, but only if the gradients are not None
                    self.actor_eligibility_trace = [
                        self.gamma * self.trace_decay * trace + advantage*grad if grad is not None else trace
                        for trace, grad in zip(self.actor_eligibility_trace, actor_grads)
                    ]
                    self.actor.optimizer.apply_gradients(zip(self.actor_eligibility_trace, self.actor.trainable_weights))


                else:
                    # Compute the target for the critic
                    target = reward + self.gamma * self.critic.predict(next_state, verbose=0) # TD-target
                    # Compute the advantage for the actor
                    advantage = target - self.critic.predict(state, verbose=0) # Advantage

                    # print(f"G: {G}, done: {done}")
                    # Update the critic using the target value
                    self.critic.fit(state, target, verbose=0)

                    # Update the actor using the advantage
                    y_true = np.zeros((1, self.action_size))
                    y_true[0][action] = I * advantage
                    self.actor.fit(state, y_true, verbose=0)

                if done:
                    print(f"episode: {e}/{num_episodes}, score: {time_step}")
                    break

                # I *= self.gamma
                state = next_state

            if config.model.save_weights.enable:
                if e%config.model.save_weights.save_frequency == 0 and e != 0:
                    self.save(os.path.join('weights', config.model.save_weights.file_name))
    
    def save(self, filepath):
        actor_filepath = filepath + '_actor'
        critic_filepath = filepath + '_critic'
        self.actor.save_weights(actor_filepath)
        self.critic.save_weights(critic_filepath)
    
    def load_weights(self, filepath):
        actor_filepath = filepath + '_actor'
        critic_filepath = filepath + '_critic'
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)

    

# class ActorCritic:
#     def __init__(self, config) -> None:
#         self.state_shape = (5,)
#         self.action_size = 2
#         self.hidden_layer_sizes = config.model.layer_sizes
#         self.learning_rate = config.model.learning_rate
#         self.gamma = config.model.discount_factor
#         self.model = self.build_shared_network(self.state_shape, self.action_size, self.hidden_layer_sizes, self.learning_rate)

#     def build_shared_network(self, state_shape, action_size, hidden_layer_sizes, learning_rate):
#         # Shared input and hidden layers
#         input_layer = Input(shape=state_shape)
#         hidden = Dense(hidden_layer_sizes[0], activation='relu')(input_layer)
#         for layer_size in hidden_layer_sizes[1:]:
#             hidden = Dense(layer_size, activation='relu')(hidden)

#         # Separate policy head
#         policy_hidden = Dense(8, activation='relu')(hidden)
#         policy_hidden = Dense(8, activation='relu')(policy_hidden)
#         policy_output = Dense(action_size, activation='softmax', name='policy_output')(policy_hidden)

#         # Separate value head
#         value_hidden = Dense(8, activation='relu')(hidden)
#         value_hidden = Dense(8, activation='relu')(value_hidden)
#         value_output = Dense(1, activation='linear', name='value_output')(value_hidden)

#         # Define the model with two outputs
#         model = Model(inputs=input_layer, outputs=[policy_output, value_output])
#         model.compile(
#             optimizer=Adam(learning_rate=learning_rate),
#             loss={'policy_output': self.custom_actor_loss, 'value_output': 'mse'}
#         )
#         return model
    
#     def custom_actor_loss(self, y_true, y_pred):
#         y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
#         log_probs = y_true * tf.math.log(y_pred_clipped)
#         return -tf.reduce_sum(log_probs)

#     def act(self, state):
#         policy, _ = self.model.predict(state, verbose=0)
#         return np.random.choice(np.arange(len(policy[0])), p=policy[0])
    
#     def get_state_representation(self, state):
#         state = state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
#         state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]])
#         state = np.reshape(state, (1, 5))
#         return state
    
#     def train(self, env, config):
#         dt = config.run.dt
#         num_episodes = config.run.num_episodes
#         episode_time = config.run.episode_time
#         episode_steps = int(episode_time/dt)

#         for e in range(num_episodes):
#             state = env.reset()
#             state = self.get_state_representation(state)
#             I = 1 # Importance sampling ratio
#             cumul_reward = 0
#             for time_step in range(episode_steps):
#                 policy, value_current = self.model.predict(state, verbose=0) # Predict the current value and policy
#                 print(f"policy: {policy}, value_current: {value_current}")
#                 action = np.random.choice(np.arange(len(policy[0])), p=policy[0]) # Sample action from policy

#                 next_state, reward, done = env.step(action, dt)
#                 next_state = self.get_state_representation(next_state)
#                 reward = reward if not done else config.model.termination_penalty
#                 cumul_reward *= self.gamma
#                 cumul_reward += reward

#                 # Predict the value of the next state
#                 _, value_next = self.model.predict(next_state, verbose=0)

#                 # Compute the target for the value head
#                 target = reward + self.gamma * value_next

#                 # Compute the advantage
#                 advantage = target - value_current 

#                 # Prepare y_true for the actor update
#                 y_true_policy = np.zeros((1, self.action_size))
#                 y_true_policy[0][action] = I * advantage

#                 # Make a single call to fit to update both heads simultaneously
#                 self.model.fit(state, {'policy_output': y_true_policy, 'value_output': target}, verbose=0)


#                 if done:
#                     print(f"episode: {e}/{num_episodes}, score: {time_step}")
#                     # Print total cumulative reward
#                     print(f"Total cumulative reward: {cumul_reward}")
#                     break

#                 I *= self.gamma
#                 state = next_state

#             if config.model.save_weights.enable:
#                 if e % config.model.save_weights.save_frequency == 0 and e != 0:
#                     self.save(os.path.join('weights', config.model.save_weights.file_name))

    

#     def save(self, filepath):
#         self.model.save_weights(filepath)

#     def load_weights(self, filepath):
#         self.model.load_weights(filepath)


