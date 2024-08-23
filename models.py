import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

def custom_loss(y_true, y_pred):
    # We now redefine the loss-function to avoid having to store the rewards in a class.
    # We can do this by incorporating the reward and gamma^t in the y_true vector.
    # Earlier we had y_true as a vector of the one-hot encoding of the selected actions. 
    # y_true was then multiplied by G*gamma^t. We now multiply G*gamma^t with y_true before passing it to the loss function.
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1-1e-7) 
    log_probs = y_true * tf.math.log(y_pred_clipped) # Log(pi(s)) * G*gamma^t (one-hot-encoded for the selected action)

    return -tf.reduce_sum(log_probs)

class REINFORCEAgent:
    def __init__(self, state_shape, action_size, hidden_layer_sizes, discount_factor, learning_rate) -> None:
        self.state_shape = state_shape
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes # List specifying the number of neurons in each layer
        self.learning_rate = learning_rate
        self.memory = deque(maxlen = 1000) # Used to store transitions to learn from.
        self.gamma = discount_factor # Discount rate
        self.model = self.build_model(state_shape, action_size, hidden_layer_sizes, learning_rate)

    def build_model(self, state_shape, action_size, hidden_layer_sizes, learning_rate):
        model = Sequential()
        model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape = state_shape))
        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(action_size, activation='softmax'))
        model.compile(loss=custom_loss, optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def remember(self, state, action, reward, G, time_step):
        self.memory.append((state, action, reward, G, time_step))

    def act(self, state):
        act_probs = self.model.predict(state, verbose=0)[0]
        return np.random.choice(np.arange(len(act_probs)), p=act_probs)
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        print("Running replay...")
        states = np.zeros((len(minibatch), self.state_shape[0]))
        # actions = np.zeros((len(minibatch), self.action_size))
        # discounted_rewards = np.zeros((len(minibatch),1))
        # time_steps = np.zeros((len(minibatch),1))
        y_true = np.zeros((len(minibatch), self.action_size))
        for i, (state, action, reward, G, time_step) in enumerate(minibatch):
            states[i] = state
            y_true[i][action] = G*tf.math.pow(self.gamma, time_step)
            # actions[i][action] = 1
            # discounted_rewards[i][0] = G
            # time_steps[i][0] = time_step
        # self.loss_instance.discounted_rewards = discounted_rewards
        # self.loss_instance.time_steps = time_steps
        self.model.fit(states, y_true, verbose = 0)

    def get_state_representation(self, state):
        state = state[[0, 2, 4, 5]] # Extract the x_pos, x_vel, angle, angle_vel (ignore y-values)
        state = np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]]) # Convert angle to sin and cos
        state = np.reshape(state, (1,5)) # Reshape to (1,5)
        return state
            

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
    
    def save(self, filepath):
        self.model.save_weights(filepath)




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


    def load_weights(self, filepath):
        self.model = load_model(filepath)
    
    def save(self, filepath):
        self.model.save(filepath)