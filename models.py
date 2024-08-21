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
        model = Sequential([
            Dense(20, activation='relu', input_shape = state_shape),
            Dense(20, activation='relu'),
            Dense(action_size, activation='softmax')
        ])
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


# A = REINFORCEAgent((4,), 2)
# state = np.array([1, 1, 3, 4])
# state = np.reshape(state, (1,4))
# print(A.model.predict(state, verbose=0))
# print(A.act(state))
