import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
from matplotlib.patches import Rectangle
from pendulum import Pendulum
from models import REINFORCEAgent, DQNAgent, DDPGAgent
from dataModels import AnimateParams, Config

def run_episode(env, agent):
    """
    Run an episode using the specified environment and agent. Returns the state history for the episode.

    Args:
    - env (Pendulum): The Pendulum environment object.
    - agent (DDPGAgent, DQNAgent, REINFORCEAgent): The agent object to act in the environment.
    """
    print("Running episode...")
    
    # Disable noise and exploration for testing
    agent.enable_noise = False
    agent.eps = 0.

    state_history = [] # Store the state history for the episode

    state = env.reset().copy()
    state_history.append(state)
    done = False

    while not done:
        force, action = agent.act(agent.get_state_representation(state))
        next_state, reward, done = env.step(force)
        state = next_state.copy()
        state_history.append(state)

    return np.array(state_history)

# Function to read episode data from the log
def read_ep_from_log(file_path, ep_number):
    """
    Reads and parses the episode data from the specified log file.
    
    Args:
    - file_path (str): The path to the CSV log file.
    - ep_number (int): The episode number to retrieve.
    
    Returns:
    - np.ndarray: Parsed state history for the specified episode.
    """
    data = pd.read_csv(file_path)
    episodes = data['ep'].unique()
    ep_number = episodes[ep_number]  # Get the episode number from the list
    episode_data = data[data['ep'] == ep_number]  # Extract data for the specified episode

    # Function to parse the state safely from the string format
    def parse_state(state_str):
        try:
            # Convert the string representation of the list to an actual list
            state = eval(re.sub(' +', ',', state_str).replace('[,', '['))
            return state
        except:
            # Handle potential parsing errors by returning a default or dummy state
            return [0, 0, 0, 0, 0, 0]

    # Apply parsing to the episode data
    episode_data['parsed_state'] = episode_data['state'].apply(parse_state)
    
    # Convert to numpy array and return
    return np.array(episode_data['parsed_state'].tolist())


# Function to animate the episode
def animate_episode(state_history, config: Config):
    """
    Animates the pendulum episode given a state history.
    
    Args:
    - state_history (np.ndarray): Array containing the state history for the episode.
    """
    print("Animating episode...")

    r = config.pendulum.length  # Length of the pendulum
    x_lim = config.animate.x_lim  # Limit for x-axis
    y_lim = config.animate.y_lim  # Limit for y-axis

    # Define the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_aspect('equal')

    # Remove grid and y-axis ticks
    ax.grid(False)
    ax.set_yticks([])  # Remove y-axis ticks

    # Customize x-axis line
    ax.axhline(0, color='black', linewidth=1.5)  # Black line for x-axis at y = 0
    x_ticks = np.arange(-x_lim, x_lim + 1, 1)  # Major ticks every meter
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.minorticks_on()
    ax.set_xticks(np.arange(-x_lim, x_lim + 0.5, 0.5), minor=True)  # Minor ticks every half meter
    ax.xaxis.set_tick_params(which='major', bottom=True, top=False, length=5)
    ax.xaxis.set_tick_params(which='minor', bottom=True, top=False, length=3)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Initialize base as a rectangle with a grey color and black edge
    base_width = r/2
    base_height = r/4
    base = Rectangle((0, -base_height / 2), base_width, base_height, facecolor='#555555', edgecolor='black', linewidth=1.5, zorder=2)  # Use facecolor for fill and edgecolor for border
    ax.add_patch(base)

    # Initialize pendulum line and bob with grey color and clipping
    pendulum_line, = ax.plot([], [], '-', lw=2, color='black', zorder=1, clip_on=True)  # Clip pendulum line
    pendulum_bob, = ax.plot([], [], 'o', markersize=8, color='#555555', markeredgecolor='black', zorder=2, clip_on=True)  # Clip pendulum bob

    # Initialize the plot elements
    def init():
        base.set_xy((0, -base_height / 2))  # Initial position of the rectangle base
        pendulum_line.set_data([], [])
        pendulum_bob.set_data([], [])
        return base, pendulum_line, pendulum_bob

    # Update function for animation
    def update(frame):
        x_pos, y_pos, x_vel, y_vel, angle, ang_vel = state_history[frame]

        # Update base position
        base.set_xy((x_pos - base_width / 2, -base_height / 2))  # Center the rectangle at x_pos

        pendulum_x = x_pos + r * np.cos(angle)
        pendulum_y = r * np.sin(angle)

        # Update pendulum line and bob
        pendulum_line.set_data([x_pos, pendulum_x], [0, pendulum_y])
        pendulum_bob.set_data([pendulum_x], [pendulum_y])

        return base, pendulum_line, pendulum_bob

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(state_history), init_func=init, blit=True, interval=50
    )

    # Save the animation if a path is provided
    if config.animate.save_path:
        ani.save(config.animate.save_path, writer='pillow')

    # Display the animation
    plt.show()
    plt.close(fig)


# # Main script to run the functions
# file_path = 'logs/DDPG_try4_1m2xlim.csv'  # Replace with the correct path to your CSV file
# ep_number = -2  # Replace with the episode number you want to visualize

# # Read and parse the episode data
# state_history = read_ep_from_log(file_path, ep_number)

# # Animate the episode
# animate_episode(state_history)
