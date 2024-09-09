

# Reinforcement Learning for the Inverted Pendulum Problem

This is a simple implementation of the inverted pendulum problem using various reinforcement learning techniques. The problem is to balance an inverted pendulum by applying a force to the base of the pendulum. This is a classical control problem and serves as a good example for exploring reinforcement learning algorithms. This project concerns the implementation of three common reinforcement learning algorithms. The first we will consider is a basic Q-Learning approach. We then move on to REINFORCE, a Policy Gradient method, and finally the Deep Deterministic Policy Gradient (DDPG) algorithm. 

## The Inverted Pendulum Environment

Before tackling the reinforcement learning algorithms, we first need to define the environment in which the agent will operate. The environment defines the state space, action space, and the dynamics of the system, and modelling the inverted pendulum is an interesting task in itself.
The environment we consider is a simple simulation of an inverted pendulum. The pendulum is attached to a base and can rotate about a pivot point. The goal is to balance the pendulum in the upright position by applying a force to the base. In this project, we consider the case where the base can only move along the \\(x\\)-axis. Another simplification is that we assume the base to be much heavier than the pendulum, so that the base is not affected by the pendulum's motion. The pendulum however, is affected by the base's motion and the gravitational force acting on it.

### Modelling the Pendulum

<div style="float: right; margin: 0 0 40px 20px; text-align: center;">
  <img src="../figures/PendulumFig.svg" alt="Pendulum Diagram" width="300"/>
  <div><em>Figure 1: Pendulum Diagram</em></div>
</div>



Given these assumptions, the state of the pendulum can be described by four variables: 
\\[
s_t = [x, \dot{x}, \theta, \dot{\theta}]
\\]
Here, \\(x\\) is the position of the base along the \\(x\\)-axis, \\(\dot{x}\\) is the velocity of the base, \\(\theta\\) is the angle of the pendulum from the \\(x\\)-axis, and \\(\dot{\theta}\\) is the angular velocity of the pendulum. To balance the pendulum, the agent can apply a force \\(F_t\\) to the base. 
The dynamics of the pendulum is described by the time derivatives of the state variables, which is resolved through Newton's laws of motion. The state derivative can be written as: 
\\[
\dot{s} = \begin{bmatrix} \dot{x} \\ \ddot{x} \\ \dot{\theta} \\ \ddot{\theta} \end{bmatrix}
\\]
While the velocity, \\(\dot{x}\\), and the angular velocity, \\(\dot{\theta}\\), is known, the acceleration, \\(\ddot{x}\\) and \\(\ddot{\theta}\\), must be derived. The acceleration of the base is simply given by the force applied to it, \\(\vec{F}\\), divided by the mass of the base, since it is considered to be independent of the pendulum. This can be written as:
\\[
\ddot{x} = \frac{\vec{F}}{m_{\text{base}}}
\\]
The angular acceleration of the pendulum is a bit more involved as it is influenced by the force applied to the base and the gravitational force acting on the pendulum. The angular acceleration can be written as:
\\[
\ddot{\theta} = \frac{(\vec{F} - (\vec{F} \cdot \vec{u})\vec{u} - \vec{F_g}) \times L\vec{u}}{mL^2} - k_d\dot{\theta},
\\]
where \\(\vec{F}\\) is the force applied to the base, \\(\vec{F_g}\\) is the gravitational force acting on the pendulum, \\(L\\) is the length of the pendulum, and \\(\vec{u}\\) is the unit vector pointing from the base to the pendulum. The term \\(k_d\dot{\theta}\\) is a damping term that models the friction acting on the pendulum. Assuming that the force \\(\vec{F}\\) is applied along the \\(x\\)-axis and that the gravitational force acts along the \\(y\\)-axis, this simplifies to:
\\[
\ddot{\theta} = \frac{F_x\sin(\theta) - mg\cos(\theta)}{mL} - k_d\dot{\theta}
\\]
With the dynamics of the pendulum defined, the next step is to define the reward function that the agent will use to learn to balance the pendulum. 

### Reward Function
Reinforcement learning is based on the idea of maximizing the cumulative reward over time. The reward function defines the goal of the agent and guides it towards the desired behavior. At each time step, the reward function assigns a reward, \\(R_t \in \mathbb{R}\\), based on the state of the environment and potentially the action taken by the agent.

Since we want the agent to balance the pendulum, the reward function should assign a high reward for keeping the pendulum close to the upright position and a low reward for letting the pendulum fall. An example of a naive reward function for our problem could be:

\\[
R_t = \begin{cases} 1 & \text{if } |\theta| < \theta_{\text{threshold}} \\ 0 & \text{otherwise} \end{cases}
\\[

which would give a reward of 1 if the pendulum is within a certain threshold of the upright position and 0 otherwise. In practice, we might want to use a more continuous reward function that gives a reward proportional to how close the pendulum is to the upright position. The goal of the agent is then to learn a policy that maximizes the expected cumulative reward over time, denoted by:

\\[
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
\\[

where \\(\gamma \in [0, 1]\\) is a discount factor that determines the importance of future rewards relative to immediate rewards.


## The Reinforcement Learning Algorithms

With the environment defined, we can now move on to the reinforcement learning algorithms. The goal of the agent is to learn a policy that maps states to actions in a way that maximizes the cumulative reward. The agent learns this policy through interaction with the environment, where it receives feedback in the form of rewards. The agent then uses this feedback to update its policy in a way that maximizes the expected cumulative reward.

### Deep Q-Network (DQN)

<!-- The first algorithm we consider is Q-Learning. Q-Learning is a model-free reinforcement learning algorithm that learns the value of taking an action \\(a\\) in a state \\(s\\) under a policy \\(\pi\\). The Q-value, denoted by \\(q_{\pi}(s, a)\\), represents the expected cumulative reward of taking action \\(a\\) in state \\(s\\) and following policy \\(\pi\\) thereafter. Mathematically, the Q-value is defined as [CITE SUTTON]: -->

The first algorithm we consider is Deep Q-Network (DQN). DQN is an extension of Q-Learning where a neural network is used to approximate the Q-value function. This allows the agent to handle environments with large or continuous state spaces, such as our pendulum environment. The Q-value function, denoted by \\(Q_{\pi}(s, a)\\), represents the expected cumulative reward of taking action \\(a\\) in state \\(s\\) and following policy \\(\pi\\) thereafter. Formally, the Q-value is defined as:

\\[
Q_{\pi}(s, a) \doteq \mathbb{E}_{\pi} \left[ G_t \mid S_t = s, A_t = a \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \, \bigg| \, S_t = s, A_t = a \right].
\\]

In our inverted pendulum environment, the input to the neural network is the state of the pendulum, \\(s = [x, \dot{x}, \theta, \dot{\theta}]\\), and the output is the Q-value for each possible action in the action space, \\(a \in \mathcal{A}\\). Since we cannot assign a Q-value to every possible action in a continuous action space, we must discretize the action space into a finite set of actions. The agent can choose to either apply a positive force, a negative force, or no force to the base of the pendulum. The action space is then defined as:

\\[
\mathcal{A} = \{ \text{push left}, \text{do nothing}, \text{push right} \}.
\\]

The goal of the agent is to approximate the optimal Q-value function, denoted by \\(Q_*(s, a)\\), which represents the maximum expected cumulative reward that can be obtained by taking action \\(a\\) in state \\(s\\) and following the optimal policy thereafter. If the agent knows the optimal Q-value function, it can simply choose the action that maximizes the Q-value in each state to maximize the expected cumulative reward. The optimal policy given the optimal Q-value function is then:
\\[
\pi_*(s) = \arg\max_{a} Q_*(s, a).
\\]

To make the neural network approximate the optimal Q-value function, it is updated iteratively based on the rewards it observes. To understand how the Q-value is updated, let's imagine the agent observes a state \\(s_t\\) and outputs the Q-values for each action \\(a_t\\) in that state. The agent can then select the action that maximizes the Q-value, \\(a_t = \arg\max_{a} Q(s_t, a)\\). Next, the agent takes this action \\(a_t\\) and observes the reward \\(R_{t+1}\\) and the next state \\(s_{t+1}\\). Given this new information, we can now form a better estimate of the Q-value for the state-action pair \\((s_t, a_t)\\) than we had before. The better estimate of the Q-value, also known as the target Q-value, is computed using the Bellman equation:
\\[
\text{Target} = R_{t+1} + \gamma \max_{a} Q(s_{t+1}, a; \theta).
\\]
This target Q-value is a better estimate since it incorporates the reward, \\(R_{t+1}\\), observed after taking action \\(a_t\\) in state \\(s_t\\) and the estimated Q-value of the next state \\(s_{t+1}\\). The agent then updates its estimate of the Q-value based on the target Q-value using a loss function. The loss function measures the squared error between the estimated Q-value and the target Q-value and is minimized using gradient descent. The loss function for training the neural network is then defined as:
\\[
\mathcal{L}(\theta) = \left( R_{t+1} + \gamma \max_{a} Q(s_{t+1}, a; \theta) -  Q(s_t, a_t; \theta) \right)^2,
\\]
where \\(\theta\\) are the parameters of the neural network. The parameters of the network are then updated to minimize this loss function using gradient descent. This process is repeated iteratively until the Q-value function converges to the optimal Q-value function.

### REINFORCE
The second algorithm we consider is REINFORCE, a Policy Gradient method. Policy Gradient methods directly learn the policy, \\(\pi_{\theta}(a_t \mid s_t)\\), that maps states to actions without explicitly computing the value function. This policy can be read as the probability of taking action \\(a_t\\) in state \\(s_t\\) given the policy parameters \\(\theta\\). The agent then samples actions from this policy and updates the policy based on the rewards observed. The goal of the agent is to maximize the expected cumulative reward by updating the policy parameters such that actions that lead to high rewards are more likely to be selected in the future.

In this project, we use a neural network to represent the policy, \\(\pi_{\theta}(a_t \mid s_t)\\), where the input is the state of the pendulum, \\(s = [x, \dot{x}, \theta, \dot{\theta}]\\) and the output of the neural network is the probability distribution over each action in the action space, \\(a \in \mathcal{A}\\). Here, the action space is the same as in the DQN algorithm, where the agent can choose to apply a positive force, a negative force, or no force to the base of the pendulum. Given a state \\(s_t\\), the network then outputs the probabilities of taking each action in the action space, \\(\pi_{\theta}(a_t \mid s_t) = \{ \pi_{\theta}(\text{push left} \mid s_t), \pi_{\theta}(\text{do nothing} \mid s_t), \pi_{\theta}(\text{push right} \mid s_t) \}\\). The agent then samples an action from this probability distribution and observes the reward \\(R_t\\) and the next state \\(s_{t+1}\\). 


















