

# Reinforcement Learning for the Inverted Pendulum Problem

This is a simple implementation of the inverted pendulum problem using various reinforcement learning techniques. The problem is to balance an inverted pendulum by applying a force to the base of the pendulum. This is a classical control problem and serves as a good example for exploring reinforcement learning algorithms. This project concerns the implementation of three common reinforcement learning algorithms. The first we will consider is a basic Q-Learning approach. We then move on to REINFORCE, a Policy Gradient method. And finally the Deep Deterministic Policy Gradient (DDPG) algorithm.

## The Inverted Pendulum Environment

The environment is a simple simulation of an inverted pendulum. The pendulum is attached to a base and can rotate about a pivot point. The goal is to balance the pendulum in the upright position by applying a force to the base. In this project, we consider the case where the base can only move along the $x$-axis. Another simplification is that we assume the base to be much heavier than the pendulum, so that the base is not affected by the pendulum's motion. The pendulum however, is affected by the base's motion and the gravitational force acting on it.

<div style="float: right; margin: 0 0 40px 20px; text-align: center;">
  <img src="./figures/PendulumFig.svg" alt="Pendulum Diagram" width="300"/>
  <div><em>Figure 1: Pendulum Diagram</em></div>
</div>



Given these assumptions, the state of the pendulum can be described by four variables: 
$
s_t = [x, \dot{x}, \theta, \dot{\theta}]
$
Here, $x$ is the position of the base along the $x$-axis, $\dot{x}$ is the velocity of the base, $\theta$ is the angle of the pendulum from the $x$-axis, and $\dot{\theta}$ is the angular velocity of the pendulum. To balance the pendulum, the agent can apply a force $F_t$ to the base. 
The dynamics of the pendulum is described by the time derivatives of the state variables, which is resolved through Newton's laws of motion. The state derivative can be written as: 
$
\frac{d}{dt} s = \begin{bmatrix} \dot{x} \\ \ddot{x} \\ \dot{\theta} \\ \ddot{\theta} \end{bmatrix}
$
While the velocity, $\dot{x}$, and the angular velocity, $\dot{\theta}$, is known, the acceleration, $\ddot{x}$ and $\ddot{\theta}$, must be derived. The acceleration of the base is given by the force applied to it, $\vec{F}$, divided by the mass of the base:
$
\ddot{x} = \frac{\vec{F}}{m_{\text{base}}}
$
The angular acceleration of the pendulum is a bit more involved and is influenced by the force applied to the base and the gravitational force acting on the pendulum. The angular acceleration can be written as:
$
\ddot{\theta} = \frac{(\vec{F} - (\vec{F} \cdot \vec{u})\vec{u} - \vec{F_g}) \times L\vec{u}}{mL^2},
$
where $\vec{F}$ is the force applied to the base, $\vec{F_g}$ is the gravitational force acting on the pendulum, $L$ is the length of the pendulum, and $\vec{u}$ is the unit vector pointing from the base to the pendulum.









