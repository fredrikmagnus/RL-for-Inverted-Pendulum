

# Reinforcement Learning for the Inverted Pendulum Problem

This is a simple implementation of the inverted pendulum problem using various reinforcement learning techniques. The problem is to balance an inverted pendulum by applying a force to the base of the pendulum. This is a classical control problem and serves as a good example for exploring reinforcement learning algorithms. This project concerns the implementation of three common reinforcement learning algorithms. The first we will consider is a basic Q-Learning approach. We then move on to REINFORCE, a Policy Gradient method. And finally the Deep Deterministic Policy Gradient (DDPG) algorithm.

## The Inverted Pendulum Environment

The environment is a simple simulation of an inverted pendulum. The pendulum is attached to a base and can rotate about a pivot point. The goal is to balance the pendulum in the upright position by applying a force to the base. 

<div style="float: right; margin: 0 0 10px 10px; text-align: center;">
  <img src="./figures/PendulumFig.svg" alt="Pendulum Diagram" width="300"/>
  <div><em>Figure 1: Pendulum Diagram</em></div>
</div>

<a name="fig-pendulum"></a>
![Pendulum Diagram](./figures/PendulumFig.svg)
*Figure 1: Pendulum Diagram*




| ![Pendulum Diagram](./figures/PendulumFig.svg) |
|:--:|
| *Figure 1: Pendulum Diagram* |

<table style="border: none;">
  <tr>
    <td style="text-align: center;">
      <img src="./figures/PendulumFig.svg" alt="Pendulum Diagram" />
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <em>Figure 1: Pendulum Diagram</em>
    </td>
  </tr>
</table>