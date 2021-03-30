# RL_maze_escape

## Description of implementation process

The following paragraphs will describe the main decisions that were taken through- out the implementation of question 2 and the reasons behind them. Note that as the code is well commented, there will not be any explanation of it in this paragraphs.<br>

Firstly, a Q-Learning implementation with the following hyper-parameters was found to perform well under a number of different environments.

- Q-Network architecture: many experiments were conducted to fine-tune net- work architecture, the best model found consisted of an input layer taking the xy coordinates of the state, 2 hidden layers with 100 features coupled with ReLU and an output layer with 4 nodes, one for each possible action.
- &epsilon; decay rate: ε must be decayed throughout the episodes in order to achieve an exploitable optimal policy, many decay strategies were tried (exponential, linear, inverse, etc.), in the end it was chosen to settle with a simple 5% decrease in ε at the end of every episode as it gave more consistent results.
- variable episode length: We want the agent to learn how to get to the goal quickly, therefore we should not allow it to take a large number of steps. However, in the early epochs, we want the agent to explore as much of the environment as possible. Hence, a variable episode length was chosen, starting from 1500 steps and decreasing by 2.5% at the and of each episode, without going under 150 steps per episode. Again this hyper-parameter was found through experiments.
- target network updates: after some trial and error, consistency was found up- dating the target network every 1000 steps.
- loss function: after some literature review, it was concluded that state of the art DQN suggest the use of the Huber loss rather than MSE loss due to its higher resistance to outliers. Significant improvement was seen upon this change.
- buffer hyper-parameters: an initial maximum capacity of 5000 was chosen, but was quickly updated to 10000 as the agent could not converge to the goal state otherwise. As the network architecture was increased from a single hidden layer to 2 hidden layers, it was decided to double the batch size from an initial guess of 128 to 256, as bigger network requires more training samples to converge.

Secondly, a simple modification to the DQN class (which handled the training of the Q-network) sufficed to implement Double Q-Learning, great improvement was ob- served upon this simple change. Finally, a prioritised replay buffer was implemented, and an &alpha; prioritisation factor of 1 was found to give more consistent results. A strong boost in performance was found by developing an early-stop condition, which tested the greedy policy when the agent managed to get to the goal state at low values of ε and reduced the learning rate until the agent consistently hit the target.
