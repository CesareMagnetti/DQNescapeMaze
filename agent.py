############################################################################
############################################################################
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
############################################################################
############################################################################

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import collections
from matplotlib import cm
from matplotlib import  pyplot as plt
import os
import random
import time

# default configuration for the network
default_net_specs = {'input_dimension': 2, 'output_dimension': 4, 'hidden_layers': 1,
                     'hidden_features': 100}
# this is just a handle to avoid having messy code later
tensor = torch.FloatTensor

# prints some message when instanciating the Network class with a wrong dictionary of parameters
def help_message():
    net_specs = {'input_dimension': "int, number of features in the input layer",
                 'output_dimension': "int, number of features in the output layer",
                 'hidden_layers': "OPTIONAL: int, number of hidden layers (default = 1)",
                 'hidden_features': "OPTIONAL: int, number of features in each hidden layer (default = 100)"
                 }

    print("the input parameter <net_specs> should look like the following:\n\n")
    for key, item in net_specs.items():
        print("{}:\t{}".format(key, item))



# ========== NEURAL NETWORK CLASS ==========

class Network(torch.nn.Module):

    def __init__(self, input_dimension = 2, output_dimension = 4, hidden_layers = 1,
                 hidden_features = 100):

        '''
        :param input_dimension: int, number of features in the input layer (default = 2)
        :param output_dimension: int, number of features in the output layer (default = 4)
        :param hidden_layers: int, number of hidden layers (default = 1)
        :param hidden_features: int-list<int>, number of features in each hidden layer (default = 100)
        '''

        super(Network, self).__init__()

        if isinstance(hidden_features, int):
            hidden_features = [hidden_features,]*(hidden_layers)
        elif isinstance(hidden_features, (tuple, list)):
            assert len(hidden_features) == hidden_layers, "if list/tuple is parsed for ``hidden_features``," \
                                                          " it must be of length ``hidden_layers``."
        main = []
        main.append(nn.Linear(in_features=input_dimension, out_features=hidden_features[0]))
        main.append(nn.ReLU(inplace=True))

        for i in range(hidden_layers-1):
            main.append(nn.Linear(in_features=hidden_features[i], out_features=hidden_features[i+1]))
            main.append(nn.ReLU(inplace=True))

        main.append(nn.Linear(in_features=hidden_features[-1], out_features=output_dimension))

        self.main = nn.Sequential(*main)

    def forward(self, input):
        return self.main(input)


# ========== REPLAY BUFFER CLASS ==========

class ReplayBuffer():

    def __init__(self, **kwargs):

        '''
        :param kwargs: dict, optional paramters for th buffer class:
                             - capacity: int, maximum number of transitions stored (default = 5000).
                             - batch_size: int, batch size to sample from buffer (default = 100)
                             - prioritised_experience: bool, flag to use prioritised replay sampling.
                             - alpha: float, hyperparamter for prioritised sampling. in (0,1).
        '''

        # instanciate the replay buffer
        self.capacity = kwargs.pop('buffer_capacity', 5000)
        # save transitions as (s, a, r, s')
        self.memory = collections.deque(maxlen=self.capacity)
        # additional parameters
        self.batch_size = kwargs.pop('batch_size', 128)
        self.prioritised = kwargs.pop('prioritised_experience', True)
        if self.prioritised:
            # instanciate a buffer for the transitions weights
            self.weights = collections.deque(maxlen=self.capacity)
            # prioritisation factor
            self.alpha = kwargs.pop('alpha', 0.1)
            print("Using prioritized experience replay buffer with an alpha facto of %.2f" % self.alpha)
            self.idxs = None


    ##### class methods #####

    # function to add a transition to the buffer
    def add(self, transition):
        # pop left if max length exceeded
        if not len(self.memory)<self.capacity:
            self.memory.popleft()
            if self.prioritised:
                self.weights.popleft()

        # append transition to deque
        self.memory.append(transition)

        if self.prioritised:
            # give the transition maximum weight, or a weight of 1 if first transition stored
            self.weights.append(max(self.weights) if len(self.weights)>0 else 1)

    # function to sample a minibatch of transitions
    def sample_minibatch(self):
        # return None if there are not enough transitions in the buffer
        if len(self.memory)>self.batch_size:
            # use random choice to select the minibatch. self.get_p() will either output a prioritised distribution
            # to sample from or None, in which case np.random.choice() will sample from a uniform distribution.
            # store idxs so that we can update the weights later.
            self.idxs = np.random.choice(len(self.memory), self.batch_size, replace=False, p = self.get_p())
            # extract the sampled transitions
            batch = [self.memory[i] for i in self.idxs]
            # this will convert N x (s, a, r, s') to [N x s, N x a, N x r, N x s'] for easier handling
            return [*zip(*batch)]
        else:
            return None

    # function to update the weights assigned to each transition (if needed)
    def update_weights(self, deltas, eps = 10e-5):
        if self.prioritised:
            # loop through sampled indices and update their weight according to theory
            for idx, d in zip(self.idxs, deltas):
                # eps ensures non-zero probabilities
                self.weights[idx] = torch.norm(d).item() + eps

    # function to create a prioritised distribution from which to sample the transitions (if needed)
    def get_p(self):
        if self.prioritised:
            # update the probability distribution of transition according to the theory
            p = [w**self.alpha for w in self.weights]
            tot = sum(p)
            return [pp/tot for pp in p]
        else:
            return None


# ========== DQN TRAINING CLASS ==========
class DQN:

    def __init__(self, net_specs, **kwargs):
        '''
        :param net_specs: dict, specification to build the neural networks (see class Network for more details)
        :param kwargs: additional parameters for the DQN (i.e. discount, learning rate, DoubleQLearning etc)
        '''

        # tries to load the networks, if it fails it will output an informative message
        try:
            self.q_network = Network(**net_specs)
            self.q_network.train()
            self.target_network = Network(**net_specs)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
        except:
            help_message()
            raise ValueError('error while parsing <net_specs> input parameters.\n')

        # print network architecture
        print('\n\nthe following Q-network architecture was instanciated:\n\n')
        print(self.q_network)
        # Define the optimiser which is used when updating the Q-network.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=kwargs.pop('lr', 0.0002))
        self.scheduler = LambdaLR(self.optimiser, lr_lambda=lambda i: 0.8) # lr*=0.8 when stepped
        # Define the loss to train the Q network
        self.loss = kwargs.pop('loss', nn.SmoothL1Loss())
        # Define the discount factor
        self.discount = kwargs.pop('discount', 0.9)
        # Define the delay (in steps) after wich we set the target network equal to the q network
        self.delay = kwargs.pop('delay', 1000)
        # Define if using Double Q Learning
        self.DoubleQLearning = kwargs.pop('DoubleQLearning', True)

        if self.DoubleQLearning:
            print('\nUsing Double Q-Learning algorithm. Remove ``--DoubleQLearning`` to use Q-Learning.\n')
        else:
            print('\nUsing Q-learning algorithm. Parse ``--DoubleQLearning = True`` to use Double Q-Learning.\n')


    ##### class methods #####
    def train_q_network(self, buffer, testing):
        # get transition data
        batch = buffer.sample_minibatch()
        # if None there are not enough samples in the buffer, wait.
        if batch is None:
            return None

        state, action, R, next_state = (torch.tensor(batch[i]) for i in range(4))
        state, R, next_state = state.type(tensor), R.type(tensor), next_state.type(tensor)

        ### pass forward ###

        # get value of the current state-action
        Q = self.q_network(state).gather(1, action.unsqueeze(1))

        if self.DoubleQLearning:
            # get the action indeces of the best possible next actions
            action_indices = torch.argmax(self.q_network(next_state), dim=1)
            # evaluate the target network at those indices
            maxQ = self.target_network.forward(next_state).gather(1, action_indices.unsqueeze(1)).detach()
        else:
            # get the maximum state-action for the next state
            maxQ = self.target_network(next_state).max(1)[0].detach().unsqueeze(-1)

        # get the expected state-action values (Q^hat)
        Q_hat = (maxQ * self.discount) + R.unsqueeze(-1)

        # update buffer weights if needed
        buffer.update_weights(Q_hat-Q)

        # evaluate the loss
        loss = self.loss(Q, Q_hat)

        ### pass backward ###

        # only propagate gradients if we are not going through a testing episode
        if not testing:
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        # return the training loss
        return loss.item()

    # function to set the target network equal to the q network
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # function to update the lr
    def update_lr(self):
        print("decreasing learning rate")
        self.scheduler.step()

    # function to evaluate the q-network on an input state(s)
    def evaluate(self, state):
        state = torch.tensor(state)
        return self.q_network(state.type(tensor)).detach().numpy().squeeze()


# ========== AGENT CLASS ==========
class Agent:

    # Function to initialise the agent
    def __init__(self, **kwargs):

        '''
        :param kwargs: additional parameters that will affect all other classes, here is a list of all possible options:

                       ---------- CONTROLLING THE AGENT CLASS ----------

                       - episode_length: int, initial number of steps in an episode (DEFAULT = 1000)
                       - episode_length_decay_rate: float, how much to decrease (in percentage) the episode length
                                                         after each episode. (DEFAULT: 0.025 --> 2.5%)
                       - egreedy: bool, flag if the agent is in egreedy mode. (DEFAULT = True)
                       - greedy: bool, flag if the agent is in greedy mode. (DEFAULT = False)
                       - epsilon: float, epsilon parameter for the egreedy policy. (DEFAULT = 1)
                       - epsilon_decay_rate: float, how much to decrease (in percentage) epsilon after
                                                     each episode. (DEFAULT = 0.05 --> 5%)
                       - reward: callable, handle to get the current reward, should accept the euclidean
                                           distance from the goal as a single parameter. (DEFAULT = 1-d)
                       - discount: float, discount factor for future rewards. (DEFAULT = 0.999)
                       - action_size, int: number of possible actions at each state. (DEFAULT = 4)
                       - savedir: str/pathlike, where to save training logs/plots. (DEFAULT - "./agent_training_stats/")
                       - display: bool, flag if displaying interactive logs as the agent learns. WARNING: seriously
                                        slows down training. (DEFAULT = False)

                       ---------- CONTROLLING THE NETWORK CLASS ----------

                       - net_specs: dict, specifications for the network architecture. (DEFAULT: see top of the file)

                       ---------- CONTROLLING THE REPLAY BUFFER CLASS ----------

                       - buffer_capacity: int, maximum buffer_capacity of the replay buffer. (DEFAULT = 5000)
                       - batch_size: int, size of the minibatch to sample. (DEFAULT = 100)
                       - prioritised_experience: bool, if replay buffer uses prioritised sampling. (DEFAULT = True)
                       - alpha: float in [0,1], weight of the prioritisation. (DEFAULT = 0.1)

                       ---------- CONTROLLING THE DQN CLASS ----------

                       -loss: callable/torch.nn.loss, loss function to use during training. (DEFAULT = nn.MSELoss())
                       -lr: float, learning rate to use during the optimisation process. (DEFAULT = 0.0002)
                       - discount: float in [0,1], discount factor for future rewards. (DEFAULT = 0.999)
                       - delay: int, number of steps to wait before updating the target network. (DEFAULT = 500)
                       - DoubleQLearning: bool, flag is using double q-learning algorithm . (DEFAULT = True)

                       -----------------------------------------------

        The following variables will be instanciated:

        - num_steps_taken: int, number of steps taken so far through the episode.
        - steps_to_target_update: int, count number of steps before updating the target network.
        - num_episodes: int, number of episodes taken so far through the training process.
        - state: ndarray, contains the current state coordinates.
        - action: int: contains the current DISCRETE action.
        - distance_to_goal: float, contains the current euclidean distance from the goal state.
        - testing: bool, flag if we are testing the greedy policy during the current episode.
        - logs: dict, dictionary to store training logs.
        - buffer: ReplayBuffer(), instance for the experience replay buffer.
        - dqn: DQN(), instance for the q-network training process.
        - lossVisualiser: LossVisualiser(), instance to handle plotting functions for the logs.
        '''


        # custom class attributes
        self.episode_length = kwargs.pop('episode_length', 1000)
        self.episode_length_decay_rate = kwargs.pop('episode_length_decay_rate', 0.025)
        self.egreedy = kwargs.pop('egreedy', True)
        self.greedy = kwargs.pop('greedy', False)
        assert not (self.greedy and self.egreedy), "agent can either be in greedy or egreedy mode, not both."
        if not self.greedy and not self.egreedy:
            print('WARNING: agent was initiated in random mode.')

        if self.egreedy:
            self.epsilon = kwargs.pop('epsilon', 1)
            self.epsilon_decay_rate = kwargs.pop('epsilon_decay_rate', 0.05)

        self.reward = kwargs.pop('reward', lambda d: (1-d))
        self.action_size = kwargs.get('action_size', 4)
        # (using .get because this parameter is also used by DQN class)
        self.discount = kwargs.get('discount', 0.9)
        self.savedir = kwargs.pop('savedir', os.path.join(os.getcwd(), 'agent_training_stats'))

        # Set up the DQN handle to train the agent.
        net_specs = kwargs.pop('net_specs', default_net_specs)
        net_specs['output_dimension'] = self.action_size
        self.dqn = kwargs.pop('dqn', DQN(net_specs = net_specs, **kwargs))

        # Set up the ReplayBuffer
        self.buffer = kwargs.pop('buffer', ReplayBuffer(**kwargs))

        # Set up a visualiser for the loss
        self.lossVisualiser = kwargs.pop('loss_visualiser', LossVisualiser(Nsteps=self.episode_length,
                                                                           savedir = self.savedir,
                                                                           **kwargs))

        # base class attributes

        self.num_steps_taken = 0
        self.steps_to_target_update = 0
        self.num_episodes = 0
        self.state = None
        self.distance_to_goal = 10e10 # initiate to random high value
        self.testing = False
        self.action = None
        self.logs = {'episode_loss': [], 'losses': [], 'epsilon': [],
                     'collected_reward': [], 'episode_rewards': [], 'episode_length': []}


    ##### class methods #####


    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):

        # if agent ran out of steps
        if self.num_steps_taken > self.episode_length:

            # if we were testing the greedy policy, then it failed to arrive to the goal
            if self.testing:
                print('testing failed, continue to train...')
                self.Egreedy() # set the agent back in egreedy mode
                self.testing = False # unflag self.testing
                self.dqn.update_lr() # decrease learning rate as we are approaching optimal policy
            # else it was a normal training episode, update agent for next episode
            else:
                self.Update()
            # reset variables for next episode
            self.Reset()
            return True

        # if agent has a low epsilon value and arrived to the goal before running out of steps:
        # test its greedy policy!
        elif self.distance_to_goal<0.03 and self.epsilon<0.3:

            # if we were already testing it means that the agent can arrive to the goal using the greedy policy,
            # keep the agent in testing mode until we finish training as we don't need to update it anymore, it works.
            if self.testing:
                # reset variables for next episode
                self.Reset()
                return True
            # test the greedy policy during the next episode
            else:
                print('testing the agents greedy policy...')
                # flag that we are testing
                self.testing = True
                # set agent in greedy mode
                self.Greedy()
                # reset variables for next episode
                self.Reset()
                return True

        # agent has not arrived to the goal nor has run out of steps
        else:
            return False

    # function to update the agent: perform any useful function at the end of an episode
    def Update(self):

        # add to num episodes count
        self.num_episodes+=1

        # store logs
        self.logs['episode_length'].append(self.episode_length)
        self.logs['epsilon'].append(self.epsilon)
        self.logs['losses'].append(self.logs['episode_loss'])
        self.logs['collected_reward'].append(sum(self.logs['episode_rewards']))

        # plot loss (will plot only if ``display`` was set to True in Visualiser)
        self.lossVisualiser.interactive_logs(self.logs)
        self.lossVisualiser.plot_step_loss(self.logs['episode_loss'])

        # every ten episodes save logs to sself.avedir
        if self.num_episodes % 10 == 0 and self.num_episodes>0:
            self.lossVisualiser.plot_logs(self.logs, save=self.num_episodes)
        # save logs up to here in self.savedir
        self.lossVisualiser.plot_logs(self.logs, save="_LAST")

        # decay epsilon according to the chosen decay rate
        self.epsilon *= (1-self.epsilon_decay_rate)
        # decrease episode length according to the chosen decay rate
        self.episode_length = int(max(200, # don't go below 200 steps
                                      self.episode_length*(1-self.episode_length_decay_rate)))


    # function to reset the agent for the next episode
    def Reset(self):
        # restore variables for next episode
        self.logs['episode_loss'] = []
        self.logs['episode_rewards'] = []
        self.num_steps_taken = 0
        self.state = None
        self.action = None

    # set agent in greedy mode
    def Greedy(self):
        self.greedy = True
        self.egreedy = False

    # set agent in egreedy mode
    def Egreedy(self):
        self.egreedy = True
        self.greedy = False

    # set agent in random mode
    def Random(self):
        self.egreedy = False
        self.greedy = False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state = None):
        # store the current state
        self.state = state
        if self.egreedy:
            # see self.get_egreedy_action()
            action  = self.get_egreedy_action(state)
        elif self.greedy:
            # see self.get_greedy_action
            action = self.get_greedy_action(state)
        else:
            # see self.get_random_action
            action = self.get_random_action()
        return action


    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # store distance to goal as an escape condition
        self.distance_to_goal = distance_to_goal
        # evaluate the immediate reward for the current state)
        reward = self.reward(distance_to_goal)
        # add to discounted reward stats
        self.logs['episode_rewards'].append(reward*(self.discount**(self.num_steps_taken-1)))
        # Create a transition
        transition = [self.state, self.action, reward, next_state]
        # Add transition to the buffer
        self.buffer.add(transition)
        # Train Qnetwork
        loss = self.dqn.train_q_network(self.buffer, self.testing)
        # store the loss
        if loss is not None:
            self.logs['episode_loss'].append(loss)
        # update target network every self.dqn.delay steps
        self.steps_to_target_update += 1
        if self.steps_to_target_update % self.dqn.delay == 0:
            self.dqn.update_target_network()
            self.steps_to_target_update = 0

    # function to perform an greedy action
    def get_greedy_action(self, state):
        # increase steps count
        self.num_steps_taken += 1
        # get greedy action and store it
        Q = self.dqn.evaluate(state)
        self.action = np.argmax(Q)
        # return a continuous version of the action
        return self._discrete_action_to_continuous(self.action)

    # function to perform a random action
    def get_random_action(self):
        # increase steps count
        self.num_steps_taken += 1
        #get random action and store it
        self.action =  np.random.choice(self.action_size)
        return self._discrete_action_to_continuous(self.action)

    # function to perform an e-greedy action
    def get_egreedy_action(self, state):
        # chance of taking a random action decays as self.epsilon decays
        if random.random() > self.epsilon:
            return self.get_greedy_action(state)
        # chance of taking a greedy action increases as epsilon decays
        else:
            return self.get_random_action()

    # convert discrete actions (0: up etc) to xy coordinates of the corresponding movement (up, right, down, left)
    def _discrete_action_to_continuous(self, discrete_action):

        if discrete_action == 0:
            # Move 0.02 to the right, and 0 upwards/downwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move 0.02 upwards, and 0 right/left
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 2:
            # Move 0.02 to the left, and 0 upwards/downwards
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        else:
            # Move 0.02 downwards, and 0 right/left
            continuous_action = np.array([0, -0.02], dtype=np.float32)

        return continuous_action





# ====== LOSS VISUALISER CLASS ======

# the following code was used to produce informative visualisations of training logs, not well commented as superflous
class LossVisualiser:
    def __init__(self, Nsteps, savedir, **kwargs):
        '''

        :param Nsteps: int, number of steps in an episode
        :param savedir: str, path to save results
        :param kwargs: dict, additional parameters for the visualiser:
                             - display: bool, flag if displaying losses interactively throughout training.
        '''
        self.Nsteps = Nsteps
        self.savedir = savedir
        self.display = kwargs.pop('display', False)
        self.fig = plt.figure(0)
        if self.display:
            plt.ion()

    def Display(self):
        self.display = True
        self.fig = plt.figure(0)
        if self.display:
            plt.ion()


    def interactive_logs(self, logs):
        if self.display:
            # clear the figure
            plt.clf()
            self.fig.suptitle('Q-network loss', fontsize=12)
            plt.xlabel('steps')
            plt.ylabel('MSE loss')
            plt.plot([], [], 'r', label='current episode')
            plt.plot([], [], 'b', label='past episodes')
            plt.yscale("log")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

            # plot previous losses in blue
            for a, l in enumerate(logs['losses'][:-1], 1):
                plt.plot(range(len(l)), l, c='b', alpha=a / len(logs['losses']))
            plt.xticks(range(0, self.Nsteps + 1, int((self.Nsteps)/10)))
            cbar = self.fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap("Blues")), ticks=[0.25, 0.75], label='episodes')
            cbar.ax.set_yticklabels(['older', 'recent'], rotation=90)
            plt.show()


    def plot_step_loss(self, losses):
        if self.display:
            # plot losses for current episode in red
            plt.plot(range(len(losses)), losses, c='r', ls='--')
            plt.xticks(range(0, self.Nsteps + 1, int((self.Nsteps)/10)))
            plt.yscale("log")
            plt.show()
            # Sleep, so that you can observe the agent moving.
            # Note: this line should be removed when you want to speed up training
            time.sleep(0.01)


    def plot_logs(self, logs, save = None):
        fig1, axs = plt.subplots(2,2, figsize = (10, 5))

        # subplot 1 (average training loss over the episode +/- std)
        mean, std = [], []
        for loss in logs['losses']:
            mean.append(np.mean(loss))
            std.append(np.std(loss))

        axs[0,0].errorbar(range(len(mean)), mean, yerr = std, c='b', ecolor='r')
        axs[0,0].plot([], [], 'b', label='mean')
        axs[0,0].plot([], [], 'r', label='std')
        axs[0,0].legend()
        axs[0,0].set_xticks(range(0, len(mean), max(int(len(mean)/10), 1)))
        axs[0,0].set_xlabel('episodes')
        axs[0,0].set_ylabel(r'MSE[Q(s,a), $\hat{Q}$(s,a)]')
        axs[0,0].set_yscale("log")
        axs[0,0].set_title("average Qnetwork loss per episode")

        # subplot 2, training loss as the episode processes
        for a, l in enumerate(logs['losses'][:-1], 1):
            axs[0,1].plot(range(len(l)), l, c='b', alpha=a / len(logs['losses'][:-1]))
        axs[0,1].plot(range(len(logs['losses'][-1])), logs['losses'][-1], c='r')

        axs[0,1].set_xticks(range(0, self.Nsteps, int(self.Nsteps/5)))
        cbar = fig1.colorbar(cm.ScalarMappable(cmap=plt.get_cmap("Blues")), ax = axs[0,1],
                             ticks=[0.25, 0.75], label='episodes')
        cbar.ax.set_yticklabels(['older', 'recent'], rotation=90)

        axs[0,1].set_xlabel('steps')
        axs[0,1].set_ylabel(r'MSE[Q(s,a), $\hat{Q}$(s,a)]')
        axs[0,1].set_yscale("log")
        axs[0,1].plot([], [], 'r', label='current episode')
        axs[0,1].plot([], [], 'b', label='past episodes')
        axs[0,1].legend()
        axs[0,1].set_title("Q-network loss throughout the episode steps")

        # subplot 3, epsilon decay
        axs[1,0].plot(range(len(logs['epsilon'])), logs['epsilon'])
        axs[1,0].set_xticks(range(0, len(mean), max(int(len(mean)/10), 1)))
        axs[1,0].set_xlabel('episodes')
        axs[1,0].set_ylabel(r'$\epsilon$')
        axs[1,0].set_title(r"$\epsilon$ decay through episodes")

        # subplot 4, total return per episode
        axs[1,1].plot(range(len(logs['collected_reward'])), logs['collected_reward'])
        axs[1,1].set_xticks(range(0, len(mean), max(int(len(mean)/10), 1)))
        axs[1,1].set_xlabel('episodes')
        axs[1,1].set_ylabel('discounted rewards')
        axs[1,1].set_title('cumulative discounted rewards per episode.')

        plt.tight_layout()
        if save is not None:
            if not os.path.exists(self.savedir):
                os.makedirs(self.savedir)
            plt.savefig(os.path.join(self.savedir, 'Q-network_loss_episode{}.png'.format(save)))
        else:
            plt.show()
        plt.close()