import time
import numpy as np
from random_environment import Environment
from agent import Agent
import argparse

# parsing arguments
parser = argparse.ArgumentParser(description='train an agent using DQN to escape a randomly built maze.')
parser.add_argument('--train_time', '-t',  type=int, default=600, help='training time (in seconds).')
parser.add_argument('--display', action='store_true', help='flag to display the environment at every step.')
parser.add_argument('--display_after',  type=int, default=None, help='if display was flagged, set how much time (second to go through\n'
                                                                  ' before starting to display results on screen. (code will be faster)')
parser.add_argument('--magnification',  type=int, default=500, help='magnification factor to display the environment.')
parser.add_argument('--savedir', default = "./training_stats/", help = 'path to save any output the code might produce')

# controls DQN
parser.add_argument('--DoubleQLearning', action='store_true', help='if using double QLearning strategy (with target network).\n'
                                                                   'if this argument is passed --QLearning will be set to False.')
parser.add_argument('--QLearning', action='store_true', help='if using QLearning strategy (without target network).\n'
                                                             'if this argument is parsed, --DoubleQLearning will be se to False.')
parser.add_argument('--learning_rate', '-lr',  type=float, default=0.0002, help='learning rate for the Q network.')
parser.add_argument('--delay', type=int, default=500, help='delay (in steps) after which the target network is updated.')

# controls agent
parser.add_argument('--episode_length',  type=int, default=1000, help='maximum number of steps in an episode. (default = 1000)')
parser.add_argument('--episode_length_decay_rate',  type=float, default=0.025, help='rate of decay of the episode length. (default = 2.5%)')
parser.add_argument('--epsilon',  type=float, default=1, help='intial value of epsilon for egreedy policy (in [0,1]). (default = 1.0)')
parser.add_argument('--epsilon_decay_rate',  type=float, default=0.025, help='rate of decay of epsilon for egreedy policy. (default = 2.5%)')
parser.add_argument('--discount', '-d',  type=float, default=0.999, help='discount factor for future rewards. (default = 0.999)')

# controls buffer
parser.add_argument('--buffer_capacity',  type=int, default=5000, help='maximum length (capacity) for the replay buffer.')
parser.add_argument('--batch_size', '-bs',  type=int, default=128, help='batch size to train the Q-network.')
parser.add_argument('--alpha',  type=float, default=0.01, help='hyperparamter for prioritised sampling in (0,1). (default = 1%)')
parser.add_argument('--prioritised_experience', action='store_true', help='flag to use prioritised replay sampling. (default = False)')
args = vars(parser.parse_args())

# Main entry point
if __name__ == "__main__":

    display = args.pop('display')
    display_after = args.pop('display_after')
    if display_after:
        print("waiting %d seconds before displaying environment."%display_after)
        display = True

    train_time = args.pop('train_time')
    # Create a random environment fixing the random seed
    random_seed = int(time.time())
    np.random.seed(random_seed)
    environment = Environment(args.pop('magnification'))

    # Create an agent
    assert not (args['QLearning'] and args['DoubleQLearning']), "ERROR: both --QLearning and --DoubleQLearning were parsed, chose only one!"
    args['DoubleQLearning'] = not args.pop('QLearning')
    # instanciate agent, DQN and buffer all from the agent class
    agent = Agent(**args)

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + train_time
    display_after_time = start_time + display_after

    # Train the agent, until the time is up
    while time.time() < end_time:

        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state

        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display and time.time()>display_after_time:
            environment.show(state)

    # Test the agent for 100 steps, using its greedy policy
    while True:
        state = environment.init_state
        agent.Reset()
        has_reached_goal = False
        for step_num in range(100):
            action = agent.get_greedy_action(state)
            next_state, distance_to_goal = environment.step(state, action)
            environment.show(state)
            time.sleep(0.02)
            # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
            if distance_to_goal < 0.03:
                has_reached_goal = True
                break
            state = next_state

        # Print out the result
        if has_reached_goal:
            print('Reached goal in ' + str(step_num) + ' steps.')
        else:
            print('Did not reach goal. Final distance = ' + str(distance_to_goal))