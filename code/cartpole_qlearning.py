# Code Credits
# Phil Tabor
# https://github.com/philtabor/OpenAI-Cartpole

# Import required libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# Continuous to Discrete Space (Maximum)
MAXSTATES = 10**4

# Discount Rate
GAMMA = 0.9

# Learning Rate
ALPHA = 0.01


def max_dict(d):
    """Finds the maximum key-value pair in a dictionary"""
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v


def create_bins():
    """Convert Continuous Space to Discrete Space for each of the
    four observation values"""
    bins = np.zeros((4, 10))

    # obs[0] -> cart position --- -4.8 - 4.8
    bins[0] = np.linspace(-4.8, 4.8, 10)

    # obs[1] -> cart velocity --- -inf - inf
    bins[1] = np.linspace(-5, 5, 10)

    # obs[2] -> pole angle    --- -41.8 - 41.8
    bins[2] = np.linspace(-.418, .418, 10)

    # obs[3] -> pole velocity --- -inf - inf
    bins[3] = np.linspace(-5, 5, 10)

    return bins


def assign_bins(observation, bins):
    """For each propoerty of the observation, assign it to appropriate bins.
    This will become discrete states."""
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state


def get_state_as_string(state):
    """Convert a state from an integer to a string"""
    string_state = ''.join(str(int(e)) for e in state)
    return string_state


def get_all_states_as_string():
    """Append upto 4 zeros to a state to observe all the bins"""
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4))
    return states


def initialize_Q():
    """Initialise an empty Q-Table"""
    Q = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            # All state values are initialised to 0
            Q[state][action] = 0
    return Q


def play_one_game(bins, Q, eps=0.5):
    """Play one game"""
    observation = env.reset()
    done = False
    cnt = 0  # number of moves in an episode
    state = get_state_as_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        cnt += 1

        # Exploration vs Exploitation
        if np.random.uniform() < eps:
            act = env.action_space.sample()  # epsilon greedy
        else:
            act = max_dict(Q[state])[0]  # choose from the q-table

        # Take a action
        observation, reward, done, _ = env.step(act)

        total_reward += reward

        # If the game finishes prematurely (pole falls over),
        # then penalize the agent for not completing the game.
        if done and cnt < 200:
            reward = -300

        # Update Q-Table
        state_new = get_state_as_string(assign_bins(observation, bins))

        # Get the max key-value pair for the new state
        a1, max_q_s1a1 = max_dict(Q[state_new])

        # Get new values using the Bellman Equation for Q-Learning
        Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, cnt


def play_many_games(bins, N=10000):
    Q = initialize_Q()

    length = []
    reward = []
    for n in range(N):
        # Slowly decrease exploration and start exploitation
        eps = 1.0 / np.sqrt(n+1)

        # Play one episode
        episode_reward, episode_length = play_one_game(bins, Q, eps)

        if n % 100 == 0:
            print("Ep. No:", n, "Exploration:", '%.4f' %
                  eps, "Reward:", episode_reward)
        length.append(episode_length)
        reward.append(episode_reward)

    return length, reward, Q


if __name__ == '__main__':
    bins = create_bins()
    episode_lengths, episode_rewards, Q = play_many_games(bins)

    # Final Game after the training is finished
    observation = env.reset()
    done = False

    while not done:
        env.render()
        state = get_state_as_string(assign_bins(observation, bins))
        act = max_dict(Q[state])[0]
        observation, reward, done, _ = env.step(act)

    env.close()
