# Code for Cartpole
# Random Policy

# Import all the libraries
import gym
import numpy as np
from gym import wrappers

# Load the environment
env = gym.make('CartPole-v0')

done = False
count = 0

# Record the environment
env = wrappers.Monitor(env, 'recording', force=True)

# Reset
observation = env.reset()

while not done:
    # Render the environment on the screen
    env.render()

    count += 1

    # Take a random action
    action = env.action_space.sample()

    # Observe the environment and get the reward
    observation, reward, done, _ = env.step(action)

# Close
env.close()

print('Game Duration: ', count, 'moves')
