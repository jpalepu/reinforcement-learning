import gym
import numpy as np
import random



env = gym.make('Taxi-v3')
# print(env.action_space.n, env.observation_space.n)
action_space = env.action_space


def epsilon_greedy_selection(Q, state, action_space, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(action_space))
    
