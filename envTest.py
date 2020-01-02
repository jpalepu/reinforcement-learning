import gym
import numpy as np

env  = gym.make('FrozenLake-v0')

def value_function(gamma, env):
    value_table = np.zeros(env.observation_space.n)
    no_of_iter = 1000
    threshold = 1e-20
    for i in range(no_of_iter):
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            q_value = []
            for action in range(env.action_space.n):
                next_state_reward = []
                for next_sr in env.P[state][action]:
                    tran_prob,next_state,  reward_prob,  _ = next_sr
                    next_state_reward.append((tran_prob * (reward_prob + gamma + updated_value_table[next_state])))
                q_value.append(np.sum(next_state_reward))
            value_table[state] = max(q_value)
        if(np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return value_table, q_value

def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n) 



optimal_value_function = value_function(env =env, gamma= 0.8)
print(optimal_value_function)

