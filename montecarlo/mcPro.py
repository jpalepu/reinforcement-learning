import gym, sys
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy

env = gym.make("Blackjack-v0")

def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        prob = [0.8,0.2] if state[0] > 18 else [0.2,0.8]
        action = np.random.choice(np.arange(2), p=prob)
        print("action is:", action)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state,action,reward))
        state = next_state
        if done:
            break
    return episode

# for i in range(3):
#     print(generate_episode_from_limit_stochastic(env))

def mc_prediction(env, num_episodes, generate_episode, gamma=0.1):
    return_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for i_episodes in range(1, num_episodes + 1):
        if i_episodes % 1000 == 0:
            print('episode {}/{}.'.format(i_episodes, num_episodes), end="")
            break
        episode = generate_episode(env)
        states, actions, rewards= zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            return_sum[state][actions[i]] += sum(rewards[i:]*discounts[:-(1+i)])
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = return_sum[state][actions[i]]/N[state][actions[i]]
    return Q

# Q = mc_prediction(env, 1000, generate_episode_from_limit_stochastic)
# V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
#     for k, v in Q.items())
# # print(action)
# # plot the state-value function
# plot_blackjack_values(V_to_plot)

def update_Q(env, episode, Q, alpha, gamma):
    states, actions, rewards = zip(*episode)
    discounts = np.array([gamma**i for i in range(len(rewards+1))])
    for i,state in enumerate(states):
        old_Q = Q[states][actions[i]]
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q

def mc_control(env, num_episodes, alpha, gamma=1, eps_start = 1.0, eps_decay = .9999, eps_min= 0.05):
    nA = env.action_space.n
    Q = defaultdict(lambda : np.zeros(nA))
    epsillion = eps_start
    for i_episode in range(1, num_episodes):
        if i_episode % 1000 == 0:
            print("/r Episode {}/{}.".format(i_episode, num_episodes))
            break
        epsillion = max(epsillion*eps_decay, eps_min)
        episode = generate_episode_from_limit_stochastic(env)
        Q = update_Q(env, episode, Q, alpha, gamma)
    policy = dict((k, np.argmax(v)) for k,v in Q.items())
    return policy, Q

policy, Q = mc_control(env,num_episodes= 2000,alpha= 0.02 )
V = dict((k,np.max(v)) for k, v in Q.items())
plot_blackjack_values(V)
plot_policy(policy)