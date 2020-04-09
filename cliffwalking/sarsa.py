#-----EXPLORATION ENVIRONMENT: Cliffwalker-V0----
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import check_test 
from plot_utils import plot_values
import random

env = gym.make("CliffWalking-v0")
# print(env.action_space, env.observation_space)
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13
print(V_opt)

def epsilon_greedy_selection(Q, state, nA, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(env.action_space.n))
    

#-------SARSA(0)---------------------

def updateQ_sarsa(alpha, gamma,Q, state, action, reward, next_state=None, next_action=None):
    
    current = Q[state][action]
    Qsa = Q[next_state][next_action] if next_state is not None else 0
    target = reward + Qsa*gamma
    new_value = current + alpha * (target-current)
    return new_value

def sarsa(env, num_episodes, alpha, gamma=0.9, plot_every =100):
    Q = defaultdict(lambda: np.zeros(env.nA))
    tem_scores = deque(maxlen=plot_every)
    avg_scores = deque(maxlen=num_episodes)
    for i_episode in range(1, num_episodes+1):
        print('computing', i_episode)
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()  
        score = 0
        state = env.reset()
        eps  = 1.0/i_episode
        action = epsilon_greedy_selection(Q, state, env.action_space.n, eps)
        
        while True:
            next_state, reward, done, _ = env.step(action)
            score += reward
            if not done:
                next_action = epsilon_greedy_selection(Q, next_state, env.action_space.n, eps)
                Q[state][action] = updateQ_sarsa(alpha, gamma,Q, state, action,reward, next_state, next_action)  
                state = next_state
                action = next_action
            if done:
                Q[state][action] = updateQ_sarsa(alpha, gamma,Q, state, action,reward, next_state, next_action)
                tem_scores.append(score)
                break
        if(i_episode % plot_every == 0):
            avg_scores.append(np.mean(tem_scores))
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))   

    return Q

if __name__ == '__main__':
    Q_sarsa = sarsa(env, 5000, .01)
    sarsa_policy = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', sarsa_policy)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(sarsa_policy)

    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)

