import numpy as np
from collections import defaultdict

class Agent:

    def update(self, Q_sa, q_next,reward):
        Q_sa = Q_sa + self.alpha * (reward + ( self.gamma * q_next) - Q_sa)
        return Q_sa

    def greedy_policy(self,Q_state,i_episode,eps = None ):
        
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy = np.ones(self.nA) * epsilon / self.nA
        policy[np.argmax(Q_state)] = 1 - epsilon + (epsilon / self.nA)

        return self.policy

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.9
        self.eps = 0.05
        self.gamma = 1.0
        self.i_episode = 1
        self.policy = None
    
        


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.policy = self.greedy_policy(self.Q[state], self.i_episode)
        self.i_episode += 1
        return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.Q[state][action] = self.update(self.Q[state][action], np.max(self.Q[next_state]), reward)
