

from agentAtari import Agent
import gym
import torch 

env = gym.make('SpaceInvaders-ram-v0')
agent = Agent(128,6,0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(3):
    state = env.reset()
    for j in range(100000):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break 
env.close()

