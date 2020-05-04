import gym



if __name__ == "__main__":
    env = gym.make("SpaceInvaders-ram-v0")
    print('obs space',env.observation_space)
    print('action space', env.action_space)
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
