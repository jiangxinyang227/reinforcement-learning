import gym

env = gym.make("CartPole-v0")
state = env.reset()
print(state)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.shape)