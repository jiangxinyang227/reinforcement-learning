import gym
import tensorflow as tf
import numpy as np


# 定义模型超参数
GAMMA = 0.95  # 衰减系数
LEARNING_RATE = 0.01


class Reinforce(object):
    """
    Reinforce 算法
    """
    def __init__(self, env):
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []