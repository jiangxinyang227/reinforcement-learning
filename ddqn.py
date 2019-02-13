import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque


# 定义DQN的超参数
GAMMA = 0.9  # 衰减系数
INITIAL_EPSILON = 0.5  # epsilon 贪婪法中的epsilon初始值
FINAL_EPSILON = 0.01  # epsilon 贪婪法中的ε最终值，ε值在减小，说明随机性越来越低
REPLAY_SIZE = 10000  # 经验回放池的尺寸大小
BATCH_SIZE = 32  # batch size 的尺寸大小
REPLACE_TARGET_FREQ = 10  # 更新目标网络参数的频率


class DQN(object):
    """
    定义DQN引擎
    """
    def __init__(self, env):
        """
        初始化一些实例对象
        :param env: 外部环境
        """
        self.replay_buffer = deque()  # 将经验回放池初始化为一个队列
        self.time_step = 0  # 记录训练的步数
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]  # 状态空间的维度
        self.action_dim = env.action_space.n  # 动作空间的维度
        self.hidden_dim = 20

    def create_Q_network(self):
        """
        定义q网络，在这里用dnn来实现，所有的动作和状态都是索引表示
        :return:
        """
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        with tf.variable_scope("current_net"):
            W1 = tf.get_variable("W1", shape=[self.state_dim, self.hidden_dim],
                                 initializer=tf.truncated_normal_initializer())
            b1 = tf.Variable(tf.constant(0.01, shape=[self.hidden_dim]), name="b1")

            W2 = tf.get_variable("W2", shape=[self.hidden_dim, self.action_dim],
                                 initializer=tf.truncated_normal_initializer())
            b2 = tf.Variable(tf.constant(0.01, shape=[self.action_dim]), name="b1")

            h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            self.Q_value = tf.matmul(h_layer, W2) + b2

        # 定义目标网络，网络结构和上面的网络结构一致
        with tf.variable_scope("target_net"):
            W1_t = tf.get_variable("W1", shape=[self.state_dim, self.hidden_dim],
                                   initializer=tf.truncated_normal_initializer())
            b1_t = tf.Variable(tf.constant(0.01, shape=[self.hidden_dim]), name="b1")

            W2_t = tf.get_variable("W2", shape=[self.hidden_dim, self.action_dim],
                                   initializer=tf.truncated_normal_initializer())
            b2_t = tf.Variable(tf.constant(0.01, shape=[self.action_dim]), name="b1")

            h_layer_t = tf.nn.relu(tf.matmul(self.state_input, W1_t) + b1_t)
            self.target_Q_value = tf.matmul(h_layer_t, W2_t) + b2_t

        # 拿到Q网络的参数和目标Q网络的参数
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        print(t_params)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
        print(e_params)
        # 将Q网络的参数复制给目标Q网络
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_training_method(self):
        """
        定义训练方法
        :return:
        """
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # ont-hot表示
        self.y_input = tf.placeholder("float", [None])
        # tf.multiply是元素对应相乘的乘法，Q_action是获得当前状态下的Q值
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # 利用当前状态的Q值和下一状态的Q值（目标Q值）计算均方误差，不过当前状态的Q值是用
        # ε-贪婪法得到的，而下一状态的Q值是max最大动作的。
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def train_Q_network(self):
        """
        训练Q网络
        :return:
        """
        self.time_step += 1
        # 在经验回放池中随机采样
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        # 将各种数据依次拿出来
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # 通过Q网络计算出目标Q值
        y_batch = []
        # 通过原Q网络计算出下一状态的Q值
        current_Q_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # 得到Q值最大的动作
        max_action_next = np.argmax(current_Q_batch, axis=1)
        # 通过目标Q网络计算出下一状态的Q值
        target_Q_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                # 如果是终止状态，则当前的Q值等于当前的奖励
                y_batch.append(reward_batch[i])
            else:
                # 在更新Q网络时的目标Q值选取由原Q网络得到的最大动作对应的Q值
                target_Q_value = target_Q_batch[i, max_action_next[i]]
                y_batch.append(reward_batch[i] + GAMMA * target_Q_value)

        self.optimizer.run(feed_dict={self.y_input: y_batch,
                                      self.action_input: action_batch,
                                      self.state_input: state_batch})

    def update_target_q_network(self, episode, session):
        """
        更新目标网络参数
        :param episode:
        :param session:
        :return:
        """
        if episode % REPLACE_TARGET_FREQ == 0:
            session.run(self.target_replace_op)

    def egreedy_action(self, state):
        """
        根据当前的q网络计算所有动作对应的q值，然后基于egreedy来选择当前的动作
        :param state: 状态的向量表示
        :return:
        """
        # 先获得q值, 获得是一个维度等于action_dim的向量, 向量中每个值可以认为是每个动作对应的Q值
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]

        # 采用ε-贪婪法来选择动作，如果随机的数小于epsilon值，则随机一个动作，否则选q值最大的动作
        if random.random() <= self.epsilon:
            if self.epsilon > FINAL_EPSILON:
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            if self.epsilon > FINAL_EPSILON:
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def perceive(self, state, action, reward, next_state, done):
        """
        将五元组加入到经验回放池中，并且提供Q网络训练的入口
        :param state: 当前状态向量
        :param action: 当前动作索引
        :param reward: 奖励
        :param next_state: 下一状态向量
        :param done: 是否终止状态
        :return:
        """
        # 将动作用one-hot向量表示
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        # 将五元组添加到经验回放池中
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # 如果池子大于最大值，则从左边删除
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        # 如果池子中的五元组数量大于batch size的值，则开始训练Q网络
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]})[0])


# 定义一些训练时的超参数
ENV_NAME = "CartPole-v0"  # 游戏的名称/环境的名称
EPISODE = 3000  # 定义了状态序列的数量
STEP = 300  # 每条序列的长度
TEST = 10  # 测试时的序列数量


def main():
    env = gym.make(ENV_NAME)  # 初始化环境
    agent = DQN(env)  # 初始化agent

    # 初始化一些变量，必须在这里初始化，因为之后要session初始化变量
    agent.create_Q_network()
    agent.create_training_method()

    # 初始化session，采用tf.InteractiveSession()可以直接调用eval方法返回值
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    for episode in range(EPISODE):
        # 随机初始化状态，得到其状态向量
        state = env.reset()

        # 训练一条序列
        for step in range(STEP):
            # 通过ε贪婪法获得当前的动作
            action = agent.egreedy_action(state)
            # env.step可以在给定当前动作下获得下一个状态，奖赏等值
            next_state, reward, done, _ = env.step(action)
            # 定义reward
            reward = -1 if done else 0.1
            # 将五元组存到经验回放池中
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        # 每100个episode测试一次
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # 用动画的形式渲染出来
                    env.render()
                    # 测试时的动作用max最大动作选择
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)

        # 更新目标网络参数
        agent.update_target_q_network(episode, session)


if __name__ == "__main__":
    main()