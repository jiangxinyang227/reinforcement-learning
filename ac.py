import gym
import tensorflow as tf
import numpy as np


# 定义Actor模型超参数
GAMMA = 0.95  # 衰减系数
LEARNING_RATE = 0.01


# 定义Critic模型超参数
EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32
REPLACE_TARGET_FREQ = 10


class Actor(object):
    """
    Reinforce 算法
    """
    def __init__(self, env, sess):
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hidden_dim = 20

        # 初始化网络中的各个变量
        self.create_softmax_netword()
        self.session = sess
        self.session.run(tf.global_variables_initializer())

    def create_softmax_netword(self):
        """
        对于离散空间用softmax函数来表示策略函数
        :return:
        """
        # 采样中每个时间步的状态
        self.state_input = tf.placeholder("float", [None, self.state_dim], name="state_input")
        # 采样中每个时间步的动作
        self.action_input = tf.placeholder(tf.int32, [None, self.action_dim], name="action_input")
        # 这里是利用Critic网络计算出来的TD误差
        self.td_errors = tf.placeholder(tf.float32, None, name="action_value")

        with tf.variable_scope("Actor_net"):
            W1 = tf.get_variable("W1", shape=[self.state_dim, self.hidden_dim],
                                 initializer=tf.truncated_normal_initializer())
            b1 = tf.Variable(tf.constant(0.01, shape=[self.hidden_dim]), name="b1")
            W2 = tf.get_variable("W2", shape=[self.hidden_dim, self.action_dim],
                                 initializer=tf.truncated_normal_initializer())
            b2 = tf.Variable(tf.constant(0.01, shape=[self.action_dim]), name="b2")

            h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            self.softmax_input = tf.matmul(h_layer, W2) + b2

            # 通过softmax预测每个时间步下各个动作被选择的概率，其实类似于softmax多分类，
            # 在这里将动作空间看作类别，所以这个tensor就是softmax网络的输出结果
            self.act_pred_prob = tf.nn.softmax(self.softmax_input, name="act_pred_prob")

            # 构造交叉上损失函数，利用预测的结果和采样中的结果（看作真实的结果）的差异来构造目标函数
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.softmax_input,
                                                                               labels=self.action_input)
            # 策略梯度中真实的损失函数是用每个时间步的动作价值乘以该时间步的交叉熵
            self.loss = tf.reduce_mean(self.neg_log_prob * self.td_errors)

            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(-self.loss)

    def choose_action(self, state):
        """
        根据输入的状态选择动作
        :param state: 状态向量
        :return:
        """
        act_pred_prob = self.session.run(self.act_pred_prob,
                                         feed_dict={self.state_input: [state]})
        # 根据每个动作的softmax概率来选择动作，参数p表示的就是每个动作的概率值，
        # 这种思想也符合多分类的思想
        action = np.random.choice(range(act_pred_prob.shape[1]), p=act_pred_prob.ravel())
        return action

    def train_network(self, state, action, td_error):
        """
        训练Actor网络
        :param state:
        :param action:
        :param td_error:
        :return:
        """
        s = state[np.newaxis, :]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        a = one_hot_action[np.newaxis, :]
        # 训练策略网络
        self.session.run(self.train_op,
                         feed_dict={
                             self.state_input: s,
                             self.action_input: a,
                             self.td_errors: td_error
                         })


class Critic(object):
    """
    Critic网络模型
    """
    def __init__(self, env, sess):
        self.time_step = 0
        self.epsilon = EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.output_dim = 1
        self.hidden_dim = 20

        # 初始化网络中的变量
        self.create_Q_network()
        self.create_training_method()

        self.session = sess
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        """
        只用了简单的dqn网络
        :return:
        """
        self.state_input = tf.placeholder("float", [None, self.state_dim], name="state_input")
        with tf.variable_scope("Critic_net"):
            W1 = tf.get_variable("W1", shape=[self.state_dim, self.hidden_dim],
                                 initializer=tf.truncated_normal_initializer())
            b1 = tf.Variable(tf.constant(0.01, shape=[self.hidden_dim]), name="b1")
            W2 = tf.get_variable("W2", shape=[self.hidden_dim, self.action_dim],
                                 initializer=tf.truncated_normal_initializer())
            b2 = tf.Variable(tf.constant(0.01, shape=[self.action_dim]), name="b2")

            h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            self.Q_value = tf.matmul(h_layer, W2) + b2

    def create_training_method(self):
        """
        构建训练方法
        :return:
        """
        self.next_state_value = tf.placeholder(tf.float32, [None, self.action_dim], name="next_state_value")
        self.reward = tf.placeholder(tf.float32, None, "reward")

        with tf.variable_scope("Critic_loss"):
            self.td_error = self.reward + GAMMA * tf.reduce_mean(self.next_state_value) - tf.reduce_mean(self.Q_value)
            self.loss = tf.square(self.td_error)

        with tf.variable_scope("Critic_train"):
            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def train_Q_network(self, state, reward, next_state):
        """
        训练Critic网络
        :param state: 当前状态
        :param reward: 奖励
        :param next_state: 下一状态
        :return:
        """
        current_state, next_state = state[np.newaxis, :], next_state[np.newaxis, :]
        next_state_value = self.session.run(self.Q_value,
                                            feed_dict={self.state_input: next_state})

        td_error, _ = self.session.run([self.td_error, self.train_op],
                                       feed_dict={
                                           self.state_input: current_state,
                                           self.next_state_value: next_state_value,
                                           self.reward: reward
                                       })
        return td_error


# 模型训练时的超参数
ENV_NAME = "CartPole-v0"
EPISODE = 3000
STEP = 3000
TEST = 10


def main():
    sess = tf.InteractiveSession()
    env = gym.make(ENV_NAME)
    actor = Actor(env, sess)
    critic = Critic(env, sess)

    for episode in range(EPISODE):
        # 随机初始化状态向量
        state = env.reset()

        # 开始训练
        for step in range(STEP):
            # 按照softmax的概率选择动作
            action = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # 用当前的一条数据来更新Q网络，并且用Critic来计算出Td误差，用来作为Actor中损失计算的一部分
            td_error = critic.train_Q_network(state, reward, next_state)
            # 每次用一条数据来更新Actor网络
            actor.train_network(state, action, td_error)
            state = next_state
            if done:
                break

        # 执行测试
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST

            print("episode: ", episode, "Evaluation Average Reward: ", ave_reward)


if __name__ == "__main__":
    main()