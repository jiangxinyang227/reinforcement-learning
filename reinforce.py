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
        self.hidden_dim = 20
        # 初始化存储采样序列中状态，动作，奖励的列表
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # 初始化网络中的各个变量
        self.create_softmax_netword()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_softmax_netword(self):
        """
        对于离散空间用softmax函数来表示策略函数
        :return:
        """
        # 采样中每个时间步的状态
        self.state_input = tf.placeholder("float", [None, self.state_dim], name="state_input")
        # 采样中每个时间步的动作
        self.action_input = tf.placeholder(tf.int32, [None, ], name="action_input")
        # 采样中每个时间步的状态价值（标准化处理了的）
        self.action_values = tf.placeholder(tf.float32, [None, ], name="action_value")

        with tf.variable_scope("softmax_net"):
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
            self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax_input,
                                                                               labels=self.action_input)
            # 策略梯度中真实的损失函数是用每个时间步的动作价值乘以该时间步的交叉熵
            self.loss = tf.reduce_mean(self.neg_log_prob * self.action_values)

            self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

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

    def store_seq(self, s, a, r):
        """
        将采样序列存储起来
        :param s: 状态
        :param a: 动作
        :param r: 奖励
        :return:
        """
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def train_network(self):
        """
        训练网络
        :return:
        """
        # env.step返回的奖励是当前动作，状态的奖励，我们要根据这个奖励来计算序列中每个时间步
        # 下对应的状态价值，状态价值是衰减相加的，如序列长度为t，则最后时刻的Vt = Rt，
        # 而最初时刻v1 = R1 + γ(R2 + γR3 + ...)
        action_values = np.zeros_like(self.ep_rs)
        action_value = 0
        for t in reversed(range(0, len(self.ep_rs))):
            action_value = action_value * GAMMA + self.ep_rs[t]
            action_values[t] = action_value

        # 为了用于之后的损失函数计算，我们将动作价值做标准化
        action_values -= np.mean(action_values)
        action_values /= np.std(action_values)

        # 训练策略网络
        self.session.run(self.train_op,
                         feed_dict={
                             self.state_input: np.vstack(self.ep_obs),
                             self.action_input: np.array(self.ep_as),
                             self.action_values: action_values
                         })


# 模型训练时的超参数
ENV_NAME = "CartPole-v0"
EPISODE = 3000
STEP = 3000
TEST = 10


def main():
    env = gym.make(ENV_NAME)
    agent = Reinforce(env)

    for episode in range(EPISODE):
        # 随机初始化状态向量
        state = env.reset()

        # 执行采样
        for step in range(STEP):
            # 按照softmax的概率选择动作
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            # 将当前序列中采样的每个时间步的状态，动作，奖励都存起来
            agent.store_seq(state, action, reward)
            state = next_state
            if done:
                # 如果获得一条完整的序列，则用这条序列作为一个学习样本去训练网络
                agent.train_network()
                break

        # 执行测试
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.choose_action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST

            print("episode: ", episode, "Evaluation Average Reward: ", ave_reward)


if __name__ == "__main__":
    main()