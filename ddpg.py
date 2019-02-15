import tensorflow as tf
import numpy as np
import gym

MAX_EPISODES = 2000
MAX_EP_STEPS = 200
LR_A = 0.001  # actor网络的学习速率
LR_C = 0.002  # critic网络的学习速率
GAMMA = 0.9  # 奖励衰减因子
TAU = 0.01  # 软参数复制，在将原网络中的参数复制给目标网络时用的
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, env):
        self.pointer = 0
        self.sess = tf.Session()

        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.a_bound = env.action_space.high
        # 经验回放池子
        self.memory = np.zeros((MEMORY_CAPACITY, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        # 当前状态
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        # 下一状态
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            # 训练原网络，目标网络不训练，只是用来生成动作的
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # 原网络的q当作预测值，目标网络得到的q_是用来计算目标值的，这里得动作直接用actor网络的结果，
            # 这样在通过q值反向传播时才能更新actor网络的参数
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # 收集四个网络的参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # 通过软复制的方式对目标actor和目标critic网络进行参数替换
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # 得到目标q值
        q_target = self.R + GAMMA * q_
        # 得到td误差，直接将该td误差当作critic网络的损失函数
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        # 创建critic网络的训练入口op
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        # 直接将预测出来的q值得负均值作为actor网络得损失函数
        a_loss = - tf.reduce_mean(q)
        # 创建actor网络的训练入口op，最小化损失实际上是最大化预测的q值
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        """
        选择动作，用原网络选择
        :param s:
        :return:
        """
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # 利用软替换将原网络的参数复制给目标网络
        self.sess.run(self.soft_replace)

        # 采样
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # 训练网络
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        """
        将状态，动作，奖励，下一状态存入到池子中
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        """
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        """
        actor网络结构
        :param s:
        :param scope:
        :param trainable:
        :return:
        """
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # 对返回的动作进行缩放处理 a_bound = 2
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        """
        critic网络结构，网络的输入有状态和动作两个，通过网络将两个整合到一起
        :param s:
        :param a:
        :param scope:
        :param trainable:
        :return:
        """
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)
    # 实例化ddpg对象
    ddpg = DDPG(env)

    # ddpg中在选择动作时，对选择的动作增加了噪声，主要是增加一些随机性，因为ddpg是确定性策略，动作都是确定的，
    # 这里的方差就给动作添加噪声用的
    var = 3  # control exploration

    for episode in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # 动过actor网络选择动作
            a = ddpg.choose_action(s)
            # 对选择的动作增加一些噪声，主要是增加一些随机性
            a = np.clip(np.random.normal(a, var), -2, 2)
            s_, r, done, info = env.step(a)
            # 将采样序列加入到经验回放池中
            ddpg.store_transition(s, a, r / 10, s_)
            # 当池子中的样本数大于某一个值，开始训练网络
            if ddpg.pointer > MEMORY_CAPACITY:
                # 衰减方差，减少动作选择的随机性
                var *= .9995
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(10):
                state = env.reset()
                for j in range(MAX_EP_STEPS):
                    env.render()
                    action = ddpg.choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / 300
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
