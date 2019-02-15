import os
import shutil

import threading
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt


GAME = "CartPole-v0"
OUTPUT_GRAPH = True
LOG_DIR = "./log"
N_WORKERS = 3
MAX_GLOBAL_EP = 3000
GLOBAL_NET_SCOPE = "Global_Net"
UPDATE_GLOBAL_ITER = 100
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001
LR_C = 0.001
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
STEP = 3000
TEST = 10

ENV = gym.make(GAME)
STATE_DIM = ENV.observation_space.shape[0]
ACTION_DIM = ENV.action_space.n


class ACNet(object):
    """
    Actor-Critic网络结构
    """
    def __init__(self, scope, globalAC=None):
        """
        初始化
        :param scope: 不同线程下的命名空间
        :param globalAC: 全局ac对象
        :return:
        """
        self.a_hidden_dim = 200
        self.c_hidden_dim = 100

        if scope == GLOBAL_NET_SCOPE:
            # 如果是全局网络
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32, [None, STATE_DIM], name="state_input")
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            # 如果是局部网络（即线程控制的）
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32, [None, STATE_DIM], name="state_input")
                # 序列采样时的动作，可看作真实的动作
                self.action_input = tf.placeholder(tf.int32, [None, ], name="action_input")
                # 序列采样时每个时间步下的动作价值
                self.value_target = tf.placeholder(tf.float32, [None, 1], name="value")
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                # 用采样时的动作价值和由critic网络得到的状态价值计算TD误差
                td = tf.subtract(self.value_target, self.v, name="TD_error")

                with tf.name_scope("c_loss"):
                    # critic网络直接用TD误差的均方来作为损失函数
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope("a_loss"):
                    #
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) *
                                             tf.one_hot(self.action_input, ACTION_DIM, dtype=tf.float32),
                                             axis=1, keepdims=True)

                    # 这里用tf.stop_gradient来组织在优化actor网络时，td的反向传播也会更新critic网络的参数
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)

                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope("local_grad"):
                    # 计算梯度
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope("sync"):
                # 在训练网络时只会用local net中获得的梯度来更新global net的参数
                # local net的参数会间隔性的从global net中复制过来
                with tf.name_scope("pull"):
                    # 将global net的参数复制给local net
                    self.pull_a_params_op = [l_p.assign(g_p)
                                             for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p)
                                             for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope("push"):
                    # 利用local net的梯度来更新global net的参数
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        """
        定义Actor和Critic的网络结构
        :param scope: 命名空间
        :return:
        """
        w_init = tf.random_normal_initializer(0.0, 0.1)
        with tf.variable_scope("actor"):
            l_a = tf.layers.dense(self.state_input, self.a_hidden_dim, tf.nn.relu6,
                                  kernel_initializer=w_init, name="l_a")
            a_prob = tf.layers.dense(l_a, ACTION_DIM, tf.nn.softmax,
                                     kernel_initializer=w_init, name="a_prob")
        with tf.variable_scope("critic"):
            l_c = tf.layers.dense(self.state_input, self.c_hidden_dim, tf.nn.relu6,
                                  kernel_initializer=w_init, name="l_c")
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name="v")

        # 收集actor网络和critic网络的参数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope + "/actor")
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope + "/critic")

        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):
        """
        用梯度来更新global net 的参数
        :param feed_dict:
        :return:
        """
        SESS.run([self.update_a_op, self.update_c_op],
                 feed_dict=feed_dict)

    def pull_global(self):
        """
        将global net的参数复制给local net
        :return:
        """
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, state):
        """
        根据softmax后的动作的概率来选择动作
        :param state: 状态向量
        :return:
        """
        a_prob = SESS.run(self.a_prob,
                          feed_dict={self.state_input: state[np.newaxis, :]})
        action = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

        return action


class Worker(object):
    def __init__(self, worker_name, globalAC):
        """
        创建线程所需要执行的功能类
        :param worker_name:
        :param globalAC:
        """
        self.env = gym.make(GAME).unwrapped
        self.worker_name = worker_name
        # 创建属于每个线程的local net
        self.local_AC = ACNet(worker_name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        # 用来存储一条序列中的状态，动作，奖励
        buffer_s, buffer_a, buffer_r = [], [], []

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            # 获得初始状态
            state = self.env.reset()
            # 用来记录一条序列中的奖励值
            ep_r = 0
            # 死循环去生成序列，终止条件是遇到最终状态
            while True:
                # 通过local net获得动作
                action = self.local_AC.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -5
                ep_r += reward
                # 将单条序列的状态，动作，奖励存储起来
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                # 如果迭代到一定步数时，或者遇到了终止状态，即得到一条完整的序列时，训练global net，并更新local net
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        # 终止状态的状态价值为0
                        next_state_value = 0
                    else:
                        # 非终止状态时就通过local net去获得状态价值
                        next_state_value = SESS.run(self.local_AC.v,
                                                    feed_dict={
                                                        self.local_AC.state_input: next_state[np.newaxis, :]
                                                            })[0, 0]
                    # 创建一个列表存储序列中各时间步下的状态价值
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        next_state_value = r + GAMMA * next_state_value
                        buffer_v_target.append(next_state_value)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), \
                                                          np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.local_AC.state_input: buffer_s,
                        self.local_AC.action_input: buffer_a,
                        self.local_AC.value_target: buffer_v_target
                    }
                    # 利用local net的梯度来更新global net的参数
                    self.local_AC.update_global(feed_dict)
                    # 训练完一次之后将存储列表置为空
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # 用global net的参数复制到local net中
                    self.local_AC.pull_global()

                total_step += 1
                state = next_state

                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)

                    print(
                        self.worker_name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()
    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    # 创建一个global net
    global_AC = ACNet(GLOBAL_NET_SCOPE)

    # 创建一个列表用来存储Worker对象
    workers = []
    for i in range(N_WORKERS):
        worker_name = "worker_{}".format(i)
        # 将创建的worker对象加入到列表中
        workers.append(Worker(worker_name, global_AC))

    COORD = tf.train.Coordinator()  # 创建线程调配器
    # 初始化所有session中的变量
    SESS.run(tf.global_variables_initializer())

    # 执行多线程
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)

    # 用线程调配器管理线程
    COORD.join(worker_threads)

    # 创建测试线程
    testWorker = Worker("test", global_AC)
    testWorker.local_AC.pull_global()

    total_reward = 0
    for i in range(TEST):
        current_state = ENV.reset()
        for j in range(STEP):
            ENV.render()
            action = testWorker.local_AC.choose_action(current_state)
            next_state, reward, done, _ = ENV.step(action)
            total_reward += reward
            if done:
                break
    ave_reward = total_reward / TEST
    print('episode: ', GLOBAL_EP, 'Evaluation Average Reward:', ave_reward)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()


