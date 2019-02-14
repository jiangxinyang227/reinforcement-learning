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

OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

ENV = gym.make(GAME)
STATE_DIM = ENV.observation_space.shape[0]
ACTION_DIM = ENV.action_space.n


class ACNet(object):
    """
    Actor-Critic网络结构
    """
    def __init__(self, scope, sess, globalAC=None):
        """
        初始化
        :param scope: 不同线程下的命名空间
        :param globalAC: 全局ac对象
        :return:
        """
        self.a_hidden_dim = 200
        self.c_hidden_dim = 100
        self.session = sess

        if scope == GLOBAL_NET_SCOPE:
            # 如果是全局网络
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32, [None, STATE_DIM], name="state_input")
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            # 如果是局部网络（即线程控制的）
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32, [None, STATE_DIM], name="state_input")
                self.action_input = tf.placeholder(tf.int32, [None, ], name="action_input")
                self.value_target = tf.placeholder(tf.float32, [None, 1], name="value")
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.value_target, self.v, name="TD_error")

                with tf.name_scope("c_loss"):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope("a_loss"):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) *
                                             tf.one_hot(self.action_input, ACTION_DIM, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob + tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope("local_grad"):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope("sync"):
                with tf.name_scope("pull"):
                    self.pull_a_params_op = [l_p.assign(g_p)
                                             for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p)
                                             for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope("push"):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        """
        定义Actor和Critic的网络结构
        :param scope: 命名空间
        :return:
        """
        w_init = tf.random_normal_initializer(0.0, 1.0)
        with tf.variable_scope("actor"):
            l_a = tf.layers.dense(self.state_input, self.a_hidden_dim, tf.nn.relu6,
                                  kernel_initializer=w_init, name="l_a")
            a_prob = tf.layers.dense(l_a, ACTION_DIM, tf.nn.softmax,
                                     kernel_initializer=w_init, name="a_prob")
        with tf.variable_scope("critic"):
            l_c = tf.layers.dense(self.state_input, self.c_hidden_dim, tf.nn.relu6,
                                  kernel_initializer=w_init, name="l_c")
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name="v")

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
        self.session.run([self.update_a_op, self.update_c_op],
                         feed_dict=feed_dict)

    def pull_global(self):
        """
        将global net的参数复制给local net
        :return:
        """
        self.session.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, state):
        """
        根据softmax后的动作的概率来选择动作
        :param state: 状态向量
        :return:
        """
        a_prob = self.session.run(self.a_prob,
                                  feed_dict={self.state_input: state[np.newaxis, :]})
        action = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

        return action


class Worker(object):
    def __init__(self, worker_name, sess, coord, globalAC):
        """

        :param worker_name:
        :param sess:
        :param coord:
        :param globalAC:
        """
        self.env = gym.make(GAME).unwrapped
        self.worker_name = worker_name
        self.local_AC = ACNet(worker_name, sess, globalAC)
        self.coord = coord
        self.session = sess

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while not self.coord.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state = self.env.reset()
            ep_r = 0
            while True:
                action = self.local_AC.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -5
                ep_r += reward
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        next_state_value = 0
                    else:
                        next_state_value = self.session.run(self.local_AC.v,
                                                            feed_dict={
                                                                self.local_AC.state_input: next_state[np.newaxis, :]
                                                            })[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        next_state_value = r + GAMMA * next_state_value
                        buffer_v_target.append(next_state_value)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), \
                                                          np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.local_AC.state_input: buffer_s,
                        self.local_AC.action_input: buffer_a,
                        self.local_AC.value_target: buffer_v_target
                    }
                    self.local_AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.local_AC.pull_global()

                state = next_state
                total_step += 1

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


def main():
    sess = tf.Session()
    # 创建一个global net
    global_AC = ACNet(GLOBAL_NET_SCOPE, sess)

    coord = tf.train.Coordinator()  # 创建线程调配器
    # 用来存储Worker对象
    workers = []
    for i in range(N_WORKERS):
        worker_name = "worker_{}".format(i)
        # 将创建的worker对象加入到列表中
        workers.append(Worker(worker_name, sess, coord, global_AC))

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

    testWorker = Worker("test", sess, coord, global_AC)
    testWorker.local_AC.pull_global()

    total_reward = 0
    for i in range(TEST):
        state = ENV.reset()
        for j in range(STEP):
            ENV.render()
            action = testWorker.local_AC.choose_action(state)  # direct action for test
            state, reward, done, _ = ENV.step(action)
            total_reward += reward
            if done:
                break
    ave_reward = total_reward / TEST
    print('episode: ', GLOBAL_EP, 'Evaluation Average Reward:', ave_reward)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()


if __name__ == "__main__":
    main()