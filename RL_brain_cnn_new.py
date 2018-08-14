"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        # consist of [target_net, evaluate_net]
        self._build_net()
        # TODO: 权值更新
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./convolution_graph', self.sess.graph)
        self.saver = tf.train.Saver()

    @staticmethod
    def deep_nn(inputs, scope_name):
        map_ = inputs[:, :-12]
        players = inputs[:, -12:]
        map_ = tf.reshape(map_, (-1, 12, 12, 3))
        with tf.variable_scope(scope_name):
            # conv1
            map_ = tf.layers.conv2d(
                inputs=map_,
                filters=32,
                kernel_size=[7, 7],
                strides=2,
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer())

            # conv2
            map_ = tf.layers.conv2d(
                inputs=map_,
                filters=64,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer())

            # max_pooling1
            map_ = tf.layers.max_pooling2d(inputs=map_, pool_size=[2, 2], strides=2)

            # flat
            map_ = tf.contrib.layers.flatten(map_)

            # map dense1
            map_ = tf.layers.dense(inputs=map_, units=64, activation=tf.nn.relu,
                                   kernel_initializer=tf.glorot_uniform_initializer())

            # player dense1
            players = tf.layers.dense(inputs=players, units=128, activation=tf.nn.relu,
                                      kernel_initializer=tf.glorot_uniform_initializer())

            # player dense2
            players = tf.layers.dense(inputs=players, units=64, activation=tf.nn.relu,
                                      kernel_initializer=tf.glorot_uniform_initializer())

            # aggregate
            out = tf.concat([map_, players], 1)

            # dense1
            out = tf.layers.dense(inputs=out, units=256, activation=tf.nn.relu,
                                  kernel_initializer=tf.glorot_uniform_initializer())

            # out
            out = tf.layers.dense(inputs=out, units=9, kernel_initializer=tf.glorot_uniform_initializer())

        return out

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.q_eval = self.deep_nn(self.s, 'eval_net')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.q_next = self.deep_nn(self.s_, 'target_net')
        tf.summary.scalar('loss', self.loss)

    def store_transition(self, s, a, r, d, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r, d], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\n---------------------target_params_replaced----------------------\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        done_batch = batch_memory[:, self.n_features + 2]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index[done_batch == 0], eval_act_index[done_batch == 0]] \
            = reward[done_batch == 0] + self.gamma * (np.max(q_next, axis=1)[done_batch == 0])
        q_target[batch_index[done_batch != 0], eval_act_index[done_batch != 0]] = reward[done_batch != 0]

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        if self.learn_step_counter % 5 == 0:
            summary = self.sess.run(self.merged, feed_dict={self.s: batch_memory[:, :self.n_features],
                                                            self.q_target: q_target})
            self.train_writer.add_summary(summary, self.learn_step_counter)
            self.saver.save(self.sess, './new_log/my_test_model', global_step=self.learn_step_counter)

        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



