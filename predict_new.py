import tensorflow as tf
import numpy as np
from cnn_board import *

class PredictProcessor(object):

    def __init__(self, log_path=r'./log'):
        self.map_ = tf.placeholder(tf.float32, [None, 12, 12, 3], name='map')
        self.players = tf.placeholder(tf.float32, [None, 4], name='players')
        self.q_eval = self.deep_nn(self.map_, self.players, 'eval_net')
        self.__sess = tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(log_path)
        saver.restore(self.__sess, ckpt.model_checkpoint_path)

    @staticmethod
    def deep_nn(map_, players, scope_name):
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
            out = tf.layers.dense(inputs=out, units=64, activation=tf.nn.relu,
                                  kernel_initializer=tf.glorot_uniform_initializer())

            # out
            out = tf.layers.dense(inputs=out, units=9, kernel_initializer=tf.glorot_uniform_initializer())

        return out

    @staticmethod
    def next(state, action):
        # input 12*12*5
        next_state = None
        cur = state.reshape(12 * 12, 5).transpose().reshape(5, 12, 12)
        game_map = cur[0]
        #print(game_map)
        playerA = cur[1]
        #print(playerA)
        playerB = cur[2]
        #print(playerB)
        gas_pos = cur[3]
        #print(gas_pos)
        weapon_pos = cur[4]
        pos = np.nonzero(playerA > 0)
        #print(pos)
        row = pos[0]
        col = pos[1]
        new_r = row
        new_c = col
        if action == 1 or action == 5 or action == 6:  # up
            new_r -= 1
        elif action == 2 or action == 7 or action == 8:  # down
            new_r += 1
        elif action == 3 or action == 5 or action == 7:  # left
            new_c -= 1
        elif action == 4 or action == 6 or action == 8:  # right
            new_c += 1

        is_valid = True
        if new_r < 0 or new_c < 0 or new_r >= 12 or new_c >= 12:
            is_valid = False
        if game_map[new_r, new_c] > 0:
            is_valid = False
        if weapon_pos[new_r, new_c] > 0:
            weapon_pos[new_r, new_c] = 0
        playerA = np.zeros(shape=(12, 12))
        playerA[new_r, new_c] = 1
        new_state = np.array([game_map, playerA, playerB, gas_pos, weapon_pos])
        return is_valid, new_state.reshape(5, 12 * 12).transpose().reshape(12, 12, 5)

    def predict(self, map_, players):
        action_list = []
        y = self.__sess.run(self.q_eval, feed_dict={self.map_: map_, self.players: players})
        action_list.append(np.argmax(y))
        return action_list


if __name__ == '__main__':
    prdt = PredictProcessor(log_path="../new_log")
    print(prdt.predict([np.random.choice(1, size=432).reshape((12, 12, 3))], [[1, 2, 3, 4]]))
