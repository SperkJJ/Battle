import numpy as np
import random
from emulator import *


class State(object):

    def __init__(self, game_map, player_a, player_b, player_now, r):
        self.game_map = game_map # 12 * 12 matrix, 0 for empty,  1 for walls, 2 for player, 3 for enermy, 4 for gas,5 for props1,6 for props2
        self.playerA = player_a  # (r, c) pos
        self.playerB = player_b  # same as above
        self.player_now = player_now  # 1 for playerA, 2 for playerB
        self.round = r


class Board(object):
    def __init__(self, n_row, n_col, n_walls):
        self.n_actions = 9
        self.n_features = 23
        self.num_map = {'empty': 0, 'obstacle': 1, 'player': 2, 'enemy': 3, 'gas': 4, 'props1': 5, 'props2': 6}
        self.nrow = n_row
        self.ncol = n_col
        self.weapon_A = [-1, -1]
        self.weapon_B = [-1, -1]
        self.reset_wall(n_row, n_col, n_walls)
        self.reset_players()
        self.path = {}

    def reset_wall(self, n_row, n_col, n_walls):
        self.map = np.zeros(shape=(n_row, n_col))  # only include walls and empty
        self.nwalls = n_walls
        self.player_now = 1
        self.round = 1
        self.gas_pos = np.zeros(shape=(n_row, n_col))
        self.weapon_pos = np.zeros(shape=(n_row, n_col))
        self.n_actions = 9
        self.n_features = 144
        # initialize walls randomly
        for _ in range(n_walls):
            r_pos = random.randint(0, n_row - 1)
            c_pos = random.randint(0, n_col - 1)
            if self.is_valid_wall(r_pos, c_pos):
                self.map[r_pos][c_pos] = 1

    def reset_players(self):
        # initialize player's positions randomly
        while True:
            r_pos = random.randint(0, self.nrow - 1)
            c_pos = random.randint(0, self.ncol - 1)
            if self.map[r_pos][c_pos] == 0:
                self.playerA = [r_pos, c_pos]
                break
        while True:
            r_pos = random.randint(0, self.nrow - 1)
            c_pos = random.randint(0, self.ncol - 1)
            if self.map[r_pos][c_pos] == 0 and (self.playerA[0] != r_pos or self.playerA[1] != c_pos):
                self.playerB = [r_pos, c_pos]
                break

    def reset_gas_pos(self):
        self.gas_pos = np.zeros(shape=(self.nrow, self.ncol))
        rn = int((self.round - 1) / 5)
        for i in range(rn):
            self.gas_pos[i, :] = 1
            self.gas_pos[:, i] = 1
        for i in range(12 - rn, 12):
            self.gas_pos[i, :] = 1
            self.gas_pos[:, i] = 1

    def is_valid_wall(self, r_pos, c_pos):
        adj = self.get_adj(r_pos, c_pos)
        if len(adj) < 1:
            return False
        elif len(adj) > 0:
            return True

    def get_adj(self, r_pos, c_pos):
        adj = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_r = r_pos + i
                new_c = c_pos + i
                if new_r < 0 or new_c < 0 or new_r >= self.nrow or new_c >= self.ncol:
                    continue
                if self.map[new_r][new_c] == 0 or self.map[new_r][new_c] == 5 or self.map[new_r][new_c] == 6:
                    adj.append((new_r, new_c))
        return adj

    @staticmethod
    def distance(s1, t1, s2, t2):
        return abs(s1 - s2) + abs(t1 - t2)

    def get_gain(self, r1, c1, r2, c2, r3, c3, total):
        dist = self.distance(r1, c1, r3, c3)
        if dist == 0:
            score_s = 0
        else:
            score_s = total / dist
        dist = self.distance(r2, c2, r3, c3)
        if dist == 0:
            score_t = 0
        else:
            score_t = total / dist
        r = score_t - score_s
        return r

    def step(self, action, start):
        done = False
        terminal = False
        if self.player_now == 1:
            row = self.playerA[0]
            col = self.playerA[1]
            contra_row = self.playerB[0]
            contra_col = self.playerB[1]
        else:
            row = self.playerB[0]
            col = self.playerB[1]
            contra_row = self.playerA[0]
            contra_col = self.playerA[1]

        new_r, new_c, _ = self.get_reward(row, col, action, start)
        # r = 0
        # if new_r, new_c out of board
        if new_r == contra_row and new_c == contra_col:
            done = True
            if self.player_now == 2:
                terminal = True
        elif new_r < 0 or new_c < 0 or new_r >= self.nrow or new_c >= self.ncol \
                or self.map[new_r][new_c] == self.num_map['obstacle'] \
                or new_r == contra_row and new_c == contra_col:
            done = True
            if self.player_now == 2:
                terminal = True

        if action == 0:
            if self.player_now == 1:
                self.weapon_pos = np.zeros(shape=(self.nrow, self.ncol))
                self.weapon_A = [-1, -1]
                self.generate_weapon()
                done = True
            else:
                self.weapon_pos = np.zeros(shape=(self.nrow, self.ncol))
                self.weapon_B = [-1, -1]
                done = True
                terminal = True
        path = np.zeros(shape=(12, 12))
        path[self.playerA[0]][self.playerA[1]] = 1
        m = np.array([self.map,self.gas_pos, path]).reshape(3, 12*12)
        m_t = m.transpose()
        m_t = m_t.reshape(12, 12, 3)
        observation = np.hstack(([self.playerA[0]], [self.playerA[1]],
                                 [self.playerB[0]], [self.playerB[1]],
                                 self.weapon_A, self.weapon_B,
                                 self.playerA, 0, 0))
        r,new_state_tensor, new_state_array, valid, is_done = reward(m_t, observation, action)

        if done:
            self.player_now = 3 - self.player_now
            self.path = {}
        return new_state_tensor, new_state_array, r, done, terminal

    # judge whether the position is adjacent
    def is_adjacent(self, row_1, col_1, row_2, col_2):
        #valid = set([(-1, 0), (1, 0), (0, -1), (0, 1)])
        r_dist = row_1 - row_2
        c_dist = col_1 - col_2
        if abs(r_dist) + abs(c_dist) == 1:
            return True
        #if (r_dist, c_dist) in valid:
         #   return True
        return False

    # give reward or penalty base on state and action
    def get_reward(self, row, col, action, start):
        r = -1
        new_row = row
        new_col = col
        adj = self.get_adj(row, col)
        if action == 0 and len(adj) > 0 and not start:  # stay
            r = 2000 / len(adj)
        elif action == 0 and start:
            r = -1000
        if action == 1 or action == 5 or action == 6:  # up
            new_row -= 1
        if action == 2 or action == 7 or action == 8:  # down
            new_row += 1
        if action == 3 or action == 5 or action == 7:  # left
            new_col -= 1
        if action == 4 or action == 6 or action == 8:  # right
            new_col += 1
        if new_row < 0 or new_col < 0 or new_row >= self.nrow or new_col >= self.ncol or self.map[new_row][new_col] > 0:
            r = -1000
        elif self.gas_pos[new_row][new_col] > 0:
            r = -50
        return new_row, new_col, r

    def update(self, i):
        self.round = i
        if self.round % 5 == 1:
            self.generate_weapon()
        #self.set_danger_region()

    def set_danger_region(self):
        red = int((self.round - 1)/5)
        for i in range(red):
            self.gas_pos[i, :] = 1
            self.gas_pos[:, i] = 1
        for i in range(self.nrow - red, self.nrow):
            self.gas_pos[i, :] = 1
        for i in range(self.ncol - red, self.ncol):
            self.gas_pos[:, i] = 1
        # equal 0
        # random
        # 5/6
        # two available

    def generate_weapon(self):
        while True:
            r_pos = random.randint(0, self.nrow - 1)
            c_pos = random.randint(0, self.ncol - 1)
            if (self.if_available(r_pos, c_pos)):
                if self.weapon_pos[r_pos][c_pos] == 0:
                    self.weapon_pos[r_pos][c_pos] = 1
                    self.weapon_A = [r_pos, c_pos]
                    break
        while True:
            r_pos = random.randint(0, self.nrow - 1)
            c_pos = random.randint(0, self.ncol - 1)
            if (self.if_available(r_pos, c_pos)):
                if self.weapon_pos[r_pos][c_pos] == 0:
                    self.weapon_pos[r_pos][c_pos] = 1
                    self.weapon_B = [r_pos, c_pos]
                    break

    def if_available(self, r_pos, c_pos):
        total_num = 0

        if (r_pos - 1 >= 0 and c_pos - 1 >= 0):
            if (self.map[r_pos - 1][c_pos - 1] == 0):
                total_num = total_num + 1
            if (self.map[r_pos - 1][c_pos] == 0):
                total_num = total_num + 1
            if (self.map[r_pos][c_pos - 1] == 0):
                total_num = total_num + 1
        if (r_pos + 1 < self.nrow and c_pos + 1 < self.ncol):
            if (self.map[r_pos + 1][c_pos + 1] == 0):
                total_num = total_num + 1
            if (self.map[r_pos + 1][c_pos] == 0):
                total_num = total_num + 1
            if (self.map[r_pos][c_pos + 1] == 0):
                total_num = total_num + 1
        if (r_pos + 1 < self.nrow and c_pos - 1 >= 0):
            if (self.map[r_pos + 1][c_pos - 1] == 0):
                total_num = total_num + 1
        if (r_pos - 1 >= 0 and c_pos + 1 < self.ncol):
            if (self.map[r_pos - 1][c_pos + 1] == 0):
                total_num = total_num + 1
        if (total_num >= 2):
            return True
        else:
            return False

ACTIONS = 9

def test():
    board = Board(12, 12, 6)
    for i in range(1, 2):
        print(board.map)
        print(board.playerA, board.playerB)
        board.update(i)
        while True:
            stop = False
            # select action randomly
            action_index = random.randrange(ACTIONS)
            if action_index == 0:
                stop = True
            obs, r, terminal = board.step(action_index)
            print(obs.shape)
            #print(board.playerA, board.playerB, r, board.round)
            #print(obs)
            if terminal:
                break


if __name__ == '__main__':
    test()
