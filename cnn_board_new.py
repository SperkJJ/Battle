import numpy as np
import random
from log import TheLogger

NUM = 12
ACTIONS = 9

class Board(object):
    def __init__(self, n_row, n_col, n_walls):
        self.n_actions = 9
        self.n_features = 444
        self.round = 1
        self.player_now = 1
        self.num_map = {'empty': 0, 'obstacle': 1, 'player': 2, 'enemy': 3, 'gas': 4, 'props1': 5, 'props2': 6}
        self.nrow = n_row
        self.ncol = n_col
        self.weapon_A = [-1,-1]
        self.weapon_B = [-1,-1]
        self.playerA_weapon_create = False
        self.playerB_weapon_create = False
        self.player_now = 1
        self.playerA = None
        self.playerB = None
        self.gas_pos = np.zeros(shape=(n_row, n_col))
        self.weapon_pos = np.zeros(shape=(n_row, n_col))
        self.reset_wall(n_row, n_col, n_walls)
        self.reset_players()
        self.reset_weapon_pos()

    def reset_wall(self, n_row, n_col, n_walls):
        self.map = np.zeros(shape=(n_row, n_col))  # only include walls and empty
        self.nwalls = n_walls

        # initialize walls randomly
        for _ in range(n_walls):
            while True:
                r_pos = random.randint(0, n_row - 1)
                c_pos = random.randint(0, n_col - 1)
                if self.map[r_pos][c_pos] == 0:
                    self.map[r_pos][c_pos] = 1
                    break

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
        self.gas_pos = np.ones(shape=(self.nrow, self.ncol))
        r = int(self.round / 5)
        self.gas_pos[r:self.nrow - r][r:self.ncol - r] = 0

    def reset_weapon_pos(self):
        # init weapon_pos
        self.weapon_pos = np.zeros(shape=(self.nrow, self.ncol))
        self.weapon_A = [-1,-1]
        self.weapon_B = [-1,-1]

    def is_valid_wall(self, r_pos, c_pos):
        adj = self.get_adj_eight(r_pos, c_pos)
        return len(adj) >= 2

    def get_adj_four(self, r_pos, c_pos):
        adj = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if abs(i) + abs(j) != 1:
                    continue
                new_r = r_pos + i
                new_c = c_pos + i
                if new_r < 0 or new_c < 0 or new_r >= self.nrow or new_c >= self.ncol:
                    continue
                if self.map[new_r][new_c] == 0:
                    adj.append((new_r, new_c))
        return adj

    def get_adj_eight(self, r_pos, c_pos):
        adj = []
        for i in range(-1, 2):
            new_r = r_pos + i
            if new_r < 0 or new_r >= self.nrow :
                continue
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                new_c = c_pos + j
                if new_c < 0 or new_c >= self.ncol:
                    continue
                if self.map[new_r][new_c] == 0:
                    adj.append((new_r, new_c))
        return adj

    @staticmethod
    def distance(s1, t1, s2, t2):
        return abs(s1 - s2) + abs(t1 - t2)

    def get_gain(self, r1, c1, r2, c2, r3, c3, total):
        dist = self.distance(r1, c1, r3, c3)
        score_s = total / dist
        dist = self.distance(r2, c2, r3, c3)
        score_t = total / dist
        r = score_t - score_s
        return r

    # add stop action, stop when action == 8
    def step(self, action):
        done = False  # record whether to change to next player
        terminal = False  # record whether to get to next round
        invalid = False  # record whether the action is valid
        r = 0  #default reward is zero
        # set row, col to current player, contra_row, contra_col to enemy
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
        # get next position
        new_r, new_c = Board.get_next_pos(row, col, action)
        # decide whether next positon is valid, is done or is terminal

        # if next pos is in enemy or invalid, done = True, if current is player2, then terminal = True
        if new_r == contra_row and new_c == contra_col or new_r < 0 or new_c < 0 or new_r >= self.nrow or new_c >= self.ncol or self.map[new_r][new_c] > 0:
            # if next position is invalid, then operate as before
            r = -1
            invalid = True
            done = True
            if self.player_now == 2:
                terminal = True
        elif self.weapon_pos[new_r][new_c] == -1:
            self.weapon_pos[new_r, new_c] = 0
            self.weapon_A = [-1,-1]
            r = 0.5
        elif self.weapon_pos[new_r][new_c] == 1:
            self.weapon_pos[new_r, new_c] = 0
            self.weapon_B = [-1, -1]
            r = 0.5
        else:
            # if next positon is empty
            # if stop, take turn, the reward to stop state is relate to  wall num
            if action == 8:
                adj = self.get_adj_eight(new_r, new_c)
                if len(adj) == 0:
                    r = -1.0
                else:
                    r = 1.0 / (len(adj) + 1)
                done = True
                invalid = False
                if self.player_now == 2:
                    terminal = True
            # if next postion is adjacent to enemy
            if self.is_adjacent(new_r, new_c, contra_row, contra_col):
                r = 1
                # update current player position
            if self.player_now == 1 and not invalid:
                self.playerA[0] = new_r
                self.playerA[1] = new_c
            elif self.player_now == 2 and not invalid:
                self.playerB[0] = new_r
                self.playerB[1] = new_c

        # modify current player
        if done:
            self.player_now = 3 - self.player_now

        # modify network input
        valid_action_list = self.get_valid_action(row, col, contra_row, contra_col)
        m = np.array([self.map, self.gas_pos, self.weapon_pos]).reshape(3, NUM * NUM)
        m_t = m.transpose()
        observation = m_t.reshape(NUM, NUM, 3)
        observation = np.hstack([observation.flatten(), [self.playerA[0]], [self.playerA[1]],
                                 [self.playerB[0]], [self.playerB[1]], valid_action_list])
        return observation, r, done, terminal, invalid

    def get_valid_action(self, row, col, con_row, con_col):
        actions = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if row + i in range(self.nrow) and col + j in range(self.ncol):
                    if self.map[row + i][col + j] > 0 or row + i == con_row and col + j == con_col:
                        actions.append(0)
                    else:
                        actions.append(1)
                else:
                    actions.append(0)
        return actions

    # judge whether the position is adjacent
    def is_adjacent(self, row_1, col_1, row_2, col_2):
        r_dist = row_1 - row_2
        c_dist = col_1 - col_2
        if abs(r_dist) + abs(c_dist) == 1:
            return True
        return False

    # give reward or penalty base on state and action
    @staticmethod
    def get_next_pos(row, col, action):
        new_row = row
        new_col = col

        if action == 0 or action == 4 or action == 5:  # up
            new_row -= 1
        if action == 1 or action == 6 or action == 7:  # down
            new_row += 1
        if action == 2 or action == 4 or action == 6:  # left
            new_col -= 1
        if action == 3 or action == 5 or action == 7:  # right
            new_col += 1
        return new_row, new_col

    def update(self, i):
        #set round
        self.round = i

        if self.round % 5 == 1:
            if self.player_now == 1:    #round 6,11,16,21,26 gas_pos only need update once
                self.set_danger_region()
                if self.playerA_weapon_create == False:
                    self.generate_weapon()
                    self.playerA_weapon_create = True
            else:
                if self.playerB_weapon_create == False:
                    self.generate_weapon()
                    self.playerB_weapon_create = True
        else:
            self.reset_weapon_pos()
            self.playerA_weapon_create = False
            self.playerB_weapon_create = False

    def set_danger_region(self):
        radius = 6 - self.round // 5
        self.gas_pos[:, :] = 1
        self.gas_pos[6 - radius: 6 + radius, 6 - radius: 6 + radius] = 0

    def generate_weapon(self):
        self.reset_weapon_pos()
        while True:
            r_pos = random.randint(0, self.nrow - 1)
            c_pos = random.randint(0, self.ncol - 1)
            if (r_pos != self.playerA[0] or c_pos != self.playerA[1])\
                and (r_pos != self.playerB[0] or c_pos != self.playerB[1])\
                and self.map[r_pos][c_pos] == 0:
                self.weapon_pos[r_pos][c_pos] = -1
                self.weapon_A = [r_pos, c_pos]
                break
        while True:
            r_pos = random.randint(0, self.nrow - 1)
            c_pos = random.randint(0, self.ncol - 1)
            if (r_pos != self.playerA[0] or c_pos != self.playerA[1])\
                and (r_pos != self.playerB[0] or c_pos != self.playerB[1])\
                and (self.weapon_A[0] != r_pos or self.weapon_A[1] != c_pos)\
                and self.map[r_pos][c_pos] == 0:
                self.weapon_pos[r_pos][c_pos] = 1
                self.weapon_B = [r_pos, c_pos]
                break

logger = TheLogger().getlogger("")

def test():
    games = 0
    while games < 10:

        nwalls = random.randint(15, 50) # random create walls
        board = Board(12, 12, nwalls)

        logger.info("                              ")
        logger.info("       Games{0} start!!!".format(games)       )
        logger.info("           map info           ")
        logger.info(format(board.map))
        logger.info("  A init_pos:{0}".format(board.playerA) + ", B init_pos:{0}".format(board.playerB))
        print(board.map)
        for i in range(1, 31):
            #打印非安全区
            # logger.info("           gas_pos info           ")
            # logger.info("           wea_pos info           ")
            while True:
                board.update(i)
                #print(i)
                #print(board.gas_pos)
                #logger.info(format(board.weapon_pos))
                curr_play = board.player_now   #获取当前回合进行移动的玩家

                # 打印该回合当前玩家所拥有的道具的位置，只在道具生成的回合打印，一回合打印两次
                # if i % 5 == 1:
                #     logger.info("         weapon_map          ")
                #     logger.info(format(board.weapon_pos))

                action_index = random.randrange(ACTIONS)
                observation, r, done, terminal, invalid = board.step(action_index)

                # 打印当前玩家在该回合所采取的动作、下一步的位置、以及回报
                logger.info("r{0}_{1}".format(i,curr_play)\
                            + "  Action:{0}".format(action_index) \
                            +"  A_pos:{0}".format(board.playerA)\
                            +"  B_pos:{0}".format(board.playerB)\
                            +"  R:{0}".format(r)\
                            +"  Done:{0}".format(done) \
                            + "  terminal:{0}".format(terminal))
                # logger.info(format(observation[-12:]))

                if terminal:
                    break

        games = games + 1

if __name__ == '__main__':
    test()
