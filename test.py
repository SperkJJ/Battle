from collections import deque
from RL_brain import PolicyGradient
from emulator import *


NUM = 12


def get_inputs(data):
    # env.map, env.gas_pos, env.weapon_pos, player_a, player_b, path_array
    obstacle_map = data['map']
    gas_map = data['gas_map']
    props_map = np.zeros(shape=(NUM, NUM))
    player_a = np.zeros(shape=(NUM, NUM))
    player_b = np.zeros(shape=(NUM, NUM))
    path = np.zeros(shape=(NUM, NUM))
    cur_player = data['cur_player']

    # set players and path
    for player, pos in data['players'].items():
        if player == cur_player:
            player_a[pos[0], pos[1]] = 1
            path[pos[0], pos[1]] = 1
        else:
            player_b[pos[0], pos[1]] = 1

    # set props_map
    for prop, pos in data["props"].items():
        props_map[pos[0], pos[1]] = 1

    mats = np.array([obstacle_map, gas_map, props_map, player_a, player_b, path]).transpose((1, 2, 0))
    return mats


def get_valid_action(map_, row, col, con_row, con_col):
    actions = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if 0 <= row + i < 12 and 0 <= col + j < 12:
                if map_[row + i][col + j] > 0 or (row + i == con_row and col + j == con_col):
                    actions.append(0)
                else:
                    actions.append(1)
            else:
                actions.append(0)
    return actions


def next(state, action, phase):
    """

    :param state: 12*12*6 :array
    :param action: 0-9
    :return: next_state:12*12*6, next_player_pos:1*8
    """
    # input 12*12*5
    # env.map, env.gas_pos, env.weapon_pos, player_a, player_b, path_array
    next_state = None
    map_ = state[:, :, 0]
    gas_pos = state[:, :, 1]
    weapon = state[:, :, 2]
    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
    path = state[:, :, -1]
    in_gas = False
    if gas_pos[player_pos[0]][player_pos[1]] > 0:
        in_gas = True
    r = new_r = int(player_pos[0])
    c = new_c = int(player_pos[1])
    if action == 0 or action == 4 or action == 5:  # up
        new_r -= 1
    if action == 1 or action == 6 or action == 7:  # down
        new_r += 1
    if action == 2 or action == 4 or action == 6:  # left
        new_c -= 1
    if action == 3 or action == 5 or action == 7:  # right
        new_c += 1
    is_valid = True
    has_weapon = False

    if new_r < 0 or new_c < 0 or new_r >= 12 or new_c >= 12:
        is_valid = False
    else:
        if map_[new_r, new_c] > 0:
            is_valid = False
        elif path[new_r, new_c] > 0:
            is_valid = False
        elif new_r == enemy_pos[0] and new_c == enemy_pos[1]:
            is_valid = False
        else:
            if weapon[new_r, new_c] > 0:
                has_weapon = True
                weapon[new_r, new_c] = 0
            path[new_r, new_c] = 1
            state[:, :, 3][r, c] = 0
            state[:, :, 3][new_r, new_c] = 1
    #valid_action = (get_valid_action(map_, new_r, new_c, enemy_pos[0], enemy_pos[1])
    #                if is_valid else [0]*8)
    # modify by tj
    valid_action = get_valid_action(map_, new_r, new_c, enemy_pos[0], enemy_pos[1])
    new_phase = trans_phase(phase, state)

    return is_valid, state, valid_action, new_phase,  has_weapon, in_gas


def is_adjacent(row_1, col_1, row_2, col_2):
    r_dist = row_1 - row_2
    c_dist = col_1 - col_2
    if abs(r_dist) + abs(c_dist) == 1:
        return True
    return False


def get_adj_four(map_, r_pos, c_pos):
    adj = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i) + abs(j) != 1:
                continue
            new_r = r_pos + i
            new_c = c_pos + i
            if new_r < 0 or new_c < 0 or new_r >= map_.shape[0] or new_c >= map_.shape[1]:
                continue
            if map_[new_r][new_c] == 0:
                adj.append((new_r, new_c))
    return adj


def is_attack(map_, cur, enemy):
    adj = get_adj_four(map_, enemy[0], enemy[1])
    for k in adj:
        if k[0] == cur[0] and k[1] == cur[1]:
            return True
    return False


#def get_reward(state, valid, action, has_weapon, in_gas):
#    map_ = state[:, :, 0]
#    gas_pos = state[:, :, 1]
#    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
#    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
#    path = state[:, :, -1]
#    r = -0.01
#
#    if not valid:
#        return -1.0
#    elif not in_gas and gas_pos[player_pos[0]][player_pos[1]] > 0:
#        return -1.0
#    elif action == 8 and len(path[path > 0]) == 1:
#        return -1.0
#
#    if is_adjacent(player_pos[0], player_pos[1], enemy_pos[0], enemy_pos[1]):
#        r += 0.3
#    if has_weapon:
#        r += 0.15
#    if action == 8:
#        adj = get_adj_four(map_, player_pos[0], player_pos[1])
#        r += (4 - len(adj)) * 0.2
#    return r


def get_adj_eight(pos, r_limit=12, c_limit=12):
    r = pos[0]
    c = pos[1]
    adj = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            #if i == 0 and j == 0:
            #    continue
            new_r = r + i
            new_c = c + j
            if new_r < 0 or new_r >= r_limit or new_c < 0 or new_c >= c_limit:
                continue
            adj[(new_r, new_c)] = 1
    return adj


def bfs(map_, gas_, path_, src, target):
    # get src adj eight position
    adj_eight = get_adj_eight(src)
    adj_depth = {}
    for k in adj_eight:
        #if map_[k[0]][k[1]] > 0 or gas_[k[0]][k[1]] > 0 or path_[k[0]][k[1]] > 0:
        if map_[k[0]][k[1]] > 0 or gas_[k[0]][k[1]] > 0 or (k != (src[0], src[1]) and path_[k[0]][k[1]] > 0):
            adj_depth[k] = -1

    dq = deque()
    dq.append([target, 0])
    visited = {}
    while len(dq):
        cur = dq.popleft()
        if (cur[0][0], cur[0][1]) in visited:
            continue
        visited[(cur[0][0], cur[0][1])] = 1
        if (cur[0][0], cur[0][1]) not in adj_depth and (cur[0][0], cur[0][1]) in adj_eight:
            adj_depth[(cur[0][0], cur[0][1])] = cur[1]
        #if (cur[0] == src).all():
        #    src_depth[(src[0], src[1])] = cur[1]
        #    break
        if len(adj_depth) == len(adj_eight):
            break
        cur_adj_eight = get_adj_eight(cur[0])

        for k in cur_adj_eight:
            if k == (cur[0][0], cur[0][1]):
                continue
            if map_[k[0]][k[1]] > 0 or gas_[k[0]][k[1]] > 0 or k != (src[0], src[1]) and path_[k[0]][k[1]] > 0:
                continue
            if (k[0], k[1]) in visited:
                continue
            if cur[1] > 0:
                dq.append([k, cur[1] + 1])
            else:
                if is_adjacent(k[0], k[1], target[0], target[1]):
                    dq.append([k, cur[1] + 1])
                else:
                    dq.append([k, cur[1] + 2])
    return adj_depth






#def get_reward(phase, state, valid, action, has_weapon, in_gas):
#    map_ = state[:, :, 0]
#    gas_pos = state[:, :, 1]
#    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
#    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
#    path = state[:, :, -1]
#    r = -0.01
#
#    if not valid:
#        return -1.0
#    elif not in_gas and gas_pos[player_pos[0]][player_pos[1]] > 0:
#        return -1.0
#    elif action == 8 and len(path[path > 0]) == 1:
#        return -1.0

#    if action == 8:
#        adj = get_adj_four(map_, player_pos[0], player_pos[1])
#        r += (4 - len(adj)) * 0.2
#
#    elif phase == 'W':
#        adj_r = get_reward_weapon(map_, gas_pos, path, player_pos, enemy_pos)
#        r = adj_r[(player_pos[0], player_pos[1])]
#    elif phase == 'A':
#        adj_r = get_reward_attack(map_, gas_pos, path, player_pos, enemy_pos)
#        r = adj_r[(player_pos[0], player_pos[1])]
#    return r


def get_reward(state, adj_r):
    map_ = state[:, :, 0]
    gas_ = state[:, :, 1]
    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
    path_ = state[:, :, -1]
    return adj_r[(player_pos[0], player_pos[1])]


# get reward of getting weapons
def get_reward_weapon(state):
    map_ = state[:, :, 0]
    gas_ = state[:, :, 1]
    # src is player pos
    src = np.argwhere(state[:, :, 3] == 1)[0]
    # target is weapon pos
    target = np.argwhere(state[:, :, 4] == 1)
    path_ = state[:, :, -1]
    adj_dist = {}
    # for multi-target, visit each target
    for i in range(target.shape[0]):
        adj_tmp = bfs(map_, gas_, path_, src, target[i])
        for k, v in adj_tmp.items():
            if k not in adj_dist:
                adj_dist[k] = v
            else:
                adj_dist[k] = min([adj_dist[k], v])
    adj_reward = {}
    for k, v in adj_dist.items():
        if v > 1:
            adj_reward[k] = 1.0/float(v) - 1.0/(adj_dist[(src[0], src[1])])
        elif v == 0:
            adj_reward[k] = 1.0
        else:
            adj_reward[k] = v
    return adj_reward


# get attack reward
def get_reward_attack(state):
    map_ = state[:, :, 0]
    gas_ = state[:, :, 1]
    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
    path_ = state[:, :, -1]
    adj_dist = bfs(map_, gas_, path_, player_pos, enemy_pos)
    adj_reward = {}
    for k, v in adj_dist.items():
        if k == (player_pos[0], player_pos[1]):
            adj_reward[k] = -1
        elif v > 1:
            adj_reward[k] = 1.0/float(v) - 1.0/adj_dist[(player_pos[0],player_pos[1])]
        elif v == 1:
            adj_reward[k] = 1
        else:
            adj_reward[k] = v
    return adj_reward


# get rewards of stop place
def get_reward_escape(state):
    map_ = state[:, :, 0]
    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
    adj = get_adj_four(map_, player_pos[0], player_pos[1])
    r = (4 - len(adj)) * 0.2
    r = r if r > 0 else -0.8
    return r


# init phase
def init_phase(state):
    phase = ''
    gas_ = state[:, :, 1]
    weapon_ = state[:, :, 2]
    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]

    if (weapon_[weapon_ > 0] == gas_[weapon_ > 0]).all():
        phase = 'A'
    if gas_[enemy_pos[0]][enemy_pos[1]] > 0:
        phase = 'E'
    else:
        if phase == '':
            phase = 'W'
        else:
            phase = 'A'
    return phase


# phase transition
def trans_phase(phase, state):
    weapon_ = state[:, :, 2]
    if phase == 'W' and np.sum(weapon_ > 0) == 0:
        phase = 'A'
    elif phase == 'A':
        # if no good choice, end attack
        attack_again = False
        adj_rewards = get_reward_attack(state)
        for k, v in adj_rewards.items():
            if v > 0:
                attack_again = True
                break
        if not attack_again:
            phase = 'E'
    return phase


# represent phase as array
def get_phase_array(phase):
    phase_arr = np.zeros(shape=(3,))
    if phase == 'W':
        phase_arr[0] = 1
    elif phase == 'A':
        phase_arr[1] = 1
    elif phase == 'E':
        phase_arr[2] = 1
    return phase_arr


def run_game():
    step = 0
    for _ in range(1000000):
        eml = Emulator(p=0.1)  # 重置模拟器
        while 1:
            s = eml.get_state()
            if s['is_done']:  # 如果进入终盘则跳出
                break
            state = get_inputs(s)  # 将模拟器返回状态转换为12*12*6 tensor
            map_ = state[:, :, 0]
            player_pos = np.argwhere(state[:, :, 3] == 1)[0]
            enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
            actions = np.array(get_valid_action(map_,
                                                int(player_pos[0]), int(player_pos[1]),
                                                enemy_pos[0], enemy_pos[1]))
            phase = init_phase(state)
            phase_arr = get_phase_array(phase)
            observation = np.hstack((state.flatten(), actions, phase_arr))  # 展平并拼接

            moves = [player_pos]  # 路径列表
            has_attack = False
            while True:
                # RL choose action based on observation
                action = RL.choose_action(observation=observation)
                # RL take action and get next observation and reward
                # get rewards  of adjacent positions based on phase
                adj_r = {}
                if phase == 'W':
                    adj_r = get_reward_weapon(state)
                elif phase == 'A':
                    adj_r = get_reward_attack(state)

                valid, state, actions, phase, has_weapon, in_gas = next(state, action, phase)  # 状态转移
                # get phase array
                phase_arr = get_phase_array(phase)
                # get input
                observation_ = np.hstack((state.flatten(), actions, phase_arr))
                done = 0 if action != 8 and valid else 1  # 本轮是否终止

                if phase == 'W' or phase == 'A':
                    reward = get_reward(state, adj_r)
                elif action == 8 and phase == 'E':
                    reward = get_reward_escape(state)
                elif action == 8:
                    reward = -1.0
                else:
                    reward = 0.0
                RL.store_transition(observation_, action, reward)
                player_pos = np.argwhere(state[:, :, 3] == 1)[0]
                moves.append(player_pos)
                if done:
                    RL.learn()
                    step += 1
                    break
                # swap observation
                observation = observation_
            eml.next(list(map(lambda x: (x[0], x[1]), moves)))


if __name__ == "__main__":
    RL = PolicyGradient(n_actions=9, n_features=144 * 6 + 8 + 3, learning_rate=1e-7)
    run_game()
