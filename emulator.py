import numpy as np
from cnn_board_new import *

def is_valid(map_, path, new_player_a_pos, player_b_pos):
    # TODO: 记得测试一下这函数
    """
    判断当前状态下的操作是否合法
    :param map_: ndarray (12, 12)
    :param path: ndarray (12, 12)
    :param new_player_a_pos: ndarray (2,)
    :param player_b_pos: ndarray (2,)
    :return: valid: bool
    """
    # 出界
    if not (0 <= new_player_a_pos[0] < 12 and 0 <= new_player_a_pos[1] < 12):
        return False
    # 撞玩家
    if (new_player_a_pos == player_b_pos).all():
        return False

    pos = new_player_a_pos.astype(np.int8)
    # 撞墙
    if map_[pos[0], pos[1]]:
        return False
    # 重复路径
    if path[pos[0], pos[1]] and path.sum() > 1:
        return False

    return True


def get_next_pos(r, c, action):
    if action == 1 or action == 5 or action == 6:  # up
        r -= 1
    if action == 2 or action == 7 or action == 8:  # down
        r += 1
    if action == 3 or action == 5 or action == 7:  # left
        c -= 1
    if action == 4 or action == 6 or action == 8:  # right
        c += 1
    return [r, c]


def get_adj(r_pos, c_pos, wall_pos):
    adj = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i) + abs(j) > 1 or abs(i) + abs(j) == 0:
                continue
            new_r = r_pos + i
            new_c = c_pos + j
            if new_r < 0 or new_c < 0 or new_r >= 12 or new_c >= 12:
                continue
            if wall_pos[new_r][new_c] == 0:
                adj.append((new_r, new_c))
    return adj


def reward(state_tensor, state_array, action):
    """
    返回奖励reward以及下一次的状态new_state_tensor, new_state_array, 还有是否合法is_valid与是否终止is_done
    :param state_tensor: ndarray (12, 12, 3) [map, gas, history_path]
    :param state_array: ndarray (12,) player_a_pos[2], player_b_pos[2], prop_a_pos[2], prop_b_pos[2], start_pos[2],
                        has_prop_a[1], has_prop_b[1]
    :param action: int
    :return: reward: int, new_state_tensor: ndarray (12, 12, 3), new_state_array: ndarray (12,), is_valid: bool,
             is_done: bool
    """
    # 注意：出现异常之后，该回合英雄并不会移动，所以需要储存start_pos.
    is_done = action == 0
    r = 0
    #print(state_tensor[:, :, 0])
    #print(state_array, action)
    new_player_a_pos = np.array(get_next_pos(state_array[0], state_array[1], action))
    #print(new_player_a_pos)
    if is_valid(state_tensor[:, :, 0], state_tensor[:, :, -1], new_player_a_pos, state_array[2:4]):
        # 原地不动扣十分
        # 如果停止，每相邻一块空地扣5分
        # 如果停止，计算本回合总攻击分数，回合攻击分数为文档中计算方法
        # 吃到道具加2分
        # 根据道具情况计算每步攻击分数（有道具加7分，没道具加5分）
        # into gas -10
        # TODO: 查漏补缺，我就想到这么多

        # gas

        if np.sum(state_tensor[:, :, 1] * state_tensor[:, :, -1]) == 0 and state_tensor[:, :, 1][new_player_a_pos.astype(int)[0],
                                                                                                 new_player_a_pos.astype(int)[1]] > 0:
            r -= 10
        # attack
        attack_pos = get_adj(state_array[2], state_array[3], state_tensor[:, :, 0])
        if is_done:
            attack_times = 0
            for pos in attack_pos:
                if state_tensor[pos[0], pos[1], -1] > 0:
                    attack_times += 1
            if state_array[-2] > 0:
                r += 2 * attack_times
            else:
                r += attack_times
            if attack_times == 3:
                r += 3*5
            elif attack_times == 4:
                r += 5*5
            r -= 3 * len(get_adj(new_player_a_pos[0], new_player_a_pos[1], state_tensor[:, :, 0]))
            # update here or out of this function?
            if state_tensor[:, :, -1].sum() == 1:
                r -= 10
        if (new_player_a_pos[0], new_player_a_pos[1]) in attack_pos:
            r += 5
            if state_array[-1] > 0:
                r += 2
        state_array[0] = new_player_a_pos[0]
        state_array[1] = new_player_a_pos[1]
        #print(new_player_a_pos, state_array)
        state_tensor[new_player_a_pos[0], new_player_a_pos[1], -1] = 1
        # update weapon
        if state_array[4] == new_player_a_pos[0] and state_array[5] == new_player_a_pos[1]:
            state_array[-2] = 1
            state_array[4] = -10
            state_array[5] = -1
        elif state_array[6] == new_player_a_pos[0] and state_array[7] == new_player_a_pos[1]:
            state_array[-1] = 1
            state_array[6] = -10
            state_array[7] = -1
        #print(r)
        #print(state_array)
        return r, state_tensor, state_array, True, is_done

    # 无效操作直接停止，因为训练时不会计算终止状态下一步， 所以直接返回输入占位即可
    count = len(get_adj(state_array[0], state_array[1], state_tensor[:, :, 0])) * (-5)
    #print(r, count)
    return -10 + count, state_tensor, state_array, False, True


if __name__ == '__main__':
    env = Board(12, 12, 10)
    has_weapon_a = 0
    has_weapon_b = 0
    if env.weapon_A[0] >= 0 and env.weapon_A[1] >= 0:
        has_weapon_a = 1
    if env.weapon_B[0] >= 0 and env.weapon_B[1] >= 0:
        has_weapon_b = 1
    m = np.array([env.map, env.gas_pos, env.weapon_pos]).reshape(3, 12 * 12)
    m_t = m.transpose()
    m_tensor = m_t.reshape(12, 12, 3)
    observation = np.hstack(([env.playerA[0]], [env.playerA[1]], [env.playerB[0]], [env.playerB[1]], env.weapon_A, env.weapon_B, [has_weapon_a, has_weapon_b]))
    r, s_t, s_a, _, _ = reward(m_tensor, observation, random.randint(0, 8))

    #print(s_a)
    #print(r)
