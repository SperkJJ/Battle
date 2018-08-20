from cnn_board_new import *
from RL_brain_cnn_new import DeepQNetwork

NUM=12

def run_game(env):
    for i in range(600000):
        n_walls = int(random.random() * 30 + 1)
        env.reset_wall(NUM, NUM, n_walls)
        env.reset_players()
        step = 0
        for _ in range(1000000):
            env.reset_players()
            env.reset_gas_pos()
            m = np.array([env.map, env.gas_pos, env.weapon_pos]).reshape(3, NUM * NUM)
            m_t = m.transpose()
            observation = m_t.reshape(NUM, NUM, 3)
            row = env.playerA[0]
            col = env.playerA[1]
            contra_row = env.playerB[0]
            contra_col = env.playerB[1]

            adj = env.get_valid_action(row, col, contra_row, contra_col)
            observation = np.hstack(
                [observation.flatten(), [env.playerA[0]], [env.playerA[1]], [env.playerB[0]], [env.playerB[1]], adj])

            for episode in range(30):
                while True:
                    env.update(i)
                    # RL choose action based on observation
                    action = RL.choose_action(observation=observation)
                    # RL take action and get next observation and reward
                    observation_, reward, done, terminal, invalid = env.step(action)
                    RL.store_transition(observation, action, reward, done, observation_)

                    if step > 100 and (step % 300 == 0):
                        RL.learn()

                    # swap observation
                    observation = observation_

                    #break while loop when end of this episode
                    if terminal:
                        break
                    step += 1



if __name__ == "__main__":
    env = Board(12, 12, 10)
    RL = DeepQNetwork(env.n_actions, n_features=444,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy_increment=1e-8,
                      replace_target_iter=100,
                      memory_size=256,
                      batch_size=64
                      )
    run_game(env)
    RL.plot_cost()


