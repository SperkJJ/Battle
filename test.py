from cnn_board_new import *
from RL_brain_cnn_new import DeepQNetwork
from emulator import *

def run_game(env):
    for i in range(100):
        n_walls = int(random.random() * 30 + 1)
        env.reset_wall(12, 12, n_walls)
        env.reset_players()
        step = 0
        path = np.zeros(shape=(env.nrow, env.ncol))
        m = np.array([env.map, env.gas_pos, path]).reshape(3, 12 * 12)
        m_t = m.transpose().reshape(12, 12, 3)

        observation = np.hstack((m_t.flatten(), [env.playerA[0]], [env.playerA[1]], [env.playerB[0]],
                                 [env.playerB[1]], env.weapon_A, env.weapon_B, env.playerA, [0, 0]))
        for _ in range(1000):
            env.reset_players()
            env.reset_gas_pos()
            start = True
            for episode in range(30):
                #observation = env.reset(12, 12, n_walls)
                start = True
                while True:
                    # RL choose action based on observation
                    action = RL.choose_action(observation=observation)
                    # RL take action and get next observation and reward
                    state_tensor, state_array, reward, done, terminal = env.step(action, start)
                    observation_ = np.hstack([state_tensor.flatten(), state_array.flatten()])
                    # print(observation_.shape)
                    RL.store_transition(observation, action, reward, done, observation_)


                    if step > 200 and (step % 200 == 0):
                        RL.learn()

                    # swap observation
                    observation = observation_

                    #break while loop when end of this episode
                    if done:
                        start = True
                        break
                    step += 1






if __name__ == "__main__":
    env = Board(12, 12, 10)
    RL = DeepQNetwork(env.n_actions, 444,
                      learning_rate=1e-4,
                      reward_decay=0.95,
                      e_greedy=0.98,
                      replace_target_iter=100,
                      memory_size=2048,
                      batch_size=64
                      )

    run_game(env)
    RL.plot_cost()
