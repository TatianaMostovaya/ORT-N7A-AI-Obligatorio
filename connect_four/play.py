from stratego_agent import Stratego

def play_vs_other_agent(env, agent1, agent2):
    done = False
    obs, _ = env.reset()
    winner = 0
    while not done:
        env.render()
        action = agent1.next_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()
        if not done:
            next_action = agent2.next_action(obs)
            _, _, terminated, truncated, _ = env.step(next_action)
            done = terminated or truncated

    env.render()
    winner = env._grid.winner
    final_msg = "Player " + str(winner) + " WON!!!" if winner != 0 else "It's a tie!" 
    print(final_msg)

def play_vs_loaded_agent(env, agent):
    enemy_agent = load_enemy_agent()
    play_vs_other_agent(env, agent, enemy_agent)

def load_enemy_agent():
    return Stratego(2, 4)
