import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    modelo_aleatorio = None

    if(is_training):
        modelo_aleatorio = env.action_space
    else:
        f = open('aleatorio_entrenamiento.pkl', 'rb')
        modelo_aleatorio = pickle.load(f)
        f.close()

    rewards_per_episode = np.zeros(episodes)
    recompensas_totales = 0
    steps_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        num_steps = 0

        while(not terminated and not truncated):
            action = modelo_aleatorio.sample()
            new_state,reward,terminated,truncated,_ = env.step(action)
            state = new_state
            num_steps += 1

        steps_per_episode[i] = num_steps

        if reward == 1:
            rewards_per_episode[i] = 1
            recompensas_totales += 1

    env.close()

    if(is_training):
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.show()

        f = open("aleatorio_entrenamiento.pkl","wb")
        pickle.dump(modelo_aleatorio, f)
        f.close()
    else:
        print("Proporcion de exito: \n")
        print(recompensas_totales/episodes)

        print("\nRecompensa media: \n")
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.mean(rewards_per_episode[max(0, t-10):(t+1)])
        plt.plot(sum_rewards)
        plt.title("Recompensa media por cada 10 episodios")
        plt.show()

        print("\nLongitud promedio de episodio: \n")
        print(sum(steps_per_episode) / len(steps_per_episode))
        

if __name__ == '__main__':
    run(2000)
    run(100, is_training=False)