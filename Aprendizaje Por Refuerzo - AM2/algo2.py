import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('q-learning_entrenamiento2.pkl','rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discount_factor_g = 0.9

    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    recompensas_totales = 0
    steps_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        num_steps = 0

        while(not terminated and not truncated):

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state
            num_steps += 1

        steps_per_episode[i] = num_steps
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)

        if(epsilon==0.01):
            learning_rate_a = 0.0001

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

        f = open("q-learning_entrenamiento2.pkl","wb")
        pickle.dump(q, f)
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
    run(15000)
    run(100, is_training=False)
    run(1, is_training=False, render=True)