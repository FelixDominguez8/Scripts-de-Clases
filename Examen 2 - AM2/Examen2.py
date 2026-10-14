import random
import numpy as np
import matplotlib.pyplot as plt

class SimpleMaze:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.height, self.width = grid.shape
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.height - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.width - 1)

        next_state = (x, y)

        if self.grid[x, y] == 1:  # obstacle
            reward = -10
            done = True
        elif next_state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done

def greedy(n_episodes=500, alpha=0.1, gamma=0.99,
           epsilon=0.1, min_epsilon=0.01, decay=0.995, n_actions=4):
    global env
    Q = {}
    rewards = []
    lengths = []
    successes = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total = 0
        steps = 0
        success = 0
        while not done:
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            if random.random() < epsilon:
                action = random.randrange(n_actions)
            else:
                action = int(np.argmax(Q[state]))
            next_state, reward, done = env.step(action)
            if next_state not in Q:
                Q[next_state] = np.zeros(n_actions)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
            total += reward
            steps += 1
            if done and reward > 0:
                success = 1
        epsilon = max(min_epsilon, epsilon * decay)
        rewards.append(total)
        lengths.append(steps)
        successes.append(success)
    return Q, rewards, lengths, successes

def sarsa(n_episodes=500, alpha=0.1, gamma=0.99,
          epsilon=0.1, min_epsilon=0.01, decay=0.995, n_actions=4):
    global env
    Q = {}
    rewards = []
    lengths = []
    successes = []
    for _ in range(n_episodes):
        state = env.reset()
        if state not in Q: Q[state] = np.zeros(n_actions)
        if random.random() < epsilon:
            action = random.randrange(n_actions)
        else:
            action = int(np.argmax(Q[state]))
        done = False
        total = 0
        steps = 0
        success = 0
        while not done:
            next_state, reward, done = env.step(action)
            if next_state not in Q: Q[next_state] = np.zeros(n_actions)
            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
                total += reward
                steps += 1
                if reward > 0:
                    success = 1
                break
            if random.random() < epsilon:
                next_action = random.randrange(n_actions)
            else:
                next_action = int(np.argmax(Q[next_state]))
            td_target = reward + gamma * Q[next_state][next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])
            state, action = next_state, next_action
            total += reward
            steps += 1
        epsilon = max(min_epsilon, epsilon * decay)
        rewards.append(total)
        lengths.append(steps)
        successes.append(success)
    return Q, rewards, lengths, successes


if __name__ == "__main__":
    # Create a new simple maze environment
    grid = np.zeros((5, 5))
    grid[1, 2] = 1  # obstacle
    grid[2, 2] = 1  # obstacle
    grid[3, 2] = 1  # obstacle

    start = (0, 0)
    goal = (4, 4)
    env = SimpleMaze(grid, start, goal)

    Q_g, rewards_g, lengths_g, succ_g = greedy()
    Q_s, rewards_s, lengths_s, succ_s = sarsa()

    print("Q-table de SARSA:")
    for state in sorted(Q_s):
        print(state, Q_s[state])
    
    episodes = range(1, len(rewards_g) + 1)
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), tight_layout=True)
    
    axs[0].plot(episodes, rewards_g, label='Epsilon-Greedy')
    axs[0].plot(episodes, rewards_s, label='SARSA')
    axs[0].set_title('Recompensa total por episodio')
    axs[0].set_ylabel('Recompensa')
    axs[0].legend()
    
    axs[1].plot(episodes, lengths_g, label='Episilon-Greedy')
    axs[1].plot(episodes, lengths_s, label='SARSA')
    axs[1].set_title('Longitud por episodio')
    axs[1].set_ylabel('Pasos')
    axs[1].legend()
    
    axs[2].plot(episodes, succ_g, label='Epsilon-Greedy')
    axs[2].plot(episodes, succ_s, label='SARSA')
    axs[2].set_title('Exito por episodio')
    axs[2].set_ylabel('Exito')
    axs[2].set_xlabel('Episodio')
    axs[2].legend()
    
    plt.show()