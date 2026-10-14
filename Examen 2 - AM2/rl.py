import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- CONFIGURACIÓN GLOBAL ---
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1
EPISODES = 500

# 1. IMPLEMENTACIÓN DE Q-LEARNING Y SARSA (TABULARES)
def entrenar_tabulares():
    env = gym.make('CliffWalking-v1')
    states, actions = env.observation_space.n, env.action_space.n
    
    q_table_ql = np.zeros([states, actions])
    q_table_sarsa = np.zeros([states, actions])
    
    rewards_ql = []
    rewards_sarsa = []

    for i in range(EPISODES):
        # --- Q-LEARNING (OFF-POLICY) ---
        s_ql, _ = env.reset()
        done_ql, total_ql = False, 0
        while not done_ql:
            a = np.argmax(q_table_ql[s_ql]) if random.random() > EPSILON else env.action_space.sample()
            ns, r, term, trunc, _ = env.step(a)
            # Update: usa el MAX del siguiente estado
            q_table_ql[s_ql, a] += ALPHA * (r + GAMMA * np.max(q_table_ql[ns]) - q_table_ql[s_ql, a])
            s_ql, total_ql, done_ql = ns, total_ql + r, term or trunc
        rewards_ql.append(total_ql)

        # --- SARSA (ON-POLICY) ---
        s_sa, _ = env.reset()
        a_sa = np.argmax(q_table_sarsa[s_sa]) if random.random() > EPSILON else env.action_space.sample()
        done_sa, total_sa = False, 0
        while not done_sa:
            ns, r, term, trunc, _ = env.step(a_sa)
            na = np.argmax(q_table_sarsa[ns]) if random.random() > EPSILON else env.action_space.sample()
            # Update: usa la acción REAL que se tomará (na)
            q_table_sarsa[s_sa, a_sa] += ALPHA * (r + GAMMA * q_table_sarsa[ns, na] - q_table_sarsa[s_sa, a_sa])
            s_sa, a_sa, total_sa, done_sa = ns, na, total_sa + r, term or trunc
        rewards_sarsa.append(total_sa)

    return q_table_ql, q_table_sarsa, rewards_ql, rewards_sarsa

# 2. DEEP Q-LEARNING (DQN) - ESTRUCTURA Y LÓGICA
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): return self.fc(x)

def mostrar_agente(env_id, table, name):
    print(f"\nMostrando resultado de: {name}")
    env = gym.make(env_id, render_mode='human')
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(table[state])
        state, r, term, trunc, _ = env.step(action)
        env.render()
        done = term or trunc
    env.close()

# --- EJECUCIÓN Y COMPARATIVA ---
if __name__ == "__main__":
    # Entrenar algoritmos tabulares
    q_ql, q_sa, hist_ql, hist_sa = entrenar_tabulares()

    # Graficar comparativa (Q-Learning vs SARSA)
    plt.figure(figsize=(12, 6))
    plt.plot(hist_ql, label='Q-Learning (Off-Policy)', alpha=0.7)
    plt.plot(hist_sa, label='SARSA (On-Policy)', alpha=0.7)
    plt.title('Comparativa de Aprendizaje: Q-Learning vs SARSA')
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Acumulada')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Explicación de la diferencia en DQN
    print("\n--- RESUMEN TÉCNICO ---")
    print("1. Q-LEARNING: Es 'audaz'. En CliffWalking verás que camina por la pura orilla.")
    print("2. SARSA: Es 'prudente'. Camina lejos del borde porque sabe que puede caerse al explorar.")
    print("3. DQN: En lugar de usar la tabla 'q_table[state]', usaríamos 'model(state)'.")
    
    # Mostrar uno de los agentes entrenados
    mostrar_agente('CliffWalking-v1', q_sa, "SARSA (Agente Prudente)")