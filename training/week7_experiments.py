# ============================================================
# week7_experiments.py  —  VEEKSHA'S FILE  (training/ folder)
# Week 7: Seed experiments + Scalability test
#
# Run this file AFTER trainer.py is working.
# It will automatically:
#   1. Run IQL, QMIX, MAPPO with 3 different seeds each
#   2. Print a results table with mean ± std
#   3. Run scalability test (5 vs 10 agents)
#   4. Save all graphs to results/
# ============================================================

import numpy as np
import torch
import csv
import os
import sys
import matplotlib.pyplot as plt

# ── Path setup (same as trainer.py) ───────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'agents'))
sys.path.append(os.path.join(ROOT_DIR, 'environment'))

from qmix_smart_grid  import QMIXAgent
from iql_smart_grid   import IQLAgent, IQLSmartGridSystem
from mappo_smart_grid import MAPPOSystem

# ── Use real env or dummy ──────────────────────────────────────
USE_REAL_ENV = True   # ← already working, keep True


# ============================================================
# COPY of DummyEnv and Wrapper from trainer.py
# (needed here so this file runs standalone)
# ============================================================
class DummySmartGridEnv:
    def __init__(self, num_agents=5, max_steps=96):
        self.num_agents   = num_agents
        self.max_steps    = max_steps
        self.current_step = 0
        self.battery_soc  = 0.5

    def _get_obs(self):
        t      = self.current_step / self.max_steps
        solar  = float(max(0, np.sin(np.pi * t)))
        wind   = float(np.random.uniform(0.1, 0.8))
        price  = float(np.random.uniform(0.2, 0.9))
        demand = float(np.random.uniform(0.3, 1.0))
        return {'battery_soc': round(self.battery_soc,4),
                'solar_output': round(solar,4),
                'wind_output': round(wind,4),
                'electricity_price': round(price,4),
                'demand': round(demand,4),
                'time_step': self.current_step}

    def _to_array(self, obs):
        return np.array([obs['battery_soc'], obs['solar_output'],
                         obs['wind_output'], obs['electricity_price'],
                         obs['demand'], obs['time_step']/95.0], dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.battery_soc  = 0.5
        obs = self._get_obs()
        states = [self._to_array(obs) for _ in range(self.num_agents)]
        return states, np.concatenate(states)

    def step(self, actions):
        self.current_step += 1
        if actions[2] == 1: self.battery_soc = min(0.95, self.battery_soc+0.05)
        elif actions[2] == 2: self.battery_soc = max(0.10, self.battery_soc-0.05)
        obs     = self._get_obs()
        states  = [self._to_array(obs) for _ in range(self.num_agents)]
        reward  = obs['solar_output'] + obs['wind_output']*0.5
        if obs['electricity_price'] > 0.7: reward -= 2.0
        if self.battery_soc < 0.1: reward -= 3.0
        rewards = [reward]*self.num_agents
        done    = self.current_step >= self.max_steps
        return states, rewards, done, np.concatenate(states)


class SmartGridEnvWrapper:
    def __init__(self, env, num_agents=5):
        self.env        = env
        self.num_agents = num_agents

    def _to_array(self, obs_dict):
        return np.array([obs_dict['battery_soc'], obs_dict['solar_output'],
                         obs_dict['wind_output'], obs_dict['electricity_price'],
                         obs_dict['demand'], obs_dict['time_step']/95.0],
                        dtype=np.float32)

    def reset(self):
        obs_dict, _ = self.env.reset()
        arr    = self._to_array(obs_dict)
        states = [arr.copy() for _ in range(self.num_agents)]
        return states, np.concatenate(states)

    def step(self, actions):
        names = ['solar_agent','wind_agent','battery_agent','grid_agent','load_agent']
        action_dict = {n: int(actions[i]) for i,n in enumerate(names)}
        obs_dict, reward, terminated, truncated, _ = self.env.step(action_dict)
        arr     = self._to_array(obs_dict)
        states  = [arr.copy() for _ in range(self.num_agents)]
        done    = terminated or truncated
        rewards = [float(reward)] * self.num_agents
        return states, rewards, done, np.concatenate(states)


def make_env(num_agents=5):
    """Create whichever environment is active."""
    if USE_REAL_ENV:
        from smart_grid_env import SmartGridEnv
        return SmartGridEnvWrapper(SmartGridEnv(), num_agents)
    return DummySmartGridEnv(num_agents)


# ============================================================
# CORE TRAINING FUNCTION
# Runs one algorithm for num_episodes with a given seed.
# Returns list of episode rewards.
# ============================================================
def run_one_trial(algorithm, seed, num_episodes=200,
                  num_agents=5, state_dim=6, action_dim=4,
                  global_state_dim=None):
    """
    Train one algorithm with one random seed.
    Returns: list of total rewards per episode.
    """
    if global_state_dim is None:
        global_state_dim = num_agents * state_dim

    # Set all random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(num_agents)

    # Build agent
    if algorithm == 'qmix':
        agent = QMIXAgent(num_agents=num_agents, state_dim=state_dim,
                          action_dim=action_dim, global_state_dim=global_state_dim)
    elif algorithm == 'iql':
        agent = IQLSmartGridSystem(num_agents=num_agents, state_dim=state_dim,
                                   action_dim=action_dim)
    elif algorithm == 'mappo':
        agent = MAPPOSystem(num_agents=num_agents, state_dim=state_dim,
                            action_dim=action_dim, global_state_dim=global_state_dim)

    episode_rewards = []
    epsilon = 1.0

    for episode in range(num_episodes):
        states, global_state = env.reset()
        episode_reward  = 0.0
        global_states_traj = []

        for step in range(96):
            # Select actions
            if algorithm == 'qmix':
                actions = agent.select_action(states, epsilon)
            elif algorithm == 'iql':
                actions = [a.act(states[i]) for i,a in enumerate(agent.agents)]
            elif algorithm == 'mappo':
                actions = agent.select_actions(states)

            next_states, rewards, done, next_global_state = env.step(actions)

            # Store and train
            if algorithm == 'qmix':
                agent.store_experience(states, actions, float(np.mean(rewards)),
                                       next_states, global_state,
                                       next_global_state, done)
                agent.train_step()
            elif algorithm == 'iql':
                for i,a in enumerate(agent.agents):
                    a.remember(states[i], actions[i], rewards[i], next_states[i], done)
                    a.replay()
            elif algorithm == 'mappo':
                global_states_traj.append(global_state)
                agent.store_transitions(states, actions, rewards, done)

            episode_reward += sum(rewards)
            states       = next_states
            global_state = next_global_state
            if done: break

        if algorithm == 'mappo' and len(global_states_traj) > 0:
            agent.update(global_states_traj)

        epsilon = max(0.05, epsilon * 0.995)

        if (episode+1) % 10 == 0:
            print(f"    [{algorithm.upper()} seed={seed}] "
                  f"Ep {episode+1:3d}/{num_episodes} | "
                  f"Reward: {episode_reward:.1f} | ε: {epsilon:.3f}")

        episode_rewards.append(episode_reward)

    return episode_rewards


# ============================================================
# EXPERIMENT 1 — 3 SEEDS PER ALGORITHM
# ============================================================
def run_seed_experiments(seeds=[42, 123, 999], num_episodes=200):
    """
    Runs IQL, QMIX, MAPPO each with 3 random seeds.
    Saves results table and comparison graph.
    """
    os.makedirs('results', exist_ok=True)
    algorithms = ['iql', 'qmix', 'mappo']
    all_results = {}   # {algorithm: {seed: [rewards]}}

    print("\n" + "="*60)
    print("  EXPERIMENT 1: 3-Seed Reliability Test")
    print("="*60)

    for algo in algorithms:
        all_results[algo] = {}
        for seed in seeds:
            print(f"\n  Running {algo.upper()} with seed {seed}...")
            rewards = run_one_trial(algo, seed, num_episodes)
            all_results[algo][seed] = rewards

    # ── Save raw results to CSV ────────────────────────────────
    with open('results/seed_experiment_raw.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'seed', 'episode', 'reward'])
        for algo in algorithms:
            for seed in seeds:
                for ep, r in enumerate(all_results[algo][seed]):
                    writer.writerow([algo, seed, ep+1, round(r,4)])
    print("\n📊 Raw results saved → results/seed_experiment_raw.csv")

    # ── Print summary table ────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Algorithm':<10} {'Seed1':>8} {'Seed2':>8} {'Seed3':>8} "
          f"{'Mean':>10} {'Std':>8}")
    print("="*65)

    summary = {}
    for algo in algorithms:
        final_avgs = []
        for seed in seeds:
            rewards = all_results[algo][seed]
            final_avg = np.mean(rewards[-50:])   # average of last 50 episodes
            final_avgs.append(final_avg)

        mean = np.mean(final_avgs)
        std  = np.std(final_avgs)
        summary[algo] = {'per_seed': final_avgs, 'mean': mean, 'std': std}

        print(f"{algo.upper():<10} "
              f"{final_avgs[0]:>8.1f} "
              f"{final_avgs[1]:>8.1f} "
              f"{final_avgs[2]:>8.1f} "
              f"{mean:>10.1f} "
              f"±{std:>6.1f}")
    print("="*65)

    # ── Save summary to CSV ────────────────────────────────────
    with open('results/seed_experiment_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'seed1', 'seed2', 'seed3', 'mean', 'std'])
        for algo in algorithms:
            s = summary[algo]
            writer.writerow([algo.upper(),
                             round(s['per_seed'][0],2),
                             round(s['per_seed'][1],2),
                             round(s['per_seed'][2],2),
                             round(s['mean'],2),
                             round(s['std'],2)])
    print("📊 Summary saved → results/seed_experiment_summary.csv")

    # ── Plot seed comparison graph ─────────────────────────────
    colors = {'iql': 'tomato', 'qmix': 'steelblue', 'mappo': 'seagreen'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, algo in zip(axes, algorithms):
        for i, seed in enumerate(seeds):
            rewards = all_results[algo][seed]
            window  = 20
            avg = [np.mean(rewards[max(0,j-window):j+1]) for j in range(len(rewards))]
            ax.plot(avg, alpha=0.8, linewidth=1.5, label=f'Seed {seed}')

        ax.set_title(f'{algo.upper()} — 3 Seeds', fontsize=13)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg Reward (window=20)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Reliability Test — 3 Random Seeds per Algorithm', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/seed_experiment_graph.png', dpi=150)
    print("📈 Graph saved → results/seed_experiment_graph.png")
    plt.show()

    return summary


# ============================================================
# EXPERIMENT 2 — SCALABILITY TEST (5 vs 10 agents)
# ============================================================
def run_scalability_test(num_episodes=150):
    """
    Tests QMIX with 5 agents vs 10 agents.
    Proves the system scales without breaking.
    Saves graph to results/scalability_graph.png
    """
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*60)
    print("  EXPERIMENT 2: Scalability Test (5 vs 10 agents)")
    print("="*60)

    results = {}

    for num_agents in [5, 10]:
        state_dim        = 6
        global_state_dim = num_agents * state_dim   # 30 or 60

        print(f"\n  Running QMIX with {num_agents} agents...")

        torch.manual_seed(42)
        np.random.seed(42)

        # For 10 agents, use DummyEnv (Aditi's env only has 5 agents)
        if num_agents == 5 and USE_REAL_ENV:
            env = make_env(num_agents=5)
        else:
            env = DummySmartGridEnv(num_agents=num_agents)

        agent = QMIXAgent(num_agents=num_agents, state_dim=state_dim,
                          action_dim=4, global_state_dim=global_state_dim)

        episode_rewards = []
        epsilon = 1.0

        for episode in range(num_episodes):
            states, global_state = env.reset()
            episode_reward = 0.0

            for step in range(96):
                actions = agent.select_action(states, epsilon)
                next_states, rewards, done, next_global_state = env.step(actions)
                agent.store_experience(states, actions, float(np.mean(rewards)),
                                       next_states, global_state,
                                       next_global_state, done)
                agent.train_step()
                episode_reward += sum(rewards)
                states       = next_states
                global_state = next_global_state
                if done: break

            epsilon = max(0.05, epsilon * 0.995)
            episode_rewards.append(episode_reward)

            if (episode+1) % 25 == 0:
                avg = np.mean(episode_rewards[-25:])
                print(f"    [QMIX {num_agents} agents] "
                      f"Ep {episode+1:3d}/{num_episodes} | "
                      f"Avg Reward: {avg:.1f}")

        results[num_agents] = episode_rewards

    # ── Save to CSV ────────────────────────────────────────────
    with open('results/scalability_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_agents', 'episode', 'reward'])
        for n, rewards in results.items():
            for ep, r in enumerate(rewards):
                writer.writerow([n, ep+1, round(r,4)])
    print("\n📊 Scalability results saved → results/scalability_results.csv")

    # ── Plot scalability graph ─────────────────────────────────
    plt.figure(figsize=(10, 5))
    colors = {5: 'steelblue', 10: 'darkorange'}

    for num_agents, rewards in results.items():
        window = 20
        avg = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        plt.plot(avg, color=colors[num_agents], linewidth=2,
                 label=f'QMIX — {num_agents} agents')

    plt.title('Scalability Test — QMIX with 5 vs 10 Agents', fontsize=13)
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward (window=20)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/scalability_graph.png', dpi=150)
    print("📈 Graph saved → results/scalability_graph.png")
    plt.show()

    # ── Print summary ──────────────────────────────────────────
    print("\n" + "="*40)
    print(f"{'Agents':<10} {'Final Avg Reward':>18}")
    print("="*40)
    for n, rewards in results.items():
        print(f"{n:<10} {np.mean(rewards[-30:]):>18.1f}")
    print("="*40)

    return results


# ============================================================
# RUN THIS FILE
# cd MARL
# python training/week7_experiments.py
# ============================================================
if __name__ == "__main__":
    print("Week 7 Experiments")
    print("="*40)
    print("1. Seed experiments (3 seeds × 3 algorithms)")
    print("2. Scalability test (5 vs 10 agents)")
    print("3. Run both")
    print()
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == '1':
        run_seed_experiments(seeds=[42, 123, 999], num_episodes=200)

    elif choice == '2':
        run_scalability_test(num_episodes=150)

    elif choice == '3':
        summary = run_seed_experiments(seeds=[42, 123, 999], num_episodes=200)
        run_scalability_test(num_episodes=150)
        print("\n✅ All Week 7 experiments complete!")
        print("Files saved in results/ folder:")
        print("  seed_experiment_graph.png")
        print("  seed_experiment_summary.csv")
        print("  scalability_graph.png")
        print("  scalability_results.csv")

    else:
        print("Running both by default...")
        run_seed_experiments(seeds=[42, 123, 999], num_episodes=200)
        run_scalability_test(num_episodes=150)