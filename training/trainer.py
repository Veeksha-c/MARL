# ============================================================
# trainer.py  —  VEEKSHA'S FILE  (training/ folder)
# This file connects your QMIX and IQL agents to the environment.
# Right now it uses DummySmartGridEnv so you can train immediately.
# When Aditi finishes smart_grid_env.py, flip USE_REAL_ENV = True
# ============================================================

import numpy as np
import torch
import csv
import os
import sys
import matplotlib.pyplot as plt

# So Python can find the agents/ folder (one level up from training/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'agents'))   # ← points to MARL/agents/

# ── Import YOUR actual agent files ────────────────────────────
# These match the class names inside qmix_smart_grid.py and iql_smart_grid.py
from qmix_smart_grid  import QMIXAgent                      # class QMIXAgent
from iql_smart_grid   import IQLAgent, IQLSmartGridSystem   # class IQLAgent, IQLSmartGridSystem
from mappo_smart_grid import MAPPOSystem                     # class MAPPOSystem

# ── Flip this to True when Aditi sends smart_grid_env.py ──────
USE_REAL_ENV = True


# ============================================================
# ADAPTER WRAPPER — bridges Aditi's env to your trainer
# Aditi used standard Gymnasium format. Your trainer uses a
# custom format. This wrapper converts between the two so
# neither of you has to change your code.
# ============================================================
class SmartGridEnvWrapper:
    """
    Wraps Aditi's SmartGridEnv to match the format trainer expects.

    Aditi's env returns:
        reset() → (obs_dict, info)
        step()  → (obs_dict, reward, terminated, truncated, info)

    Trainer expects:
        reset() → (states_list, global_state)
        step()  → (next_states, rewards, done, next_global_state)
    """

    def __init__(self, env, num_agents=5):
        self.env        = env
        self.num_agents = num_agents

    def _obs_dict_to_array(self, obs_dict):
        """Convert Aditi's obs dict → flat numpy array shape (6,)"""
        return np.array([
            obs_dict['battery_soc'],
            obs_dict['solar_output'],
            obs_dict['wind_output'],
            obs_dict['electricity_price'],
            obs_dict['demand'],
            obs_dict['time_step'] / 95.0    # normalise to [0,1]
        ], dtype=np.float32)

    def _to_trainer_format(self, obs_dict):
        """Convert one obs dict → list of 5 identical arrays + global state"""
        arr          = self._obs_dict_to_array(obs_dict)
        states       = [arr.copy() for _ in range(self.num_agents)]
        global_state = np.concatenate(states)   # shape (30,)
        return states, global_state

    def reset(self):
        obs_dict, info       = self.env.reset()
        states, global_state = self._to_trainer_format(obs_dict)
        return states, global_state

    def step(self, actions):
        """
        actions: list of 5 ints from trainer
        Converts to Aditi's dict format, calls her step(), converts back.
        """
        agent_names = ['solar_agent', 'wind_agent', 'battery_agent',
                       'grid_agent', 'load_agent']
        action_dict = {name: int(actions[i])
                       for i, name in enumerate(agent_names)}

        obs_dict, reward, terminated, truncated, info = self.env.step(action_dict)

        states, global_state = self._to_trainer_format(obs_dict)
        done    = terminated or truncated
        rewards = [float(reward)] * self.num_agents   # same reward for all agents

        return states, rewards, done, global_state


# ============================================================
# DUMMY ENVIRONMENT
# Runs without Aditi's file. Returns data in the exact same
# format that SmartGridEnv will return — so your trainer works
# identically with both.
# ============================================================
class DummySmartGridEnv:

    def __init__(self, num_agents=5, max_steps=96):
        self.num_agents   = num_agents
        self.max_steps    = max_steps
        self.current_step = 0
        self.battery_soc  = 0.5

    def _get_obs_dict(self):
        t      = self.current_step / self.max_steps
        solar  = float(max(0, np.sin(np.pi * t)))
        wind   = float(np.random.uniform(0.1, 0.8))
        price  = float(np.random.uniform(0.2, 0.9))
        demand = float(np.random.uniform(0.3, 1.0))
        return {
            'battery_soc':       round(self.battery_soc, 4),
            'solar_output':      round(solar,  4),
            'wind_output':       round(wind,   4),
            'electricity_price': round(price,  4),
            'demand':            round(demand, 4),
            'time_step':         self.current_step
        }

    def _obs_to_array(self, obs_dict):
        return np.array([
            obs_dict['battery_soc'],
            obs_dict['solar_output'],
            obs_dict['wind_output'],
            obs_dict['electricity_price'],
            obs_dict['demand'],
            obs_dict['time_step'] / 95.0
        ], dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.battery_soc  = 0.5
        obs_dict     = self._get_obs_dict()
        states       = [self._obs_to_array(obs_dict) for _ in range(self.num_agents)]
        global_state = np.concatenate(states)
        return states, global_state

    def step(self, actions):
        self.current_step += 1

        battery_action = actions[2]
        if battery_action == 1:
            self.battery_soc = min(0.95, self.battery_soc + 0.05)
        elif battery_action == 2:
            self.battery_soc = max(0.10, self.battery_soc - 0.05)

        obs_dict          = self._get_obs_dict()
        next_states       = [self._obs_to_array(obs_dict) for _ in range(self.num_agents)]
        next_global_state = np.concatenate(next_states)

        reward  = obs_dict['solar_output']
        reward += obs_dict['wind_output'] * 0.5
        if obs_dict['electricity_price'] > 0.7:
            reward -= 2.0
        if self.battery_soc < 0.1:
            reward -= 3.0
        if self.battery_soc > 0.9:
            reward += 0.5

        rewards = [reward] * self.num_agents
        done    = self.current_step >= self.max_steps
        return next_states, rewards, done, next_global_state


# ============================================================
# MAIN TRAINER CLASS
# ============================================================
class MARLTrainer:

    def __init__(self, algorithm='qmix', num_episodes=300, num_agents=5,
                 state_dim=6, action_dim=4, global_state_dim=30,
                 results_dir='results'):

        self.algorithm    = algorithm
        self.num_episodes = num_episodes
        self.num_agents   = num_agents
        self.results_dir  = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # ── Choose environment ─────────────────────────────────
        if USE_REAL_ENV:
            sys.path.append(os.path.join(ROOT_DIR, 'environment'))
            from smart_grid_env import SmartGridEnv
            self.env = SmartGridEnvWrapper(SmartGridEnv(), num_agents)
            print("✅ Using Aditi's SmartGridEnv (wrapped)")
        else:
            self.env = DummySmartGridEnv(num_agents)
            print("⚠️  Using DummyEnv — flip USE_REAL_ENV=True when Aditi is ready")

        # ── Choose algorithm ───────────────────────────────────
        if algorithm == 'qmix':
            # QMIXAgent from qmix_smart_grid.py
            self.agent = QMIXAgent(
                num_agents       = num_agents,
                state_dim        = state_dim,
                action_dim       = action_dim,
                global_state_dim = global_state_dim
            )
            print("✅ QMIXAgent ready")

        elif algorithm == 'iql':
            # IQLSmartGridSystem manages 5 IQLAgents from iql_smart_grid.py
            self.iql_system = IQLSmartGridSystem(
                num_agents = num_agents,
                state_dim  = state_dim,
                action_dim = action_dim
            )
            self.agent = None   # IQL uses self.iql_system instead
            print("✅ IQL system (5 independent agents) ready")

        elif algorithm == 'mappo':
            self.agent = MAPPOSystem(
                num_agents       = num_agents,
                state_dim        = state_dim,
                action_dim       = action_dim,
                global_state_dim = global_state_dim
            )
            print("✅ MAPPO system (5 actors + central critic) ready")

        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'qmix', 'iql', or 'mappo'.")

        self.episode_rewards = []
        self.episode_losses  = []
        self.best_reward     = -float('inf')

        print(f"\n{'='*50}")
        print(f"  Algorithm : {algorithm.upper()}")
        print(f"  Episodes  : {num_episodes}")
        print(f"  Agents    : {num_agents}")
        print(f"  Results → : {results_dir}/")
        print(f"{'='*50}\n")


    def _iql_select_actions(self, states):
        """
        Each IQLAgent uses act(state) — no epsilon argument,
        epsilon is managed internally by each agent.
        """
        actions = []
        for i, agent in enumerate(self.iql_system.agents):
            action = agent.act(states[i])
            actions.append(action)
        return actions


    def _iql_store_and_train(self, states, actions, rewards, next_states, done):
        """Store experience and replay for each IQL agent independently."""
        for i, agent in enumerate(self.iql_system.agents):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], done)
            agent.replay()


    def train(self, epsilon_start=1.0, epsilon_end=0.05,
              epsilon_decay=0.995, target_update_freq=10):

        epsilon = epsilon_start

        for episode in range(self.num_episodes):

            states, global_state = self.env.reset()
            episode_reward = 0.0
            episode_loss   = 0.0
            loss_count     = 0
            global_states_traj = []   # MAPPO needs full episode trajectory

            for step in range(96):   # 96 steps = 24 hours

                # ── Select actions ─────────────────────────────
                if self.algorithm == 'qmix':
                    actions = self.agent.select_action(states, epsilon)
                elif self.algorithm == 'mappo':
                    actions = self.agent.select_actions(states)
                else:
                    actions = self._iql_select_actions(states)

                # ── Step environment ───────────────────────────
                next_states, rewards, done, next_global_state = self.env.step(actions)

                # ── Store + train ──────────────────────────────
                if self.algorithm == 'qmix':
                    scalar_reward = float(np.mean(rewards))
                    self.agent.store_experience(
                        states, actions, scalar_reward,
                        next_states, global_state,
                        next_global_state, done
                    )
                    loss = self.agent.train_step()
                    if loss and loss > 0:
                        episode_loss += loss
                        loss_count   += 1
                elif self.algorithm == 'mappo':
                    # MAPPO collects full episode then updates once at the end
                    global_states_traj.append(global_state)
                    self.agent.store_transitions(states, actions, rewards, done)
                else:
                    self._iql_store_and_train(states, actions, rewards, next_states, done)

                episode_reward += sum(rewards)
                states         = next_states
                global_state   = next_global_state

                if done:
                    break

            # ── MAPPO update — once per episode ────────────────
            if self.algorithm == 'mappo' and len(global_states_traj) > 0:
                actor_loss, critic_loss = self.agent.update(global_states_traj)
                episode_loss = actor_loss
                loss_count   = 1

            # ── End of episode ─────────────────────────────────
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if (episode + 1) % target_update_freq == 0:
                if self.algorithm == 'qmix':
                    self.agent.update_target_networks()

            avg_loss = episode_loss / max(1, loss_count)
            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(avg_loss)

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward

            if (episode + 1) % 50 == 0:
                recent_avg = np.mean(self.episode_rewards[-50:])
                print(f"Episode {episode+1:4d}/{self.num_episodes} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg(50): {recent_avg:7.2f} | "
                      f"ε: {epsilon:.3f}")

        print(f"\n✅ Training done. Best reward: {self.best_reward:.2f}")
        self._save_csv()
        self._save_model()
        return self.episode_rewards, self.episode_losses


    def _save_model(self):
        """
        Saves the trained model to results/models/ so it can be loaded
        later by inference.py for predictions on new data.
        """
        models_dir = os.path.join(self.results_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        prefix = os.path.join(models_dir, self.algorithm)

        try:
            if self.algorithm == 'mappo':
                self.agent.save_models(prefix)
            elif self.algorithm == 'qmix':
                self.agent.save_models(prefix + '_model.pth')
            elif self.algorithm == 'iql':
                for i, agent in enumerate(self.iql_system.agents):
                    torch.save(agent.q_network.state_dict(),
                               f"{prefix}_agent_{i}.pth")
                print(f"✅ IQL models saved → {prefix}_agent_*.pth")
        except Exception as e:
            print(f"⚠️  Could not save model: {e}")


    def _save_csv(self):
        path = os.path.join(self.results_dir, f"{self.algorithm}_results.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'total_reward', 'avg_loss'])
            for i, (r, l) in enumerate(zip(self.episode_rewards, self.episode_losses)):
                writer.writerow([i+1, round(r,4), round(l,6)])
        print(f"📊 Results saved → {path}")


    def plot_reward_curve(self, window=20, save=True):
        episodes   = list(range(1, len(self.episode_rewards) + 1))
        moving_avg = [
            np.mean(self.episode_rewards[max(0, i-window):i+1])
            for i in range(len(self.episode_rewards))
        ]
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, self.episode_rewards, alpha=0.3,
                 color='steelblue', label='Episode Reward')
        plt.plot(episodes, moving_avg, color='steelblue',
                 linewidth=2, label=f'Moving Avg ({window})')
        plt.title(f'{self.algorithm.upper()} — Training Reward Curve')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            fig_path = os.path.join(self.results_dir, f"{self.algorithm}_reward_curve.png")
            plt.savefig(fig_path, dpi=150)
            print(f"📈 Graph saved → {fig_path}")
        plt.show()


# ============================================================
# COMPARISON RUNNER — use this in Week 6
# ============================================================
def run_comparison(num_episodes=300):
    results = {}
    for algo in ['iql', 'qmix', 'mappo']:
        print(f"\n{'='*50}  TRAINING {algo.upper()}  {'='*50}")
        trainer = MARLTrainer(algorithm=algo, num_episodes=num_episodes)
        rewards, _ = trainer.train()
        results[algo] = rewards
        trainer.plot_reward_curve(save=True)

    plt.figure(figsize=(12, 5))
    colors = {'iql': 'tomato', 'qmix': 'steelblue', 'mappo': 'seagreen'}
    for algo, rewards in results.items():
        avg = [np.mean(rewards[max(0,i-20):i+1]) for i in range(len(rewards))]
        plt.plot(avg, color=colors[algo], linewidth=2, label=algo.upper())
    plt.title('IQL vs QMIX vs MAPPO — Reward Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward (window=20)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/comparison_all_algorithms.png', dpi=150)
    plt.show()

    print("\n" + "="*45)
    print(f"{'Algorithm':<10} {'Final Avg':>12} {'Best':>10}")
    print("="*45)
    for algo, rewards in results.items():
        print(f"{algo.upper():<10} {np.mean(rewards[-50:]):>12.2f} {max(rewards):>10.2f}")
    print("="*45)


# ============================================================
# RUN THIS FILE TO START TRAINING
# cd marl_smartgrid
# python training/trainer.py
# ============================================================
if __name__ == "__main__":
    print("MARL Smart Grid — Trainer")
    print("="*40)
    print("1. Train QMIX only")
    print("2. Train IQL only")
    print("3. Train MAPPO only")
    print("4. Compare all 3 — IQL vs QMIX vs MAPPO (Week 6)")
    print()
    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice == '1':
        trainer = MARLTrainer(algorithm='qmix', num_episodes=300)
        trainer.train()
        trainer.plot_reward_curve()
    elif choice == '2':
        trainer = MARLTrainer(algorithm='iql', num_episodes=300)
        trainer.train()
        trainer.plot_reward_curve()
    elif choice == '3':
        trainer = MARLTrainer(algorithm='mappo', num_episodes=300)
        trainer.train()
        trainer.plot_reward_curve()
    elif choice == '4':
        run_comparison(num_episodes=300)
    else:
        print("Running QMIX by default...")
        trainer = MARLTrainer(algorithm='qmix', num_episodes=300)
        trainer.train()
        trainer.plot_reward_curve()