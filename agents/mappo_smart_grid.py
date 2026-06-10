# ============================================================
# mappo_smart_grid.py  —  VEEKSHA'S FILE  (agents/ folder)
# MAPPO = Multi-Agent Proximal Policy Optimization
#
# How MAPPO is different from IQL and QMIX:
#   IQL   — each agent learns alone, no coordination
#   QMIX  — agents share a mixing network (value-based)
#   MAPPO — agents share a central critic that sees ALL states
#            and uses policy gradients instead of Q-values
#
# The key idea: each agent has its OWN actor (makes decisions)
# but ALL agents share ONE critic (judges how good decisions were)
# The critic sees the global state = all 5 agents' observations
# This is called Centralized Training Decentralized Execution (CTDE)
# ============================================================

import torch                          # PyTorch for neural networks
import torch.nn as nn                 # Neural network modules
import torch.nn.functional as F       # Activation functions
import torch.optim as optim           # Optimizers
import numpy as np                    # Numerical operations
from collections import deque         # Replay buffer
import os                             # File operations
import sys

# Reuse ActorNetwork and CriticNetwork you already built in marl_smart_grid.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from marl_smart_grid import ActorNetwork, CriticNetwork


# ============================================================
# MAPPO AGENT — one per agent (5 total)
# Each agent has its own actor network
# All agents share the same central critic
# ============================================================
class MAPPOAgent:
    """
    One MAPPO agent. Has its own actor but uses shared critic.
    Stores its own trajectory (states, actions, rewards, log_probs)
    for PPO update at end of each episode.
    """

    def __init__(self, agent_id, state_dim=6, action_dim=4,
                 hidden_dim=128, lr_actor=3e-4):

        self.agent_id   = agent_id    # Which agent am I? (0 to 4)
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor network — makes decisions based on LOCAL state only
        # (Decentralized Execution)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Trajectory storage — collects one full episode before updating
        self.states    = []   # Local states seen this episode
        self.actions   = []   # Actions taken this episode
        self.log_probs = []   # Log probability of each action (needed for PPO ratio)
        self.rewards   = []   # Rewards received this episode
        self.dones     = []   # Done flags this episode

    def select_action(self, state):
        """
        Selects an action using the actor network.
        Returns: action (int), log_prob (tensor)
        log_prob is needed for PPO clipping update later.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(state_tensor)               # Raw action scores
            probs  = F.softmax(logits, dim=-1)              # Convert to probabilities
            dist   = torch.distributions.Categorical(probs) # Probability distribution
            action = dist.sample()                          # Sample one action
            log_prob = dist.log_prob(action)                # Log probability of that action

        return action.item(), log_prob.item()

    def store_transition(self, state, action, log_prob, reward, done):
        """Store one step of experience in trajectory buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_trajectory(self):
        """Clear trajectory after each PPO update."""
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.dones     = []

    def compute_returns(self, gamma=0.99):
        """
        Compute discounted returns for each step.
        Return at step t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        This is what the agent is trying to maximise.
        """
        returns = []
        G = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            G = reward + gamma * G * (1 - done)
            returns.insert(0, G)
        return returns


# ============================================================
# MAPPO SYSTEM — manages all 5 agents + shared central critic
# ============================================================
class MAPPOSystem:
    """
    The full MAPPO system.
    Creates 5 MAPPOAgents and one shared CentralCritic.
    Runs PPO updates after each episode using collected trajectories.
    """

    def __init__(self, num_agents=5, state_dim=6, action_dim=4,
                 hidden_dim=128, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, clip_eps=0.2, ppo_epochs=4,
                 global_state_dim=30):

        self.num_agents      = num_agents
        self.state_dim       = state_dim
        self.action_dim      = action_dim
        self.gamma           = gamma
        self.clip_eps        = clip_eps    # PPO clipping range (0.2 = standard)
        self.ppo_epochs      = ppo_epochs  # How many times to reuse each batch
        self.global_state_dim = global_state_dim
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create 5 independent actors (one per agent)
        self.agents = [
            MAPPOAgent(i, state_dim, action_dim, hidden_dim, lr_actor)
            for i in range(num_agents)
        ]

        # ONE shared central critic — sees all agents' states combined
        # Input = global_state_dim (30 = 5 agents x 6 values each)
        # Output = single value V(s) — how good is this global state
        self.central_critic = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim * 2),  # 30 → 256
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),         # 256 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),        # 128 → 64
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)                  # 64 → 1 value
        ).to(self.device)

        self.critic_optimizer = optim.Adam(
            self.central_critic.parameters(), lr=lr_critic
        )

    def select_actions(self, states, epsilon=None):
        """
        Get actions from all 5 agents.
        epsilon is accepted but ignored — MAPPO uses probability sampling,
        not epsilon-greedy. This keeps the trainer interface consistent.
        Returns: actions (list of 5 ints)
        """
        actions   = []
        log_probs = []

        for i, agent in enumerate(self.agents):
            action, log_prob = agent.select_action(states[i])
            actions.append(action)
            log_probs.append(log_prob)

        # Store log_probs so update() can use them
        self._current_log_probs = log_probs
        return actions

    def store_transitions(self, states, actions, rewards, dones):
        """Store one timestep of experience for all agents."""
        for i, agent in enumerate(self.agents):
            agent.store_transition(
                states[i],
                actions[i],
                self._current_log_probs[i],
                rewards[i],
                dones
            )

    def update(self, global_states_trajectory):
        """
        PPO update — called once at the END of each episode.
        Uses all the stored trajectories to update actors and critic.

        global_states_trajectory: list of global states from the episode
        (shape: [steps, global_state_dim])

        PPO key idea:
            ratio = new_prob / old_prob
            clipped_ratio = clip(ratio, 1-eps, 1+eps)
            loss = -min(ratio * advantage, clipped_ratio * advantage)
        The clipping prevents the policy from changing too drastically.
        """
        total_actor_loss  = 0.0
        total_critic_loss = 0.0

        # ── Compute returns for each agent ─────────────────────
        all_returns = []
        for agent in self.agents:
            returns = agent.compute_returns(self.gamma)
            all_returns.append(returns)

        # Average returns across agents (cooperative — shared reward)
        avg_returns = np.mean(all_returns, axis=0)

        # Convert to tensor
        returns_tensor = torch.FloatTensor(avg_returns).to(self.device)

        # ── Update central critic ───────────────────────────────
        # Convert global state trajectory to tensor
        global_states_np = np.array(global_states_trajectory)
        global_states_t  = torch.FloatTensor(global_states_np).to(self.device)

        # Truncate to match returns length (episode may end early)
        min_len = min(len(returns_tensor), global_states_t.shape[0])
        returns_tensor  = returns_tensor[:min_len]
        global_states_t = global_states_t[:min_len]

        # Critic predicts state values
        values = self.central_critic(global_states_t).squeeze(-1)  # Shape: (steps,)

        # Critic loss = MSE between predicted values and actual returns
        critic_loss = F.mse_loss(values, returns_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.central_critic.parameters(), 0.5)
        self.critic_optimizer.step()
        total_critic_loss = critic_loss.item()

        # Compute advantages = actual returns - critic baseline
        with torch.no_grad():
            values_detached = self.central_critic(global_states_t).squeeze(-1)
            advantages = returns_tensor - values_detached

            # Normalize advantages for stable training
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Update each agent's actor using PPO clipping ────────
        for agent in self.agents:
            if len(agent.states) == 0:
                continue

            # Convert trajectory to tensors
            states_t   = torch.FloatTensor(
                np.array(agent.states)[:min_len]
            ).to(self.device)

            actions_t  = torch.LongTensor(
                agent.actions[:min_len]
            ).to(self.device)

            old_log_probs_t = torch.FloatTensor(
                agent.log_probs[:min_len]
            ).to(self.device)

            # PPO update for ppo_epochs iterations on same data
            for _ in range(self.ppo_epochs):

                # Get new action probabilities from updated actor
                logits   = agent.actor(states_t)
                probs    = F.softmax(logits, dim=-1)
                dist     = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions_t)

                # PPO ratio = new_prob / old_prob (in log space = difference)
                ratio = torch.exp(new_log_probs - old_log_probs_t)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                           1 + self.clip_eps) * advantages

                # Actor loss — negative because we MAXIMISE reward
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus — encourages exploration
                entropy = dist.entropy().mean()
                actor_loss = actor_loss - 0.01 * entropy

                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor_optimizer.step()

            total_actor_loss += actor_loss.item()

        # Clear all agent trajectories after update
        for agent in self.agents:
            agent.clear_trajectory()

        avg_actor_loss = total_actor_loss / self.num_agents
        return avg_actor_loss, total_critic_loss

    def update_target_networks(self):
        """
        Called by trainer for consistency — MAPPO doesn't use target
        networks (PPO is on-policy) so this does nothing.
        """
        pass

    def save_models(self, filepath_prefix):
        """Save all actor networks and the central critic."""
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(),
                       f"{filepath_prefix}_mappo_actor_{i}.pth")
        torch.save(self.central_critic.state_dict(),
                   f"{filepath_prefix}_mappo_critic.pth")
        print(f"✅ MAPPO models saved → {filepath_prefix}_mappo_*.pth")

    def load_models(self, filepath_prefix):
        """Load all actor networks and the central critic."""
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(
                torch.load(f"{filepath_prefix}_mappo_actor_{i}.pth"))
        self.central_critic.load_state_dict(
            torch.load(f"{filepath_prefix}_mappo_critic.pth"))
        print(f"✅ MAPPO models loaded from {filepath_prefix}_mappo_*.pth")


# ============================================================
# QUICK TEST — run this file directly to check it works
# python agents/mappo_smart_grid.py
# ============================================================
if __name__ == "__main__":
    print("Testing MAPPOSystem...")

    system = MAPPOSystem(num_agents=5, state_dim=6,
                         action_dim=4, global_state_dim=30)

    # Fake one episode of 10 steps
    global_states_traj = []

    dummy_states = [np.random.rand(6) for _ in range(5)]
    for step in range(10):
        actions = system.select_actions(dummy_states)
        rewards = [np.random.uniform(0, 1)] * 5
        done    = (step == 9)
        global_state = np.concatenate(dummy_states)
        global_states_traj.append(global_state)
        system.store_transitions(dummy_states, actions, rewards, done)

    actor_loss, critic_loss = system.update(global_states_traj)

    print(f"✅ MAPPOSystem works")
    print(f"   Actor loss  : {actor_loss:.4f}")
    print(f"   Critic loss : {critic_loss:.4f}")
    print(f"✅ Ready to plug into trainer.py")