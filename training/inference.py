# ============================================================
# inference.py  —  VEEKSHA'S FILE  (training/ folder)
#
# Loads the trained MAPPO model and predicts what each of the
# 5 agents would do, given ANY new observation values.
#
# This is the "brain" behind your dashboard.
#
# Usage:
#   from inference import SmartGridPredictor
#   predictor = SmartGridPredictor()
#   result = predictor.predict({
#       'battery_soc': 0.3,
#       'solar_output': 0.8,
#       'wind_output': 0.2,
#       'electricity_price': 0.9,
#       'demand': 0.7,
#       'time_step': 72
#   })
#   print(result)
# ============================================================

import torch
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'agents'))

from marl_smart_grid import ActorNetwork


# ── Action meaning lookup (from interface.md) ──────────────────
ACTION_MEANINGS = {
    'solar_agent':   {0: 'idle',     1: 'store',    2: 'supply',   3: 'curtail'},
    'wind_agent':    {0: 'idle',     1: 'store',    2: 'supply',   3: 'curtail'},
    'battery_agent': {0: 'idle',     1: 'charge',   2: 'discharge', 3: 'hold'},
    'grid_agent':    {0: 'idle',     1: 'buy',      2: 'sell',     3: 'standby'},
    'load_agent':    {0: 'normal',   1: 'reduce',   2: 'shift',    3: 'priority'},
}

AGENT_ORDER = ['solar_agent', 'wind_agent', 'battery_agent', 'grid_agent', 'load_agent']


class SmartGridPredictor:
    """
    Loads trained MAPPO actor networks and predicts agent decisions
    for any given observation.
    """

    def __init__(self, model_dir=None, state_dim=6, action_dim=4,
                 hidden_dim=128, num_agents=5):

        if model_dir is None:
            model_dir = os.path.join(ROOT_DIR, 'results', 'models')

        self.model_dir  = model_dir
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load 5 trained actor networks (one per agent)
        self.actors = []
        for i in range(num_agents):
            actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            path  = os.path.join(model_dir, f"mappo_mappo_actor_{i}.pth")

            if os.path.exists(path):
                actor.load_state_dict(torch.load(path, map_location=self.device))
                actor.eval()
                print(f"✅ Loaded {AGENT_ORDER[i]} model from {path}")
            else:
                print(f"⚠️  Model not found at {path} — using untrained network")

            self.actors.append(actor)

    def _obs_dict_to_array(self, obs_dict):
        """Convert observation dict -> flat numpy array shape (6,)"""
        return np.array([
            obs_dict['battery_soc'],
            obs_dict['solar_output'],
            obs_dict['wind_output'],
            obs_dict['electricity_price'],
            obs_dict['demand'],
            obs_dict['time_step'] / 95.0
        ], dtype=np.float32)

    def predict(self, observation: dict, deterministic=True):
        """
        Given an observation dict, return what each of the 5 agents
        would do.

        observation = {
            'battery_soc':       float 0-1,
            'solar_output':      float 0-1,
            'wind_output':       float 0-1,
            'electricity_price': float 0-1,
            'demand':            float 0-1,
            'time_step':         int 0-95
        }

        deterministic=True  -> pick the highest-probability action (best for dashboard)
        deterministic=False -> sample from probability distribution (for exploration)

        Returns: dict with action numbers, action meanings, and confidence scores
        """
        state_arr = self._obs_dict_to_array(observation)
        state_t   = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)

        results = {}

        for i, agent_name in enumerate(AGENT_ORDER):
            with torch.no_grad():
                logits = self.actors[i](state_t)
                probs  = torch.softmax(logits, dim=-1).squeeze(0)

            if deterministic:
                action = int(torch.argmax(probs).item())
            else:
                dist   = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())

            confidence = float(probs[action].item())
            meaning    = ACTION_MEANINGS[agent_name][action]

            results[agent_name] = {
                'action':     action,
                'meaning':    meaning,
                'confidence': round(confidence, 3),
                'all_probs':  {ACTION_MEANINGS[agent_name][a]: round(float(probs[a]), 3)
                               for a in range(self.action_dim)}
            }

        return results

    def explain(self, observation: dict, prediction: dict = None) -> str:
        """
        Returns a human-readable explanation of the agents' decisions.
        Useful for displaying on the dashboard.
        """
        if prediction is None:
            prediction = self.predict(observation)

        lines = []
        lines.append(f"At time step {observation['time_step']} "
                      f"({observation['time_step']/4:.0f}:00 hrs):")
        lines.append(f"  Solar output: {observation['solar_output']:.2f} | "
                      f"Wind output: {observation['wind_output']:.2f} | "
                      f"Battery SOC: {observation['battery_soc']:.2f} | "
                      f"Price: {observation['electricity_price']:.2f} | "
                      f"Demand: {observation['demand']:.2f}")
        lines.append("")
        lines.append("Agent Decisions:")

        for agent_name in AGENT_ORDER:
            r = prediction[agent_name]
            label = agent_name.replace('_', ' ').title()
            lines.append(f"  {label:18s} -> {r['meaning'].upper():10s} "
                          f"(confidence: {r['confidence']*100:.0f}%)")

        return "\n".join(lines)


# ============================================================
# COST CALCULATOR — converts decisions into Rs saved
# ============================================================
def estimate_savings(observation: dict, prediction: dict,
                      price_per_unit_rs=6.0, unit_consumption_kwh=0.25):
    """
    Rough estimate of money saved by following the agent's
    recommendations vs a naive "always buy from grid" baseline.

    price_per_unit_rs: average retail price per kWh in Rs (Karnataka ~ Rs 6-8/kWh)
    unit_consumption_kwh: energy used per 15-min step at full demand (assumption)

    Returns: dict with baseline cost, optimized cost, and savings in Rs
    """
    demand = observation['demand']

    # Baseline: naive system always buys everything from grid
    baseline_units = demand * unit_consumption_kwh
    baseline_cost  = baseline_units * price_per_unit_rs

    # Optimized: subtract renewable contribution if agents chose to supply
    renewable_fraction = 0.0

    if prediction['solar_agent']['meaning'] == 'supply':
        renewable_fraction += observation['solar_output']

    if prediction['wind_agent']['meaning'] == 'supply':
        renewable_fraction += observation['wind_output']

    if prediction['battery_agent']['meaning'] == 'discharge':
        renewable_fraction += 0.2   # assume battery covers ~20% of demand when discharging

    renewable_fraction = min(renewable_fraction, 1.0)

    grid_fraction  = 1.0 - renewable_fraction

    # If grid agent chose NOT to buy during high price, extra savings
    price_multiplier = 1.0
    if prediction['grid_agent']['meaning'] in ['idle', 'standby']:
        if observation['electricity_price'] > 0.7:
            price_multiplier = 0.0   # avoided expensive grid purchase entirely
        else:
            price_multiplier = grid_fraction
    else:
        price_multiplier = grid_fraction

    optimized_cost = baseline_units * price_multiplier * price_per_unit_rs
    savings        = baseline_cost - optimized_cost

    return {
        'baseline_cost_rs':  round(baseline_cost, 2),
        'optimized_cost_rs': round(optimized_cost, 2),
        'savings_rs':        round(savings, 2),
        'savings_percent':   round((savings / baseline_cost * 100) if baseline_cost > 0 else 0, 1)
    }


# ============================================================
# QUICK TEST
# python training/inference.py
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Smart Grid Inference Test")
    print("="*60)

    predictor = SmartGridPredictor()

    # ── Test case 1: Sunny day, high price, low battery ──────────
    test_obs_1 = {
        'battery_soc':       0.3,
        'solar_output':      0.8,
        'wind_output':       0.2,
        'electricity_price': 0.9,
        'demand':            0.7,
        'time_step':         72   # 6 PM — evening peak
    }

    print("\n--- Test Case 1: Evening peak, sunny, expensive grid ---")
    prediction_1 = predictor.predict(test_obs_1)
    print(predictor.explain(test_obs_1, prediction_1))

    savings_1 = estimate_savings(test_obs_1, prediction_1)
    print(f"\nEstimated savings vs naive grid-only system:")
    print(f"  Baseline cost : Rs {savings_1['baseline_cost_rs']}")
    print(f"  Optimized cost: Rs {savings_1['optimized_cost_rs']}")
    print(f"  Savings       : Rs {savings_1['savings_rs']} "
          f"({savings_1['savings_percent']}%)")

    # ── Test case 2: Night time, no solar, cheap electricity ──────
    test_obs_2 = {
        'battery_soc':       0.6,
        'solar_output':      0.0,
        'wind_output':       0.5,
        'electricity_price': 0.2,
        'demand':            0.4,
        'time_step':         8    # 2 AM
    }

    print("\n--- Test Case 2: Night time, no solar, cheap grid ---")
    prediction_2 = predictor.predict(test_obs_2)
    print(predictor.explain(test_obs_2, prediction_2))

    savings_2 = estimate_savings(test_obs_2, prediction_2)
    print(f"\nEstimated savings vs naive grid-only system:")
    print(f"  Baseline cost : Rs {savings_2['baseline_cost_rs']}")
    print(f"  Optimized cost: Rs {savings_2['optimized_cost_rs']}")
    print(f"  Savings       : Rs {savings_2['savings_rs']} "
          f"({savings_2['savings_percent']}%)")

    print("\n✅ Inference script working — ready to plug into dashboard!")