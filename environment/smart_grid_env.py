# ============================================================
# smart_grid_env.py  —  ADITI'S FILE  (environment/ folder)
# ============================================================
# HOW TO USE THIS FILE:
#   - Every place you see  ← ADITI: fill this in  is your job
#   - Every place you see  ← DO NOT TOUCH  leave it exactly as is
#   - The format of what reset() and step() RETURN must never change
#   - Ask Veeksha before changing any variable names
# ============================================================

import numpy as np
import pandas as pd
import os

class SmartGridEnv:

    def __init__(self):
        self.num_agents  = 5     # DO NOT TOUCH — Veeksha's code expects exactly 5
        self.max_steps   = 96    # DO NOT TOUCH — 96 steps = 24 hours in 15-min intervals
        self.state_dim   = 6     # DO NOT TOUCH — 6 values in each observation
        self.action_dim  = 4     # DO NOT TOUCH — 4 possible actions per agent

        # ── Load your datasets here ──────────────────────────────────────
        # ← ADITI: load your cleaned CSV files here using pandas
        # Example:
        #   self.solar_data  = pd.read_csv('environment/data/solar_data.csv')
        #   self.demand_data = pd.read_csv('environment/data/demand_data.csv')
        #   self.price_data  = pd.read_csv('environment/data/price_data.csv')
        #   self.wind_data   = pd.read_csv('environment/data/wind_data.csv')
        #
        # For now, if data is not ready, leave these as None and use the
        # _simulate_solar() and _simulate_wind() helper functions below
        self.solar_data  = None   # ← ADITI: replace None with pd.read_csv(...)
        self.demand_data = None   # ← ADITI: replace None with pd.read_csv(...)
        self.price_data  = None   # ← ADITI: replace None with pd.read_csv(...)
        self.wind_data   = None   # ← ADITI: replace None with pd.read_csv(...)

        # ── Internal state variables ─────────────────────────────────────
        self.current_step  = 0    # Tracks which 15-min interval we are at (0 to 95)
        self.battery_soc   = 0.5  # Battery state of charge (0.0 = empty, 1.0 = full)
        self.day_index     = 0    # Which day from dataset to use (increases each episode)

    # ================================================================
    # reset() — called at the START of every new training episode
    # ================================================================
    def reset(self):
        """
        Resets the environment to the beginning of a new day.
        Called by Veeksha's trainer at the start of each episode.

        MUST RETURN: states, global_state
            states       — list of 5 numpy arrays, one per agent, shape (6,)
            global_state — one numpy array, shape (30,) — just concatenate the 5 states
        """
        # ← ADITI: reset these every episode — DO NOT change the values
        self.current_step = 0
        self.battery_soc  = 0.5   # Always start each episode at 50% battery

        # Move to next day in dataset each episode
        # ← ADITI: increment day_index so each episode uses a different day of data
        self.day_index = (self.day_index + 1) % 365   # loops back after 1 year

        # Build the first observation
        obs_dict = self._get_obs_dict()

        # Convert to arrays — DO NOT TOUCH this part
        states       = [self._obs_to_array(obs_dict) for _ in range(self.num_agents)]
        global_state = np.concatenate(states)

        return states, global_state   # ← DO NOT TOUCH — Veeksha expects exactly this


    # ================================================================
    # step() — called EVERY 15 minutes during an episode
    # ================================================================
    def step(self, actions):
        """
        Takes one step forward in time (15 minutes).
        Called by Veeksha's trainer at every step of each episode.

        INPUT:
            actions — list of 5 integers, one per agent
                actions[0] = Solar Agent  action  (0=idle, 1=store, 2=supply, 3=curtail)
                actions[1] = Wind Agent   action  (0=idle, 1=store, 2=supply, 3=curtail)
                actions[2] = Battery Agent action (0=idle, 1=charge, 2=discharge, 3=hold)
                actions[3] = Grid Agent   action  (0=idle, 1=buy,   2=sell,    3=standby)
                actions[4] = Load Agent   action  (0=normal, 1=reduce, 2=shift, 3=priority)

        MUST RETURN: next_states, rewards, done, next_global_state
            next_states       — list of 5 numpy arrays shape (6,)
            rewards           — list of 5 floats (same value for all — cooperative)
            done              — True if episode is over, False otherwise
            next_global_state — numpy array shape (30,)
        """
        self.current_step += 1

        # ── Update battery based on Battery Agent's action ───────────────
        # ← ADITI: this is the battery SOC formula from the roadmap
        battery_action = actions[2]   # Agent index 2 = Battery Agent

        if battery_action == 1:       # Charge the battery
            # ← ADITI: apply charging formula here
            # New SOC = Old SOC + (charging_power × efficiency)
            # Charging efficiency = 90% = 0.9
            # Assume charging power = 0.05 per step (5% of capacity)
            self.battery_soc = min(0.95, self.battery_soc + (0.05 * 0.9))   # Max 95%

        elif battery_action == 2:     # Discharge the battery
            # ← ADITI: apply discharging formula here
            # New SOC = Old SOC - discharging_power
            self.battery_soc = max(0.10, self.battery_soc - 0.05)           # Min 10%

        # Actions 0 (idle) and 3 (hold) → battery SOC stays the same

        # ── Get new observation after action ────────────────────────────
        obs_dict = self._get_obs_dict()

        # ── Calculate reward ─────────────────────────────────────────────
        reward = self._calculate_reward(obs_dict, actions)

        # ── Build return values — DO NOT TOUCH FORMAT ────────────────────
        next_states       = [self._obs_to_array(obs_dict) for _ in range(self.num_agents)]
        next_global_state = np.concatenate(next_states)
        rewards           = [reward] * self.num_agents   # Same reward for all (cooperative)
        done              = (self.current_step >= self.max_steps)

        return next_states, rewards, done, next_global_state   # ← DO NOT TOUCH


    # ================================================================
    # _get_obs_dict() — builds the observation for the current step
    # ================================================================
    def _get_obs_dict(self):
        """
        Reads the current values for this time step and returns a dictionary.

        ← ADITI: this is where you read from your CSV datasets.
        Each value must be normalized between 0.0 and 1.0.
        """

        t = self.current_step   # Current time step (0 to 95)

        # ── Solar output ─────────────────────────────────────────────────
        # ← ADITI: if solar_data is loaded, read from CSV row for this time step
        # Otherwise use the sine simulation below
        if self.solar_data is not None:
            solar = float(self.solar_data.iloc[t]['solar_normalized'])  # ← column name may differ
        else:
            solar = self._simulate_solar(t)   # Fallback simulation

        # ── Wind output ──────────────────────────────────────────────────
        # ← ADITI: same — read from wind_data CSV or use simulation
        if self.wind_data is not None:
            wind = float(self.wind_data.iloc[t]['wind_normalized'])     # ← column name may differ
        else:
            wind = self._simulate_wind()      # Fallback simulation

        # ── Electricity price ────────────────────────────────────────────
        # ← ADITI: read from price_data CSV for this time step
        if self.price_data is not None:
            price = float(self.price_data.iloc[t]['price_normalized'])  # ← column name may differ
        else:
            price = self._simulate_price(t)   # Fallback simulation

        # ── Demand ───────────────────────────────────────────────────────
        # ← ADITI: read from demand_data CSV for this time step
        if self.demand_data is not None:
            demand = float(self.demand_data.iloc[t]['demand_normalized'])# ← column name may differ
        else:
            demand = self._simulate_demand(t)  # Fallback simulation

        # ── Return observation dictionary — DO NOT CHANGE KEY NAMES ─────
        return {
            'battery_soc':       round(float(self.battery_soc), 4),  # DO NOT TOUCH key name
            'solar_output':      round(solar,  4),                    # DO NOT TOUCH key name
            'wind_output':       round(wind,   4),                    # DO NOT TOUCH key name
            'electricity_price': round(price,  4),                    # DO NOT TOUCH key name
            'demand':            round(demand, 4),                    # DO NOT TOUCH key name
            'time_step':         int(self.current_step)               # DO NOT TOUCH key name
        }


    # ================================================================
    # _calculate_reward() — reward function
    # ================================================================
    def _calculate_reward(self, obs_dict, actions):
        """
        Calculates the reward for this time step.

        ← ADITI: this is your main task for Week 6.
        For now a simple version is written below — you improve it in Week 6.

        Reward logic:
            + reward for using solar and wind (renewable energy)
            - penalty for buying from grid when price is high
            - penalty if battery SOC goes dangerously low
            - penalty for unmet demand (energy deficit)
        """
        reward = 0.0

        # ← ADITI: these are the basic rules — feel free to tune the numbers in Week 6
        reward += obs_dict['solar_output'] * 1.0        # +1.0 for each unit of solar used
        reward += obs_dict['wind_output']  * 0.8        # +0.8 for each unit of wind used

        grid_action = actions[3]   # Agent 3 = Grid Agent
        if grid_action == 1:       # Grid agent chose to BUY electricity
            if obs_dict['electricity_price'] > 0.7:    # Buying when price is high → bad
                reward -= 2.0
            else:                                       # Buying when price is low → okay
                reward -= 0.5

        if obs_dict['battery_soc'] < 0.15:             # Battery nearly empty → bad
            reward -= 3.0
        elif obs_dict['battery_soc'] > 0.85:           # Battery well charged → good
            reward += 0.5

        # ← ADITI: add more rules here in Week 6 based on demand vs supply balance

        return round(reward, 4)


    # ================================================================
    # Helper functions — simulation fallbacks when data not ready
    # ← ADITI: these run automatically when your CSVs are not loaded yet
    # ================================================================
    def _simulate_solar(self, t):
        """Sine curve — zero at night, peak at step 48 (noon)"""
        return float(max(0, np.sin(np.pi * t / 95)))

    def _simulate_wind(self):
        """Random wind between 10% and 70%"""
        return float(np.random.uniform(0.1, 0.7))

    def _simulate_price(self, t):
        """High price in morning (steps 28-40) and evening (steps 68-80)"""
        if 28 <= t <= 40 or 68 <= t <= 80:
            return float(np.random.uniform(0.7, 1.0))   # Peak hours
        return float(np.random.uniform(0.2, 0.5))        # Off-peak

    def _simulate_demand(self, t):
        """High demand in morning and evening, low at night"""
        if 24 <= t <= 44 or 64 <= t <= 84:
            return float(np.random.uniform(0.6, 1.0))   # Peak demand
        return float(np.random.uniform(0.2, 0.5))        # Low demand


    # ================================================================
    # _obs_to_array() — converts dict to numpy array for neural network
    # DO NOT TOUCH — Veeksha's networks expect exactly this format
    # ================================================================
    def _obs_to_array(self, obs_dict):
        return np.array([
            obs_dict['battery_soc'],
            obs_dict['solar_output'],
            obs_dict['wind_output'],
            obs_dict['electricity_price'],
            obs_dict['demand'],
            obs_dict['time_step'] / 95.0    # Normalise to [0, 1]
        ], dtype=np.float32)                # ← DO NOT TOUCH


# ================================================================
# QUICK TEST — Aditi runs this to check her file works
# Run: python smart_grid_env.py
# If it prints without errors, it is ready to send to Veeksha
# ================================================================
if __name__ == "__main__":
    print("Testing SmartGridEnv...")
    env = SmartGridEnv()

    # Test reset
    states, global_state = env.reset()
    print(f"✅ reset() works")
    print(f"   Number of states returned : {len(states)} (should be 5)")
    print(f"   Each state shape          : {states[0].shape} (should be (6,))")
    print(f"   Global state shape        : {global_state.shape} (should be (30,))")

    # Test step
    dummy_actions = [0, 0, 1, 0, 0]   # Battery charging, all others idle
    next_states, rewards, done, next_global_state = env.step(dummy_actions)
    print(f"\n✅ step() works")
    print(f"   Rewards returned          : {rewards} (should be list of 5 floats)")
    print(f"   Done flag                 : {done} (should be False on step 1)")

    # Run a full episode
    states, global_state = env.reset()
    total_reward = 0
    for step in range(96):
        actions = [0, 0, 1, 0, 0]
        next_states, rewards, done, next_global_state = env.step(actions)
        total_reward += rewards[0]
        if done:
            break

    print(f"\n✅ Full episode works — 96 steps completed")
    print(f"   Total episode reward      : {total_reward:.2f}")
    print(f"\n✅ File is ready to send to Veeksha!")