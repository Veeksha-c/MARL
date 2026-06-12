"""
smart_grid_env.py  —  FINAL VERSION
SmartGridEnv: Gymnasium-compatible multi-agent smart grid environment.

This version uses 3 real datasets:
  Solar  : PVGIS-ERA5 Mysuru 2023 hourly  -> column 'P'      (solar_data.csv)
  Wind   : PVGIS-ERA5 Mysuru 2023 hourly  -> column 'WS10m'  (solar_data.csv)
  Demand : Grid India 15-min all-India SCADA data            (demand_data.csv)
  Price  : IEX DAM 2015-2022, 96-slot avg, Rs/kWh            (price_data.csv)

Contract format (from interface.md) - DO NOT change keys without telling Veeksha:
    observation = {
        'battery_soc':        float  0-1
        'solar_output':       float  0-1
        'wind_output':        float  0-1
        'electricity_price':  float  0-1
        'demand':             float  0-1
        'time_step':          int    0-95
    }
"""

import numpy as np
import pandas as pd
import math
import os
import gymnasium as gym
from gymnasium import spaces


# ---- Path helpers ------------------------------------------------------------
# smart_grid_env.py lives in:  MARL/environment/smart_grid_env.py
# datasets folder is at:       MARL/datasets/
_HERE     = os.path.dirname(os.path.abspath(__file__))   # MARL/environment/
_ROOT     = os.path.dirname(_HERE)                        # MARL/
_DATA_DIR = os.path.join(_ROOT, "datasets")               # MARL/datasets/

SOLAR_CSV  = os.path.join(_DATA_DIR, "solar_data.csv")    # PVGIS - solar + wind
DEMAND_CSV = os.path.join(_DATA_DIR, "demand_data.csv")   # Grid India - demand
PRICE_CSV  = os.path.join(_DATA_DIR, "price_data.csv")    # IEX - price


# ---- Data loading helpers ------------------------------------------------------

def _load_solar_and_wind(path: str):
    """
    Load PVGIS Mysuru CSV.
    File has 10 metadata rows at top, row 11 is the header row:
        time,P,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int

    We use:
      'P'     -> solar PV power output (W)   -> normalized 0-1 by max
      'WS10m' -> wind speed at 10m (m/s)      -> normalized 0-1 by /15.0

    Returns: (solar_array, wind_array) - both 1-D numpy arrays, ~8760 entries (hourly, 1 year)
    """
    df = pd.read_csv(path, skiprows=10)
    df.columns = df.columns.str.strip()

    # PVGIS adds footer text rows at the bottom - drop any row where P isn't numeric
    df["P"]     = pd.to_numeric(df["P"], errors="coerce")
    df["WS10m"] = pd.to_numeric(df["WS10m"], errors="coerce")
    df = df.dropna(subset=["P", "WS10m"])

    solar_raw = df["P"].astype(float).values
    wind_raw  = df["WS10m"].astype(float).values

    # Solar: normalize by max observed value
    max_solar = solar_raw.max() if solar_raw.max() > 0 else 1.0
    solar_norm = np.clip(solar_raw / max_solar, 0.0, 1.0)

    # Wind: normalize assuming 15 m/s is a realistic max useful wind speed
    wind_norm = np.clip(wind_raw / 15.0, 0.0, 1.0)

    return solar_norm, wind_norm


def _load_demand(path: str) -> np.ndarray:
    """
    Load Grid India 15-min demand CSV.
    Columns: time_step, time, demand_met_mw, demand_normalized
    Already 96 rows (one full day, 15-min slots) - use directly.
    Returns a 1-D numpy array of normalized (0-1) demand values, length 96.
    """
    df = pd.read_csv(path)
    return np.clip(df["demand_normalized"].astype(float).values, 0.0, 1.0)


def _load_price(path: str) -> np.ndarray:
    """
    Load IEX DAM price CSV (averaged 2015-2022, 96 slots).
    Columns: time_step, slot, avg_price_rs_kwh, price_normalized
    Returns a 1-D numpy array of normalized (0-1) price values, length 96.
    """
    df = pd.read_csv(path)
    return np.clip(df["price_normalized"].astype(float).values, 0.0, 1.0)


# ---- Fallback curves (used only if a CSV is missing) ----------------------------

def _sine_solar(time_step: int) -> float:
    """Fallback solar model: sine curve peaking at noon (time_step 48)."""
    hour = time_step / 4.0
    if hour < 6 or hour > 20:
        return 0.0
    return max(0.0, math.sin(math.pi * (hour - 6) / 14.0))


def _sine_price(time_step: int) -> float:
    """Fallback price model - morning (08:00) and evening (19:00) peaks."""
    hour = time_step / 4.0
    morning_peak = math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening_peak = math.exp(-0.5 * ((hour - 19) / 1.5) ** 2)
    base = 0.3
    return float(np.clip(base + 0.5 * morning_peak + 0.7 * evening_peak, 0.0, 1.0))


def _flat_demand(time_step: int) -> float:
    """Fallback demand model - simple daily curve with morning/evening peaks."""
    hour = time_step / 4.0
    base = 0.5
    morning = 0.3 * math.exp(-0.5 * ((hour - 8) / 2) ** 2)
    evening = 0.4 * math.exp(-0.5 * ((hour - 19) / 1.5) ** 2)
    return float(np.clip(base + morning + evening, 0.0, 1.0))


# ---- Main environment class ------------------------------------------------------

class SmartGridEnv(gym.Env):
    """
    Multi-agent smart grid environment.
    5 agents: solar_agent, wind_agent, battery_agent, grid_agent, load_agent.
    Each agent picks from 4 discrete actions (0-3).

    Action meanings (from interface.md):
        solar/wind:   0=idle, 1=store, 2=supply, 3=curtail
        battery:      0=idle, 1=charge, 2=discharge, 3=hold
        grid:         0=idle, 1=buy, 2=sell, 3=standby
        load:         0=normal, 1=reduce, 2=shift, 3=priority
    """

    metadata = {"render_modes": []}

    # ---- init ----------------------------------------------------------------

    def __init__(self, max_steps: int = 96):
        super().__init__()

        self.max_steps = max_steps          # 96 x 15-min = 1 full day

        # Battery parameters (physics-based model, see Section 4.5 of roadmap)
        self.battery_capacity  = 1.0        # normalised to 1.0
        self.charge_efficiency = 0.90       # 90% - 10% lost as heat
        self.max_charge_rate   = 0.05       # 0.5C over 1 step = 5% per step
        self.soc_min           = 0.10       # protect battery health
        self.soc_max           = 0.95

        # Load real data (fall back to curves if CSVs are missing)
        self._solar_data, self._wind_data = self._try_load_solar_wind()
        self._demand_data                 = self._try_load_demand()
        self._price_data                  = self._try_load_price()

        # Gymnasium spaces (Veeksha needs these to build her agents)
        self.observation_space = spaces.Dict({
            "battery_soc":       spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "solar_output":      spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "wind_output":       spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "electricity_price": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "demand":            spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "time_step":         spaces.Discrete(96),
        })

        # Action: dict of 5 agents, each choosing 0-3
        self.action_space = spaces.Dict({
            "solar_agent":   spaces.Discrete(4),
            "wind_agent":    spaces.Discrete(4),
            "battery_agent": spaces.Discrete(4),
            "grid_agent":    spaces.Discrete(4),
            "load_agent":    spaces.Discrete(4),
        })

        # Internal state (initialised properly in reset())
        self._battery_soc = 0.5
        self._time_step   = 0
        self._day_index   = 0       # which day of solar/wind data we're using

    # ---- data loading ----------------------------------------------------------

    def _try_load_solar_wind(self):
        if os.path.exists(SOLAR_CSV):
            try:
                solar, wind = _load_solar_and_wind(SOLAR_CSV)
                print(f"[SmartGridEnv] Loaded solar+wind data ({len(solar)} hourly rows)")
                return solar, wind
            except Exception as e:
                print(f"[SmartGridEnv] Warning: solar/wind CSV failed ({e}). Using fallback curves.")
        else:
            print(f"[SmartGridEnv] solar_data.csv not found at {SOLAR_CSV} - using fallback curves.")
        return None, None

    def _try_load_demand(self):
        if os.path.exists(DEMAND_CSV):
            try:
                data = _load_demand(DEMAND_CSV)
                print(f"[SmartGridEnv] Loaded demand data ({len(data)} slots)")
                return data
            except Exception as e:
                print(f"[SmartGridEnv] Warning: demand CSV failed ({e}). Using fallback curve.")
        else:
            print(f"[SmartGridEnv] demand_data.csv not found at {DEMAND_CSV} - using fallback curve.")
        return None

    def _try_load_price(self):
        if os.path.exists(PRICE_CSV):
            try:
                data = _load_price(PRICE_CSV)
                print(f"[SmartGridEnv] Loaded price data ({len(data)} slots)")
                return data
            except Exception as e:
                print(f"[SmartGridEnv] Warning: price CSV failed ({e}). Using fallback curve.")
        else:
            print(f"[SmartGridEnv] price_data.csv not found at {PRICE_CSV} - using fallback curve.")
        return None

    # ---- reset -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset environment to start of a new episode.
        Sets battery SOC to 0.5 and time_step to 0.
        Returns: (observation_dict, info_dict)
        """
        super().reset(seed=seed)

        self._battery_soc = 0.5
        self._time_step   = 0

        # Rotate through available days of solar/wind data so training sees variety
        if self._solar_data is not None:
            num_days = max(1, len(self._solar_data) // 24)
            self._day_index = (self._day_index + 1) % num_days
        else:
            self._day_index = (self._day_index + 1) % 365

        obs  = self._get_observation()
        info = {}
        return obs, info

    # ---- step ------------------------------------------------------------------

    def step(self, action: dict):
        """
        Advance environment by one 15-minute time step.

        Args:
            action: dict with keys matching interface.md
                {
                    'solar_agent':   int 0-3,
                    'wind_agent':    int 0-3,
                    'battery_agent': int 0-3,
                    'grid_agent':    int 0-3,
                    'load_agent':    int 0-3,
                }

        Returns:
            observation (dict), reward (float), terminated (bool),
            truncated (bool), info (dict)
        """

        # 1. Read current values
        solar_output      = self._get_solar()
        wind_output       = self._get_wind()
        electricity_price = self._get_price()
        demand            = self._get_demand()

        # 2. Apply battery agent action
        battery_action = action.get("battery_agent", 0)

        if battery_action == 1:   # charge: store energy from solar/wind
            charge_amount = min(
                self.max_charge_rate,
                self.soc_max - self._battery_soc    # don't overfill
            )
            self._battery_soc += charge_amount * self.charge_efficiency

        elif battery_action == 2: # discharge: supply energy to load
            discharge_amount = min(
                self.max_charge_rate,
                self._battery_soc - self.soc_min    # don't over-drain
            )
            self._battery_soc -= discharge_amount

        # 0 = idle, 3 = hold - no SOC change for either

        # Clamp SOC to safe limits (safety net for floating point edge cases)
        self._battery_soc = float(np.clip(self._battery_soc, self.soc_min, self.soc_max))

        # 3. Calculate reward
        reward = self._calculate_reward(
            solar_output, wind_output, electricity_price, demand, action
        )

        # 4. Advance time
        self._time_step += 1
        terminated = self._time_step >= self.max_steps
        truncated  = False   # we don't truncate mid-episode for now

        # 5. Build and return observation
        obs  = self._get_observation()
        info = {
            "solar_output":      solar_output,
            "wind_output":       wind_output,
            "electricity_price": electricity_price,
            "demand":            demand,
        }

        return obs, reward, terminated, truncated, info

    # ---- reward ------------------------------------------------------------------

    def _calculate_reward(self, solar_output, wind_output, electricity_price, demand, action) -> float:
        """
        Reward function:
          +1.0 x solar_output  if solar is being supplied (solar_agent = 2) and solar > 0
          +0.8 x wind_output   if wind is being supplied (wind_agent = 2) and wind > 0
          -2.0  if buying from grid (grid_agent = 1) when price > 0.7
          -0.5  if buying from grid when price is moderate/low (still a cost)
          -3.0  if battery SOC drops below 0.15 (critical low)
          +0.5  if battery SOC stays above 0.85 (well charged, good buffer)
        """
        reward = 0.0

        # Renewable usage bonuses
        if action.get("solar_agent") == 2 and solar_output > 0.0:
            reward += 1.0 * solar_output

        if action.get("wind_agent") == 2 and wind_output > 0.0:
            reward += 0.8 * wind_output

        # Grid purchase penalties
        grid_action = action.get("grid_agent", 0)
        if grid_action == 1:  # buying from grid
            if electricity_price > 0.7:
                reward -= 2.0
            else:
                reward -= 0.5

        # Battery health
        if self._battery_soc < 0.15:
            reward -= 3.0
        elif self._battery_soc > 0.85:
            reward += 0.5

        return float(reward)

    # ---- observation builder --------------------------------------------------

    def _get_observation(self) -> dict:
        """
        Returns observation in the exact contract format agreed with Veeksha.
        All floats clamped to [0, 1]. time_step is int 0-95.
        """
        return {
            "battery_soc":       float(np.clip(self._battery_soc, 0.0, 1.0)),
            "solar_output":      float(self._get_solar()),
            "wind_output":       float(self._get_wind()),
            "electricity_price": float(self._get_price()),
            "demand":            float(self._get_demand()),
            "time_step":         int(min(self._time_step, 95)),
        }

    # ---- per-step sensor helpers ------------------------------------------------

    def _get_solar(self) -> float:
        """
        Real PVGIS Mysuru solar data if available.
        Data is hourly - map 15-min time_step to the correct hour of the day.
        """
        if self._solar_data is not None:
            hour_index = (self._day_index * 24) + (self._time_step // 4)
            hour_index = hour_index % len(self._solar_data)
            return float(self._solar_data[hour_index])
        return _sine_solar(self._time_step)

    def _get_wind(self) -> float:
        """
        Real PVGIS Mysuru wind speed data if available.
        Same hourly mapping as solar.
        """
        if self._wind_data is not None:
            hour_index = (self._day_index * 24) + (self._time_step // 4)
            hour_index = hour_index % len(self._wind_data)
            return float(self._wind_data[hour_index])
        return float(np.random.uniform(0.0, 0.6))

    def _get_demand(self) -> float:
        """
        Real Grid India 15-min demand if available.
        File has exactly 96 rows - one full day at 15-min resolution.
        Indexed directly by time_step.
        """
        if self._demand_data is not None:
            idx = self._time_step % len(self._demand_data)
            return float(self._demand_data[idx])
        return _flat_demand(self._time_step)

    def _get_price(self) -> float:
        """
        Real IEX DAM price (averaged 2015-2022, 96 slots) if available.
        Indexed directly by time_step.
        """
        if self._price_data is not None:
            idx = self._time_step % len(self._price_data)
            return float(self._price_data[idx])
        return _sine_price(self._time_step)

    # ---- render (optional, not needed for training) -----------------------------

    def render(self):
        print(
            f"Step {self._time_step:02d} | "
            f"SOC={self._battery_soc:.2f} | "
            f"Solar={self._get_solar():.2f} | "
            f"Wind={self._get_wind():.2f} | "
            f"Price={self._get_price():.2f} | "
            f"Demand={self._get_demand():.2f}"
        )


# ---- Quick smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SmartGridEnv smoke test - FINAL VERSION (3 real datasets)")
    print("=" * 60)

    env = SmartGridEnv()

    obs, info = env.reset()
    print("\nAfter reset():")
    for k, v in obs.items():
        print(f"  {k}: {v}")

    print("\nRunning 5 steps with random actions...")
    for i in range(5):
        action = {
            "solar_agent":   env.action_space["solar_agent"].sample(),
            "wind_agent":    env.action_space["wind_agent"].sample(),
            "battery_agent": env.action_space["battery_agent"].sample(),
            "grid_agent":    env.action_space["grid_agent"].sample(),
            "load_agent":    env.action_space["load_agent"].sample(),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:+.2f}  SOC={obs['battery_soc']:.3f}  "
              f"solar={obs['solar_output']:.2f}  wind={obs['wind_output']:.2f}  "
              f"price={obs['electricity_price']:.2f}  demand={obs['demand']:.2f}  done={terminated}")

    print("\nRunning full episode (96 steps)...")
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(96):
        action = {k: env.action_space[k].sample() for k in env.action_space.spaces}
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated:
            break

    print(f"  Episode finished. Total reward: {total_reward:.2f}")
    print("\nAll tests passed - environment runs without crashing.")