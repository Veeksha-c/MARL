"""
smart_grid_env.py  —  ADITI'S FILE
SmartGridEnv: Gymnasium-compatible multi-agent smart grid environment.

Week 3 deliverable: skeleton that runs without crashing.
  - reset() loads Day 1 data, sets battery SOC to 0.5
  - step() reads real solar + demand + price from CSVs,
    random for wind (ERA5 not downloaded yet), updates battery SOC,
    returns observation dict in the contract format

Data sources used:
  Solar  : Karnataka VEDAS CSV  (Solar Radiation Watt/m2, normalized)
  Demand : CEA India CSV        (SR: DemandMet Southern Region, normalized)
  Price  : IEX DAM CSV          (96-slot avg price Rs/kWh across 2015-2022, normalized)
  Wind   : random 0-0.6         (replace with ERA5 in Week 2)

Contract format (from interface.md) — DO NOT change keys without telling Veeksha:
    observation = {
        'battery_soc':        float  0–1
        'solar_output':       float  0–1
        'wind_output':        float  0–1
        'electricity_price':  float  0–1
        'demand':             float  0–1
        'time_step':          int    0–95
    }
"""

import numpy as np
import pandas as pd
import math
import os
import gymnasium as gym
from gymnasium import spaces


# ── Path helpers ─────────────────────────────────────────────────────────────

# smart_grid_env.py lives in:  MARL/environment/smart_grid_env.py
# datasets folder is at:       MARL/datasets/
# Go one level up from environment/ to reach MARL/, then into datasets/
_HERE     = os.path.dirname(os.path.abspath(__file__))   # MARL/environment/
_ROOT     = os.path.dirname(_HERE)                        # MARL/
_DATA_DIR = os.path.join(_ROOT, "datasets")               # MARL/datasets/

SOLAR_CSV  = os.path.join(_DATA_DIR, "solar_rediation_tel_hr_karnataka_ka_1991_2020.csv")
DEMAND_CSV = os.path.join(_DATA_DIR, "demand energy.csv")
PRICE_CSV  = os.path.join(_DATA_DIR, "price_data.csv")


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_solar(path: str) -> np.ndarray:
    """
    Load Karnataka solar radiation CSV.
    Returns a 1-D numpy array of normalized (0–1) values, one entry per row.
    The raw column is 'Solar Radiation (Watt/m2)'; max observed ~ 1560 W/m².
    """
    df = pd.read_csv(path)
    raw = df["Solar Radiation (Watt/m2)"].fillna(0).values.astype(float)
    max_val = raw.max() if raw.max() > 0 else 1.0
    return np.clip(raw / max_val, 0.0, 1.0)


def _load_demand(path: str) -> np.ndarray:
    """
    Load CEA India demand CSV (demand energy.csv).
    Uses Southern Region (SR) DemandMet column — covers Karnataka.
    Returns a 1-D numpy array of normalized (0–1) daily demand values.
    """
    df = pd.read_csv(path)
    col = "SR: DemandMet"
    raw = df[col].ffill().values.astype(float)
    min_val, max_val = raw.min(), raw.max()
    if max_val == min_val:
        return np.zeros(len(raw))
    return np.clip((raw - min_val) / (max_val - min_val), 0.0, 1.0)


def _sine_solar(time_step: int) -> float:
    """
    Fallback solar model: smooth sine curve peaking at noon (time_step 48 = 12:00).
    Returns 0 at night, peaks at 1.0 around noon.
    time_step is 0–95 (96 × 15-min intervals per day).
    """
    hour = time_step / 4.0          # convert 15-min slot → hour (0–24)
    if hour < 6 or hour > 20:
        return 0.0
    return max(0.0, math.sin(math.pi * (hour - 6) / 14.0))


def _load_price(path: str) -> np.ndarray:
    """
    Load IEX DAM price_data.csv — 96 rows, one per 15-min slot.
    Column 'avg_price_rs_kwh' contains average Rs/kWh across 2015–2022.
    Normalizes to 0–1 using observed min/max in the dataset.
    Real price range in file: ~2.6 to ~4.64 Rs/kWh.
    """
    df = pd.read_csv(path)
    raw = df["avg_price_rs_kwh"].values.astype(float)
    min_val, max_val = raw.min(), raw.max()
    return np.clip((raw - min_val) / (max_val - min_val), 0.0, 1.0)


def _sine_price(time_step: int) -> float:
    """
    Fallback price model — used only if price_data.csv is missing.
    High during morning peak (08:00) and evening peak (19:00).
    """
    hour = time_step / 4.0
    morning_peak = math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening_peak = math.exp(-0.5 * ((hour - 19) / 1.5) ** 2)
    base = 0.3
    return float(np.clip(base + 0.5 * morning_peak + 0.7 * evening_peak, 0.0, 1.0))


# ── Main environment class ────────────────────────────────────────────────────

class SmartGridEnv(gym.Env):
    """
    Multi-agent smart grid environment.
    5 agents: solar_agent, wind_agent, battery_agent, grid_agent, load_agent.
    Each agent picks from 4 discrete actions (0–3).

    Action meanings (from interface.md):
        solar/wind:   0=idle, 1=store, 2=supply, 3=curtail
        battery:      0=idle, 1=charge, 2=discharge, 3=hold
        grid:         0=idle, 1=buy, 2=sell, 3=standby
        load:         0=normal, 1=reduce, 2=shift, 3=priority
    """

    metadata = {"render_modes": []}

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self, max_steps: int = 96):
        super().__init__()

        self.max_steps = max_steps          # 96 × 15-min = 1 full day

        # Battery parameters (physics-based model, see Section 4.5 of roadmap)
        self.battery_capacity  = 1.0        # normalised to 1.0
        self.charge_efficiency = 0.90       # 90% — 10% lost as heat
        self.max_charge_rate   = 0.05       # 0.5C over 1 step = 5% per step
        self.soc_min           = 0.10       # protect battery health
        self.soc_max           = 0.95

        # Load real data (fall back to curves if CSVs are missing)
        self._solar_data  = self._try_load_solar()
        self._demand_data = self._try_load_demand()
        self._price_data  = self._try_load_price()

        # Gymnasium spaces (Veeksha needs these to build her agents)
        # Observation: 6 continuous values all in [0, 1] except time_step
        self.observation_space = spaces.Dict({
            "battery_soc":       spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "solar_output":      spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "wind_output":       spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "electricity_price": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "demand":            spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "time_step":         spaces.Discrete(96),
        })

        # Action: dict of 5 agents, each choosing 0–3
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
        self._day_index   = 0       # which day's data we're using

    # ── data loading ─────────────────────────────────────────────────────────

    def _try_load_solar(self) -> np.ndarray | None:
        if os.path.exists(SOLAR_CSV):
            try:
                data = _load_solar(SOLAR_CSV)
                print(f"[SmartGridEnv] Loaded solar data ({len(data)} rows)")
                return data
            except Exception as e:
                print(f"[SmartGridEnv] Warning: solar CSV failed ({e}). Using sine fallback.")
        else:
            print(f"[SmartGridEnv] Solar CSV not found at {SOLAR_CSV} — using sine fallback.")
        return None

    def _try_load_demand(self) -> np.ndarray | None:
        if os.path.exists(DEMAND_CSV):
            try:
                data = _load_demand(DEMAND_CSV)
                print(f"[SmartGridEnv] Loaded demand data ({len(data)} rows)")
                return data
            except Exception as e:
                print(f"[SmartGridEnv] Warning: demand CSV failed ({e}). Using random fallback.")
        else:
            print(f"[SmartGridEnv] Demand CSV not found at {DEMAND_CSV} — using random fallback.")
        return None

    def _try_load_price(self) -> np.ndarray | None:
        if os.path.exists(PRICE_CSV):
            try:
                data = _load_price(PRICE_CSV)
                print(f"[SmartGridEnv] Loaded price data ({len(data)} slots)")
                return data
            except Exception as e:
                print(f"[SmartGridEnv] Warning: price CSV failed ({e}). Using sine fallback.")
        else:
            print(f"[SmartGridEnv] Price CSV not found at {PRICE_CSV} — using sine fallback.")
        return None

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset environment to start of a new episode.
        Sets battery SOC to 0.5 and time_step to 0.
        Returns: (observation_dict, info_dict)
        """
        super().reset(seed=seed)

        self._battery_soc = 0.5
        self._time_step   = 0

        # Rotate through available days so training sees varied data
        if self._demand_data is not None:
            self._day_index = (self._day_index + 1) % len(self._demand_data)

        obs = self._get_observation()
        info = {}
        return obs, info

    # ── step ─────────────────────────────────────────────────────────────────

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

        # ── 1. Read current values ─────────────────────────────────────────
        solar_output      = self._get_solar()
        wind_output       = self._get_wind()        # random until ERA5 arrives
        electricity_price = self._get_price()
        demand            = self._get_demand()

        # ── 2. Apply battery agent action ─────────────────────────────────
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

        # 0 = idle, 3 = hold — no SOC change for either

        # Clamp SOC to safe limits (safety net for floating point edge cases)
        self._battery_soc = float(np.clip(self._battery_soc, self.soc_min, self.soc_max))

        # ── 3. Calculate reward ────────────────────────────────────────────
        reward = self._calculate_reward(
            solar_output, electricity_price, action
        )

        # ── 4. Advance time ────────────────────────────────────────────────
        self._time_step += 1
        terminated = self._time_step >= self.max_steps
        truncated  = False   # we don't truncate mid-episode for now

        # ── 5. Build and return observation ───────────────────────────────
        obs  = self._get_observation()
        info = {
            "solar_output":      solar_output,
            "wind_output":       wind_output,
            "electricity_price": electricity_price,
            "demand":            demand,
        }

        return obs, reward, terminated, truncated, info

    # ── reward ───────────────────────────────────────────────────────────────

    def _calculate_reward(self, solar_output: float, electricity_price: float, action: dict) -> float:
        """
        Reward function (from Veeksha's spec):
          +1   if solar is being used (solar_agent action = 2 = 'supply')
          -2   if buying from grid (grid_agent = 1) when price > 0.7
          -3   if battery SOC drops below 0.1

        This is intentionally simple for Week 3.
        Aditi will expand it in Week 6 per roadmap.
        """
        reward = 0.0

        # Bonus for using solar power
        if action.get("solar_agent") == 2 and solar_output > 0.0:
            reward += 1.0

        # Penalty for buying grid power during expensive periods
        if action.get("grid_agent") == 1 and electricity_price > 0.7:
            reward -= 2.0

        # Penalty for running battery too low
        if self._battery_soc < 0.1:
            reward -= 3.0

        return float(reward)

    # ── observation builder ───────────────────────────────────────────────────

    def _get_observation(self) -> dict:
        """
        Returns observation in the exact contract format agreed with Veeksha.
        All floats clamped to [0, 1]. time_step is int 0–95.
        """
        return {
            "battery_soc":       float(np.clip(self._battery_soc, 0.0, 1.0)),
            "solar_output":      float(self._get_solar()),
            "wind_output":       float(self._get_wind()),
            "electricity_price": float(self._get_price()),
            "demand":            float(self._get_demand()),
            "time_step":         int(min(self._time_step, 95)),
        }

    # ── per-step sensor helpers ───────────────────────────────────────────────

    def _get_solar(self) -> float:
        """
        Real data if available (Karnataka VEDAS CSV), else sine curve fallback.
        Solar data has 46 rows so we cycle through them.
        """
        if self._solar_data is not None:
            # Map time_step (0–95) to a row in solar data
            idx = (self._day_index * self.max_steps + self._time_step) % len(self._solar_data)
            return float(self._solar_data[idx])
        return _sine_solar(self._time_step)

    def _get_wind(self) -> float:
        """
        ERA5 data not downloaded yet (Week 1 task).
        Using uniform random for now — will be replaced in Week 2.
        """
        return float(np.random.uniform(0.0, 0.6))

    def _get_demand(self) -> float:
        """
        Real demand from Robbie Andrew / CEA CSV if available.
        Data is daily so we hold it constant within a day and shift each episode.
        """
        if self._demand_data is not None:
            return float(self._demand_data[self._day_index % len(self._demand_data)])
        # Fallback: simulate a simple daily demand curve
        hour = self._time_step / 4.0
        base = 0.5
        morning = 0.3 * math.exp(-0.5 * ((hour - 8) / 2) ** 2)
        evening = 0.4 * math.exp(-0.5 * ((hour - 19) / 1.5) ** 2)
        return float(np.clip(base + morning + evening, 0.0, 1.0))

    def _get_price(self) -> float:
        """
        Real IEX DAM price for this 15-min slot if available.
        price_data.csv has 96 rows — one avg price per slot across 2015-2022.
        Normalized to 0-1 (raw range ~2.6 to ~4.64 Rs/kWh).
        Falls back to sine curve if CSV is missing.
        """
        if self._price_data is not None:
            return float(self._price_data[self._time_step % 96])
        return _sine_price(self._time_step)

    # ── render (optional, not needed for training) ────────────────────────────

    def render(self):
        print(
            f"Step {self._time_step:02d} | "
            f"SOC={self._battery_soc:.2f} | "
            f"Solar={self._get_solar():.2f} | "
            f"Price={self._get_price():.2f} | "
            f"Demand={self._get_demand():.2f}"
        )


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SmartGridEnv smoke test")
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
        print(f"  Step {i+1}: reward={reward:+.1f}  SOC={obs['battery_soc']:.3f}  "
              f"solar={obs['solar_output']:.2f}  price={obs['electricity_price']:.2f}  done={terminated}")

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
    print("\nAll tests passed — environment runs without crashing.")