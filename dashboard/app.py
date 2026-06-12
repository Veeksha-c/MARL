# ============================================================
# app.py  —  FLASK BACKEND  (dashboard/ folder)
#
# Serves your trained MAPPO model + price/demand lookup data
# to the HTML/CSS/JS dashboard via a REST API.
#
# Run with:
#   pip install flask flask-cors
#   python dashboard/app.py
#
# Then open dashboard/index.html in your browser
# (or visit http://localhost:5000 if serving statically)
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Path setup ──────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # MARL/
sys.path.append(os.path.join(ROOT_DIR, 'training'))
sys.path.append(os.path.join(ROOT_DIR, 'agents'))

from inference import SmartGridPredictor, estimate_savings, AGENT_ORDER, ACTION_MEANINGS

# ── Flask app setup ─────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)   # Allow the HTML dashboard (opened as a file or different port) to call this API

# ── Load trained model once at startup ────────────────────────
print("Loading trained MAPPO model...")
predictor = SmartGridPredictor()
print("✅ Model loaded and ready.\n")

# ── Load price and demand lookup tables (96 slots, 15-min each) ─
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')

price_df  = pd.read_csv(os.path.join(DATA_DIR, 'price_data.csv'))
demand_df = pd.read_csv(os.path.join(DATA_DIR, 'demand_data.csv'))

PRICE_ARRAY  = price_df['price_normalized'].values
DEMAND_ARRAY = demand_df['demand_normalized'].values

print(f"✅ Loaded {len(PRICE_ARRAY)} price slots and {len(DEMAND_ARRAY)} demand slots\n")


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def serve_dashboard():
    """Serve the dashboard HTML file."""
    return send_from_directory('.', 'index.html')


@app.route('/api/lookup', methods=['GET'])
def lookup_price_demand():
    """
    Given a time_step (0-95), returns the price and demand
    from the real datasets for that 15-min slot.

    Query param: time_step (int, 0-95)
    """
    time_step = int(request.args.get('time_step', 0))
    time_step = max(0, min(95, time_step))

    price  = float(PRICE_ARRAY[time_step % len(PRICE_ARRAY)])
    demand = float(DEMAND_ARRAY[time_step % len(DEMAND_ARRAY)])

    return jsonify({
        'time_step': time_step,
        'electricity_price': round(price, 4),
        'demand': round(demand, 4)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.

    Expects JSON body:
    {
        "battery_soc": 0.5,
        "solar_output": 0.8,
        "wind_output": 0.3,
        "electricity_price": 0.6,
        "demand": 0.7,
        "time_step": 48
    }

    Returns: agent decisions + estimated savings
    """
    data = request.get_json()

    observation = {
        'battery_soc':       float(data.get('battery_soc', 0.5)),
        'solar_output':      float(data.get('solar_output', 0.0)),
        'wind_output':       float(data.get('wind_output', 0.0)),
        'electricity_price': float(data.get('electricity_price', 0.5)),
        'demand':            float(data.get('demand', 0.5)),
        'time_step':         int(data.get('time_step', 0)),
    }

    # Clamp all values to [0,1] and time_step to [0,95]
    for key in ['battery_soc', 'solar_output', 'wind_output', 'electricity_price', 'demand']:
        observation[key] = float(np.clip(observation[key], 0.0, 1.0))
    observation['time_step'] = int(np.clip(observation['time_step'], 0, 95))

    # ── Run prediction ──────────────────────────────────────
    prediction = predictor.predict(observation, deterministic=True)
    savings    = estimate_savings(observation, prediction)

    # ── Build response ──────────────────────────────────────
    agents_response = {}
    for agent_name in AGENT_ORDER:
        r = prediction[agent_name]
        agents_response[agent_name] = {
            'action':     r['action'],
            'meaning':    r['meaning'],
            'confidence': r['confidence'],
            'all_probs':  r['all_probs']
        }

    return jsonify({
        'observation': observation,
        'agents':      agents_response,
        'savings':     savings
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Simple health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': True})


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("  Smart Grid Dashboard — Backend API")
    print("="*60)
    print("  Open http://localhost:5000 in your browser")
    print("="*60)
    app.run(debug=True, port=5000)