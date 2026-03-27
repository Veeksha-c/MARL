# Agentic AI Workshop - Major Project

## Overview
This repository contains implementations of reinforcement learning algorithms for my major project in the Agentic AI Workshop.

## Files

### week1_dqn_test.py
- **Purpose**: Deep Q-Network (DQN) implementation for CartPole environment
- **Framework**: PyTorch + Gymnasium
- **Features**:
  - Complete DQN agent with experience replay
  - Target network for stable learning
  - Epsilon-greedy exploration strategy
  - Training visualization with matplotlib
  - Detailed comments on every line

### marl_smart_grid.py
- **Purpose**: Multi-Agent Reinforcement Learning (MARL) system for smart grid management
- **Framework**: PyTorch
- **Architecture**: Actor-Critic with 5 agents
- **Features**:
  - 5 independent Actor networks (one per agent)
  - 1 centralized Critic network
  - Each agent observes 6 values: battery SOC, solar output, wind output, electricity price, demand, time step
  - Each agent has 4 possible actions
  - Model saving/loading functionality
  - Comprehensive test suite

## Getting Started

### Prerequisites
```bash
pip install torch gymnasium matplotlib numpy
```

### Running the Code

#### DQN CartPole
```bash
python week1_dqn_test.py
```

#### MARL Smart Grid
```bash
python marl_smart_grid.py
```

## Project Structure
```
├── week1_dqn_test.py      # DQN implementation
├── marl_smart_grid.py     # MARL smart grid system
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── venv/                  # Virtual environment (not tracked)
```

## Daily Commit Strategy
This project follows a daily commit workflow to track progress:

- **Each day**: Commit new features, bug fixes, or improvements
- **Commit messages**: Clear and descriptive of changes made
- **Branching**: Create new branches for major features

## Future Enhancements
- [ ] Add more complex environments
- [ ] Implement advanced MARL algorithms (MADDPG, QMIX)
- [ ] Add hyperparameter tuning
- [ ] Create comprehensive evaluation metrics
- [ ] Add visualization for MARL system

## Author
Veeksha - Agentic AI Workshop Major Project

## License
This project is for educational purposes as part of the Agentic AI Workshop.
