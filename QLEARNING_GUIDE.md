# Q-Learning Training Guide

## Phase 1: Core Components ✅

### `learning_agent.py`
- **QLearningAgent class** with tabular Q-learning
- State encoding: `(row, col, fire_phase, confusion_bool)`
- Methods:
  - `get_action(state, epsilon)` — ε-greedy selection
  - `update_q(state, action, reward, next_state)` — Q-value update
  - `decay_epsilon()` — Exploration annealing
  - `save_q_table()` / `load_q_table()` — Persistence

### `train_qlearning.py`
**Main training entry point**

Runs training episodes with:
- Reward function: +100 (goal), -10 (death), -1 (step), -0.5 (wall)
- Epsilon decay: `0.995` per episode (1.0 → 0.01)
- Logging: Every 10 episodes (configurable)
- Checkpoints: Every 50 episodes (configurable)

**Usage:**
```bash
python train_qlearning.py
```

Default: 500 episodes, 1000 steps/episode

**Output:**
- `q_agent_final.pkl` — Trained model
- `checkpoints/q_agent_ep*.pkl` — Intermediate checkpoints

### `eval_qlearning.py`
**Evaluation script**

Loads a trained model and runs greedy policy (epsilon=0.0)

**Usage:**
```bash
python eval_qlearning.py
```

Default: 20 evaluation episodes (configurable)

**Output:**
- Per-episode stats (success/timeout, reward, steps, deaths)
- Summary: success rate, avg reward, deaths

---

## Next Steps

### After Training (~100+ episodes)
1. **Check success rate** with `eval_qlearning.py`
2. **Export model** for deployment: `q_agent_final.pkl`
3. **Inspect Q-table** for learned patterns

### Future Enhancements

**Reward Shaping:**
If learning is slow, add potential-based shaping:
```python
reward += 0.1 * (manhattan_distance(next_pos, goal) - manhattan_distance(pos, goal))
```

**Better State Representation:**
Current state is compact but may miss important features. Options:
- Add grid neighborhood observations (5×5 window around agent)
- Track number of deaths
- Include position history

**DQN Migration:**
If you want to:
- Train on multiple mazes simultaneously
- Use larger state spaces
- Improve generalization

Then move to a neural network with DQN.

**Policy Visualization:**
Create a heatmap showing learned value function across the maze:
```python
for state in agent.q_table:
    row, col, fire_phase, confusion = state
    max_q = max(agent.q_table.get((state, a), 0) for a in Action)
    # Plot max_q at (row, col)
```

---

## Troubleshooting

**Agent not learning (flat reward curve)?**
- Check if epsilon is decaying (should see in logs)
- Increase learning rate `alpha` (try 0.2)
- Reduce step penalty (try -0.1 instead of -1.0)

**Agent dies too much?**
- Reduce death penalty (try -5 instead of -10)
- Reduce step penalty to allow more exploration

**Training too slow?**
- Increase step penalty (encourages faster paths)
- Add potential-based reward shaping
- Use smaller `max_steps` limit (e.g., 500 instead of 1000)

---

## Files Structure

```
MazeBot/
├── learning_agent.py      ← Core Q-Learning agent
├── train_qlearning.py     ← Training loop
├── eval_qlearning.py      ← Evaluation script
├── play_qlearning.py      ← (Visualization wrapper - WIP)
├── q_agent_final.pkl      ← Trained model (after training)
└── checkpoints/           ← Intermediate checkpoints
    ├── q_agent_ep50.pkl
    ├── q_agent_ep100.pkl
    └── ...
```
