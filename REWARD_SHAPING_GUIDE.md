# Reward Shaping with A* Distance Heuristic

## What Was Implemented

**Reward shaping** is now integrated into your Q-Learning training using **A*'s Manhattan distance heuristic**. This adds a guidance signal to the agent without forcing a specific path.

### The Key Change

Previously, rewards were:
```
reward = 100 (goal) or -10 (death) or -1 (step) or -0.5 (wall)
```

Now, rewards include **potential-based shaping**:
```
base_reward = ... (as above)
distance_improvement = manhattan_distance(prev_cell, goal) - manhattan_distance(next_cell, goal)
shaped_reward = base_reward + 0.1 * distance_improvement

Total reward = shaped_reward
```

### Why This Helps

- **Without shaping**: Agent learns purely from reaching goal (sparse reward). Takes ~1000+ episodes.
- **With shaping**: Agent gets continuous feedback for moving closer to goal. Learns faster.

Example:
- Agent moves closer to goal: `distance_improvement = +2`, so `reward += 0.2`
- Agent moves away: `distance_improvement = -1`, so `reward -= 0.1`

This "nudges" the agent toward good directions without overriding its learning.

---

## Current Results (150 episodes)

After 150 episodes with shaping:
- ✅ Q-table learned 13,483 state-action pairs
- ✅ Epsilon decayed from 1.0 → 0.048 (exploration → exploitation)
- ❌ Still not reaching goal in evaluation
- ℹ️ Average reward: -2300 (mostly step penalties)

**Why still failing?**
- 64×64 maze is large (4096 cells)
- Manhattan distance alone is weak guidance
- Exploration still random early on

---

## How to Improve Convergence

### Option 1: Increase Shaping Weight (Fast)

In `train_qlearning.py`, line where `compute_reward()` is called:
```python
# Change from:
reward = compute_reward(..., shaping_weight=0.1)

# To:
reward = compute_reward(..., shaping_weight=0.5)  # Or 1.0
```

**Effect**: Distance improvements weighted more heavily
**Tradeoff**: Stronger guidance but may override actual learning

### Option 2: Run Longer Training

```bash
# Train for 500 episodes instead of 150
python -c "from train_qlearning import train_qlearning; train_qlearning(num_episodes=500, max_steps=2000)"
```

More episodes = more exploration = better Q-table coverage

### Option 3: Hybrid Approach (Best)

Combine **Curriculum Learning** with shaping:
- Episodes 0-100: High shaping (weight=0.8) + high epsilon
- Episodes 100-300: Medium shaping (weight=0.4) + medium epsilon  
- Episodes 300+: Low shaping (weight=0.1) + low epsilon

This gives the agent:
1. Strong guidance early (fast learning)
2. Gradual independence
3. Fine-tuning at the end

**Implementation**: Would need to modify `train_qlearning()` function to adjust `shaping_weight` per episode.

### Option 4: A*-Guided Initialization (Advanced)

Pre-populate Q-table with values from optimal A* paths:
```python
# Before training
agent.q_table[(state, optimal_action)] = 0.0  # Good actiongets positive Q
agent.q_table[(state, bad_action)] = -1.0     # Bad actions get negative Q
```

This "seeds" learning with correct behavior before exploration.

---

## Files Modified

1. **[train_qlearning.py](train_qlearning.py)**
   - Added `manhattan_distance()` function
   - Modified `compute_reward()` to accept positions and goal
   - Training loop tracks `prev_pos` for distance shaping
   - Default shaping_weight = 0.1

2. **[eval_qlearning.py](eval_qlearning.py)**
   - Added same distance shaping for consistent evaluation
   - Tracks episode deaths correctly

---

## Testing Commands

Run training with default shaping (weight=0.1):
```bash
python train_qlearning.py
```

Run using custom shaping weight:
```bash
python -c "from train_qlearning import train_qlearning; train_qlearning(num_episodes=500, max_steps=2000)"
```

Evaluate current policy:
```bash
python eval_qlearning.py
```

---

## Next Steps Recommendation

**I suggest**: Try Option 1 first (increase shaping weight to 0.5) + Option 2 (run 500 episodes).

```bash
# Edit train_qlearning.py line ~125:
reward = compute_reward(..., shaping_weight=0.5)  # Increase weight

# Then run training:
python train_qlearning.py  # 500 episodes, 2000 max_steps
```

This should give you:
- Better convergence (stronger guidance)
- Enough episodes to explore 64×64 state space
- Goal reaching within 200-300 episodes (estimated)

Let me know the results!
