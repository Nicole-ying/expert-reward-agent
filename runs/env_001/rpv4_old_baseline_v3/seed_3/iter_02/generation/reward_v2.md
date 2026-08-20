Now I have all the evidence. Let me do the systematic component analysis before writing code.

---

## Component Analysis

### 1. `contact_bonus` — **DEAD. This is the root cause of failure.**
- **Evidence**: active_rate = 0.0% from cp20% onward. Only cp10% saw 0.8% active (ep_sum=5.162, random luck), then vanished completely.
- **Why**: Thresholds are `y_th=0.1`, `x_th=0.2`. The agent must simultaneously be within 0.1 vertical AND 0.2 horizontal AND have both legs contacting. This almost never happens. The early cp10% contact was random exploration that the agent couldn't learn to reproduce.
- **Impact**: Without this signal, there is zero reward for landing. The net per-step reward averages to ~0 (goal_proximity +3.76 ≈ orientation_penalty -3.96). Agent wanders 1000 steps every episode, never terminates.
- **Action**: Must be completely redesigned with **much looser thresholds**, graduated via `training_progress`.

### 2. `goal_proximity` — Weak but functional
- **Evidence**: Stable ~3.8 across all checkpoints, 100% active. But only ~0.004 per step — barely above noise floor.
- **Diagnosis**: Delta-distance formulation is inherently bounded. It provides some gradient but far too weak to overcome orientation penalty or drive landing behavior.
- **Action**: Strengthen with larger coefficient. Add a continuous proximity bonus that doesn't depend on delta (absolute distance-based).

### 3. `orientation_penalty` — Functional, dominates by default
- **Evidence**: Stable ~-3.3 to -4.3, 46.5% magnitude share. Nearly perfectly cancels goal_proximity.
- **Diagnosis**: In the absence of any strong positive signal, this becomes the dominant force. The agent learns to stay upright but has no reason to land.
- **Action**: Keep as a regularizer but it will be dwarfed by the new landing reward.

### 4. `velocity_penalty` — Negligible
- **Evidence**: ~-0.5, only 5.7% share. The gated formulation makes it too weak even near the pad.
- **Action**: Replace with a targeted slam-prevention penalty (strong downward velocity penalty near ground) plus mild general damping.

### Root Cause
**The reward has no working positive landing signal.** The cp10% score of +38.8 (with contact_bonus=5.162) proves the agent CAN score positively when contact happens — but the thresholds are impossibly tight. All per-step components cancel to ~0, so the agent just survives 1000 steps.

### Strategy
1. **Landing reward**: Generous thresholds annealed via `training_progress`, strong per-step bonus
2. **Contact encouragement**: Reward any leg contact as an intermediate stepping stone
3. **Progress + proximity**: Provide gradient toward the pad
4. **Slam prevention**: Penalize downward velocity near ground
5. Keep orientation/fuel costs as mild regularizers

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Progressive landing reward with annealed thresholds + distance shaping + slam prevention
    
    Core insight: contact_bonus was dead because thresholds (y<0.1, x<0.2) were impossibly tight.
    This version uses generous initial thresholds that tighten via training_progress,
    plus intermediate rewards (any contact, proximity) to provide discoverable gradient.
    """
    # --- unpack ---
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- annealed landing thresholds ---
    # Early training: loose thresholds so agent can discover landing reward.
    # Late training: tighten for precision.
    tp = training_progress  # 0.0 → 1.0
    x_thresh = 0.6 - 0.35 * tp   # 0.60 → 0.25
    y_thresh = 0.4 - 0.25 * tp   # 0.40 → 0.15
    vel_thresh = 0.6 - 0.25 * tp # 0.60 → 0.35
    ang_thresh = 0.5 - 0.25 * tp # 0.50 → 0.25

    # --- 1. progress_reward: potential-based delta-distance ---
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 3.0 * (dist_c - dist_n)

    # --- 2. proximity_bonus: continuous reward for being near pad ---
    # Provides gradient even before any contact
    proximity = max(0.0, 1.0 - dist_n / 4.0)  # radius 4.0
    proximity_bonus = 0.3 * proximity

    # --- 3. contact_reward: encourage any leg contact ---
    # Intermediate stepping stone toward full landing
    contact_reward = 1.5 * (lc_n + rc_n)

    # --- 4. landing_reward: BIG bonus for full landing state ---
    # Product of soft conditions — only fires when ALL are satisfied
    x_ok = max(0.0, 1.0 - abs(x_n) / x_thresh)
    y_ok = max(0.0, 1.0 - abs(y_n) / y_thresh)
    speed = (vx_n ** 2 + vy_n ** 2) ** 0.5
    vel_ok = max(0.0, 1.0 - speed / vel_thresh)
    ang_ok = max(0.0, 1.0 - abs(angle_n) / ang_thresh)
    both_legs = lc_n * rc_n

    landing_quality = x_ok * y_ok * vel_ok * ang_ok
    landing_reward = 8.0 * landing_quality * both_legs

    # --- 5. orientation_cost: mild regularizer ---
    orient_cost = -0.15 * abs(angle_n) - 0.03 * (angvel_n ** 2)

    # --- 6. slam_cost: penalize high downward speed near ground ---
    near_ground = max(0.0, 1.0 - abs(y_n) / 2.0)
    downward_speed = max(0.0, -vy_n)
    slam_cost = -4.0 * near_ground * downward_speed

    # --- 7. fuel_cost: tiny efficiency incentive ---
    if action == 2:
        fuel_cost = -0.08
    elif action in (1, 3):
        fuel_cost = -0.04
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + proximity_bonus
        + contact_reward
        + landing_reward
        + orient_cost
        + slam_cost
        + fuel_cost
    )

    components = {
        'progress': progress,
        'proximity_bonus': proximity_bonus,
        'contact_reward': contact_reward,
        'landing_reward': landing_reward,
        'orient_cost': orient_cost,
        'slam_cost': slam_cost,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components
```