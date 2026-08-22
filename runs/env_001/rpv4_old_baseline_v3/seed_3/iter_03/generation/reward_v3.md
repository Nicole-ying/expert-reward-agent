Now I have complete evidence. Let me do the systematic analysis.

---

## Component-by-Component Analysis

### (1) `proximity_bonus` — FATAL LOCAL OPTIMUM (73.4% magnitude share)
- **Evidence**: Grows 16.3 → 204.6 across training. Always 100% active. Dominates reward.
- **Diagnosis**: `max(0, 1 - dist_n/4.0) * 0.3` ≈ 0.25/step when near pad. Over 1000 steps = ~250. Agent discovers: "hover near pad forever" beats any landing strategy. This is the #1 cause of failure.
- **Intervention**: **REMOVE entirely.** It creates a comfortable local optimum with no incentive to finish.

### (2) `contact_reward` — DEAD (0% at every checkpoint)
- **Evidence**: 0.0 across all 10 checkpoints. Never fires.
- **Diagnosis**: Agent never makes leg contact. The `slam_cost` (see below) teaches avoidance of all descent near ground — so the agent never reaches y≈0 where legs would touch.
- **Intervention**: Keep but boost coefficient. Will only fire once slam_cost is fixed and descent is encouraged.

### (3) `landing_reward` — DEAD (0% at every checkpoint)
- **Same root cause** as contact_reward. Product `lc_n * rc_n` always 0.
- **Intervention**: Keep structure but boost. Will activate once contact is achieved.

### (4) `slam_cost` — ACTIVELY HARMFUL (-20.4, active 70%)
- **Evidence**: Grows -1.7 → -20.6, active rate 5.7% → 70%. The agent gets better at avoiding it — by never descending.
- **Diagnosis**: `-4.0 * near_ground * downward_speed` penalizes **ALL** downward speed near ground, including the gentle descent needed for landing. Agent learns: "don't go down near the pad." This directly prevents landing.
- **Intervention**: Redesign as **threshold-based**: only penalize downward speed exceeding a safe limit (e.g., vy < -0.5). Gentle descent for landing must be free.

### (5) `progress` — NEGLIGIBLE (0.5% magnitude)
- **Evidence**: Converges to ~0.68. Near-zero once agent reaches pad vicinity.
- **Diagnosis**: Potential-based shaping self-extinguishes — correct behavior. But coefficient is too small to matter.
- **Intervention**: Boost coefficient to be the primary approach driver.

### (6) `fuel_cost` — working but significant (-17%)
- Stable around -47. Not broken but contributes to the negative score.
- **Intervention**: Reduce slightly.

### (7) `orient_cost` — working fine
- Stable at -5.3. Mild regularizer.
- **Intervention**: Keep as-is.

---

## Root Cause Summary

Two fatal design flaws reinforce each other:
1. **`proximity_bonus`** creates a local optimum: hovering near pad pays ~204 reward
2. **`slam_cost`** prevents the escape from that optimum: penalizing all downward speed teaches the agent to never descend to y≈0

Together: agent hovers safely above pad, collects proximity_bonus, avoids slam_cost, and never discovers leg contact.

## New Design Strategy

1. **Remove `proximity_bonus`** — the local optimum trap
2. **Fix `slam_cost`** — threshold at vy < -0.5 so gentle descent is free
3. **Add `descent_reward`** — explicitly reward descending when x-centered
4. **Boost `progress`** — make it the primary approach driver
5. **Boost `contact_reward` + `landing_bonus`** — make landing discovery worthwhile
6. **Add `step_cost`** — small per-step penalty creates pressure to terminate (land)
7. **Keep `orient_cost`, reduce `fuel_cost`** — mild regularization

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: Remove proximity_bonus (local optimum trap), fix slam_cost (threshold-based),
    add descent_reward, boost progress/contact/landing, add step_cost.
    
    Core insight: v2 failed because proximity_bonus let agent hover forever near pad
    collecting ~0.25/step, while slam_cost punished ALL downward motion — so agent
    never descended to make leg contact. This version removes the trap completely.
    """
    # --- unpack ---
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- 1. progress: potential-based, strong gradient toward (0,0) ---
    # Self-extinguishes at target — no local optimum
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 10.0 * (dist_c - dist_n)

    # --- 2. x_centering: extra pressure to align horizontally ---
    x_improvement = abs(x_c) - abs(x_n)
    x_bonus = 2.0 * x_improvement

    # --- 3. descent_reward: reward controlled descent when x is near center ---
    # This replaces the old proximity_bonus — it pushes agent DOWN onto the pad
    x_near = max(0.0, 1.0 - abs(x_n) / 2.0)
    descending = max(0.0, y_c - y_n)
    speed_ok = max(0.0, 1.0 - (vx_n ** 2 + vy_n ** 2) ** 0.5 / 1.5)
    descent_reward = 3.0 * x_near * descending * speed_ok

    # --- 4. contact_reward: substantial per-leg reward ---
    # Only fires when legs actually touch — discoverable once agent descends
    contact_reward = 8.0 * (lc_n + rc_n)

    # --- 5. landing_bonus: big bonus for both legs on pad near origin ---
    near_origin = max(0.0, 1.0 - (x_n ** 2 + y_n ** 2) ** 0.5 / 0.8)
    landing_bonus = 20.0 * lc_n * rc_n * near_origin

    # --- 6. slam_cost: ONLY penalize EXCESSIVE downward speed near ground ---
    # Safe descent (< 0.5 units/step downward) is NOT penalized
    # This is the critical fix — v2 penalized ALL downward speed
    near_ground = max(0.0, 1.0 - abs(y_n) / 2.0)
    excess_downward = max(0.0, -vy_n - 0.5)
    slam_cost = -10.0 * near_ground * excess_downward

    # --- 7. orient_cost: mild regularizer (kept from v2) ---
    orient_cost = -0.15 * abs(angle_n) - 0.03 * (angvel_n ** 2)

    # --- 8. step_cost: tiny per-step penalty to discourage infinite hovering ---
    step_cost = -0.03

    # --- 9. fuel_cost: reduced from v2 to avoid discouraging necessary engine use ---
    if action == 2:
        fuel_cost = -0.04
    elif action in (1, 3):
        fuel_cost = -0.02
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + x_bonus
        + descent_reward
        + contact_reward
        + landing_bonus
        + slam_cost
        + orient_cost
        + step_cost
        + fuel_cost
    )

    components = {
        'progress': progress,
        'x_bonus': x_bonus,
        'descent_reward': descent_reward,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'slam_cost': slam_cost,
        'orient_cost': orient_cost,
        'step_cost': step_cost,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components
```