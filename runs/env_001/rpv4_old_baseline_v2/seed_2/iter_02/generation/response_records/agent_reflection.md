# Response Record

## Reflection: Iteration 1 Analysis

### (1) Score assessment — **fundamentally broken, needs full rewrite**

Score = **-17.9**, all 20 episodes **truncated at 1000 steps** (0 terminated). The agent never achieves `body_not_awake_or_settled` (the success termination). The reward function fails at its only job: guiding the agent to land. A complete rewrite with a different main signal framework is required.

### (2) Component-by-component analysis

**position_proximity** (99.7% magnitude_share, 100% active): This is the root cause of failure. `1/(1+dist)` pays ~0.75/step for simply hovering near (0,0) at any altitude. The agent learns: "stay near the origin in the sky forever" — which is exactly what it does (1000-step truncations). Must be **replaced entirely** as the main signal.

**contact_completion** (0.0%, 0% active): Never fires because the agent never descends to the ground. The multiplicative gate `left_contact * right_contact` is always 0. This component is useless without a descent-driving signal. Must be paired with a stronger incentive and made into a one-time bonus.

**soft_landing_velocity** (~0.0% share): The activation gate `1/(1+10|y|)` is near-zero at the altitudes the agent occupies. Effectively invisible. Needs to be paired with descent incentives so the agent experiences it near the ground.

**stable_orientation** (0.2% share): Minor penalty, not harmful but not helpful. Agent maintains reasonable orientation naturally. Keep with adjustments.

### (3) Magnitude imbalance
position_proximity's 99.7% share completely drowns all other signals. The agent is single-mindedly optimizing proximity-at-any-altitude.

### Intervention Strategy
**Complete rewrite** with a new main signal: **descent incentive** (`y_prev - y > 0`). The new framework:
- **A: descent_reward** — reward moving downward each step (new primary driver)
- **B: altitude_proximity** — small saturating reward for being near y=0 (auxiliary gradient)
- **C: horizontal_centering** — gated by altitude so it only matters near ground
- **D+E: landing quality penalties** — velocity/attitude penalties gated by ground proximity
- **F: contact_bonus** — large one-time bonus for achieving dual-leg contact with quality gates
- **G: step_cost** — per-step penalty to punish hovering/truncation
- **H: engine_cost** — small penalty for engine use per task objectives

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous and current state
    x_prev = obs[0]
    y_prev = obs[1]

    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Descent Incentive (main learning signal) ----------
    # Reward each step of downward progress toward the platform.
    descent = y_prev - y
    descent_reward = 3.0 * max(0.0, descent)

    # ---------- B: Altitude Proximity (auxiliary gradient near ground) ----------
    # Small, saturating reward for simply being near platform level.
    alt_reward = 0.3 / (1.0 + 3.0 * abs(y))

    # ---------- C: Horizontal Centering (gated by altitude) ----------
    # Only reward x-alignment when the craft is close to the platform.
    near_ground = 1.0 / (1.0 + 5.0 * abs(y))
    center_reward = 0.3 * near_ground / (1.0 + 2.0 * abs(x))

    # ---------- D: Soft Landing Velocity Penalty (gated) ----------
    # Penalise high speed when close to ground, with emphasis on vertical.
    speed_penalty = -1.0 * near_ground * (0.5 * vx * vx + vy * vy)

    # ---------- E: Attitude Penalty (gated by altitude) ----------
    # Penalise tilt and rotation when close to the platform.
    attitude_penalty = -0.3 * near_ground * (angle * angle + 0.5 * angvel * angvel)

    # ---------- F: Contact Success Bonus (one-time landing reward) ----------
    # Large bonus when both legs touch, scaled by landing quality.
    contact = left_contact * right_contact
    landing_quality = 1.0 / (1.0 + 10.0 * (vx * vx + vy * vy) + 3.0 * angle * angle + 3.0 * abs(x))
    contact_bonus = 25.0 * contact * landing_quality

    # ---------- G: Step Cost (anti-hovering pressure) ----------
    step_cost = -0.03

    # ---------- H: Engine Efficiency ----------
    if action == 0:
        engine_cost = 0.0
    elif action == 2:
        engine_cost = -0.05
    else:
        engine_cost = -0.02

    total_reward = (
        descent_reward
        + alt_reward
        + center_reward
        + speed_penalty
        + attitude_penalty
        + contact_bonus
        + step_cost
        + engine_cost
    )

    components = {
        'descent_reward': descent_reward,
        'altitude_proximity': alt_reward,
        'horizontal_centering': center_reward,
        'speed_penalty': speed_penalty,
        'attitude_penalty': attitude_penalty,
        'contact_bonus': contact_bonus,
        'step_cost': step_cost,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components
```
