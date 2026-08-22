# Response Record

Now I have the complete picture. Let me conduct the systematic analysis.

---

## Component-by-Component Analysis

### (1) Score assessment — Rule (1) partially triggered

Score is **−10.31**, hovering around zero. The agent survives the full 1000 steps (20/20 truncated, 0 terminated). It **never lands**. The root cause is structural: two components (`crash_penalty`, `main_decel_bonus`) added to iter-2 have backfired catastrophically, creating a "hover-near-pad" strategy instead of landing.

### (2) Stagnant/dead components

| Component | Status | Evidence |
|---|---|---|
| `crash_penalty` | **Dead** (0% active all checkpoints) | Never triggered — but its mere *existence* at weight 25.0 scares the agent away from ground contact entirely. |
| `contact_bonus` | **Dead at cp100%** (was alive cp10%-cp90%) | Agent *had* ground contact early (active 1.3-4.1%) but actively suppressed it. cp50% had 13.4 ep_sum → cp100% has 0. |
| `landing_quality` | **Dead at cp100%** (same trajectory) | Same regression: cp50%=17.4, cp80%=29.8, then vanished to 0 at cp100%. |
| `main_decel_bonus` | **Exploited** (23.4% magnitude, 53% active) | Steady throughout training. Agent learned: "fire main engine, slow descent, collect bonus, never touch ground." This is anti-landing. |

### (3) Magnitude shares

`proximity` at 37.9% is highest but not excessive. The real problem isn't magnitude — it's that `proximity` + `main_decel_bonus` together provide enough reward for hovering, while `crash_penalty` makes ground contact too risky. The agent found a local optimum: stay near the pad, use main engine ~half the time, never land.

### Root cause summary

The checkpoint trend tells the story: at cp50%, the agent was landing (contact_bonus=13.4, landing_quality=17.4, score=−0.5). By cp100%, it had learned to suppress ground contact entirely (both 0.0). The `crash_penalty` (weight 25.0) poisoned exploration — any high-speed ground contact during training rollouts taught the agent "ground = danger." The `main_decel_bonus` provided an alternative reward stream for hovering. Together they eliminated landing behavior.

### Fix strategy

1. **Remove `crash_penalty`** — it poisons exploration and the environment's own crash termination is sufficient
2. **Remove `main_decel_bonus`** — it rewards hovering, which is anti-landing
3. **Return to the iter-2 skeleton** (score +40, 6/20 landings) — it was proven to work
4. **Add a controlled-descent incentive** — gives direct gradient to go down, gated by speed to prevent plummeting
5. **Boost contact/landing weights** — make ground contact more attractive than iter-2

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Iter-2 skeleton restored (proven: score +40, 6/20 landings).
    + controlled-descent incentive (speed-gated to prevent plummeting)
    + boosted contact/landing weights
    - crash_penalty removed (poisons ground-contact exploration)
    - main_decel_bonus removed (rewards hovering, anti-landing)
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ------------------------------------------------------------------
    # 1. Potential-based approach — ITER 2 WEIGHT (proven safe)
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Proximity — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 3. Controlled-descent incentive (NEW — replaces harmful main_decel_bonus)
    #    Rewards descending (y decreasing), gated by speed so fast plummeting
    #    gives much less reward than slow, controlled descent.
    #    At speed=0: full weight; at speed=3: weight/4; at speed=10: weight/11.
    # ------------------------------------------------------------------
    descent = max(0.0, y_prev - y)
    w_desc = 3.0
    descent_reward = w_desc * descent / (1.0 + speed)

    # ------------------------------------------------------------------
    # 4. Velocity cost — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 5. Orientation cost — ITER 2 WEIGHTS
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 6. Per-foot contact bonus — BOOSTED from iter-2's 0.5 → 2.0
    # ------------------------------------------------------------------
    w_contact = 2.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 7. Landing quality — BOOSTED from iter-2's 2.0 → 5.0
    # ------------------------------------------------------------------
    both_feet = left * right
    w_land = 5.0
    landing_quality = w_land * both_feet / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 8. Fuel efficiency — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity + descent_reward + vel_cost +
        orientation_cost + contact_bonus + landing_quality + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "descent_reward": descent_reward,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```
