# Response Record

Now I have the complete picture. Let me do the systematic component analysis.

---

## Component Analysis

### (1) `proximity_reward` — **ROOT CAUSE: Absolute reward enables hovering trap**

| Evidence | Value |
|---|---|
| ep_sum_mean | 889.75 |
| magnitude_share | **84.7%** |
| active_rate | 100% |
| checkpoint trend | 184→263→648→**1177**→355→532→**1153**→700→823→880 |

Formula: `2.5 / (1.0 + next_dist)` — an **absolute** per-step reward, not potential-based.

**Diagnosis**: The agent gets ~2.5/step just by hovering near (0,0). It never needs to land. The wild oscillations (1177→355→1153) show the agent cycling between getting very close (then crashing, episode ends early, lower total) and recovering. This is the #1 problem — it creates a local optimum that dominates all other signals. Contrast with iter 7 where the same formula existed but speed_penalty was platform-only (giving free exploration), allowing the agent to discover landing (contact_reward active 6.8%).

**Verdict**: Must be converted to potential-based form (difference, not absolute) or removed entirely.

### (2) `contact_reward` — **DEAD. Never triggers.**

| Evidence | Value |
|---|---|
| ep_sum_mean | 0.000 |
| active_rate | **0.0%** across ALL checkpoints |
| iter 7 comparison | Same detection triggered 6.8% in iter 7 |

**Diagnosis**: The condition `nlc > 0.5 and nrc > 0.5 and next_dist < 0.5` is achievable (iter 7 proved it), but the always-active speed_penalty + time_penalty + proximity_reward combo in iter 9 creates an alternative "hover near platform" strategy that's easier and safer than attempting landing. The agent optimizes the wrong thing.

**Verdict**: Keep the contact signal but restructure the reward so landing is the only path to high return.

### (3) `touchdown_bonus` — **DEAD. Same root cause as contact.**

0.0% active across all checkpoints. One-time bonus can't fire if the agent never achieves the contact state.

### (4) `progress_reward` — **Negligible. Agent hovers.**

ep_sum_mean: -0.92 (essentially zero net movement). The agent neither approaches nor retreats — it hovers. The absolute proximity_reward pays regardless of movement, so progress_reward is irrelevant.

### (5) `speed_penalty` — **Always-active drag on exploration.**

ep_sum_mean: -67.35, always active. Punishes ALL movement, discouraging the speed needed to explore and discover the platform. In iter 7, this was platform-only (active 6.8%), which explains why iter 7 discovered landing and iter 9 didn't.

### (6) `time_penalty` — **Constant drain.**

-0.05/step, ep_sum: -44.56. Combined with speed_penalty, creates a "minimize everything" incentive that kills exploration.

### (7) `orientation_penalty` — **Reasonable. Keep.**

Stable across checkpoints, ~ -20 per episode. Does its job.

### (8) `fuel_cost` — **Mild. Keep.**

-26.73 per episode, reasonable magnitude.

---

## Intervention Plan

**Strategy**: Return to the iter 5/iter 7 architecture that proved landing was achievable, but fix the hovering trap:

1. **Kill absolute proximity**. Replace with potential-based progress: `reward = scale * (potential(next) - potential(current))` where `potential = 1/(1+dist)`. This gives zero for hovering, positive for approaching, negative for retreating.
2. **Landing reward dominates**. 50/step for both-legs contact, 5/step for one-leg contact (gradient toward full landing).
3. **Touchdown bonus**: 500 one-time transition bonus for first both-legs contact.
4. **Speed penalty: platform-only** (like iter 7). Free movement during approach, strong damping only when landed.
5. **Remove time_penalty** — it kills exploration.
6. **Keep orientation and fuel mild**.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangv = next_obs[5]
    nlc, nrc = next_obs[6], next_obs[7]
    olc, orc = obs[6], obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_speed = (nxv**2 + nyv**2) ** 0.5

    # === A. Progress: potential-based proximity (zero for hovering) ===
    # Potential Φ(s) = 1/(1+dist). Reward = γ * (Φ(next) - Φ(current))
    # Approaching: positive. Hovering: zero. Retreating: negative.
    potential = 1.0 / (1.0 + dist)
    next_potential = 1.0 / (1.0 + next_dist)
    progress_reward = 10.0 * (next_potential - potential)

    # === B. Contact: per-step reward for leg contact, no distance condition ===
    prev_both = (olc > 0.5 and orc > 0.5)
    curr_both = (nlc > 0.5 and nrc > 0.5)
    curr_one = ((nlc > 0.5) or (nrc > 0.5)) and not curr_both

    # Both legs on platform: massive per-step reward (the main driver)
    landing_reward = 50.0 if curr_both else (5.0 if curr_one else 0.0)

    # === C. Touchdown: one-time transition bonus for first full contact ===
    touchdown_bonus = 500.0 if (not prev_both and curr_both) else 0.0

    # === D. Speed penalty: platform-only (free movement during approach) ===
    on_platform = curr_both
    speed_penalty = -1.5 * next_speed if on_platform else 0.0

    # === E. Orientation: stay upright, always active but mild ===
    orientation_penalty = -0.5 * abs(nangle) - 0.15 * abs(nangv)

    # === F. Fuel cost: mild regularization ===
    fuel_cost = -0.03 if action != 0 else 0.0

    total = (progress_reward + landing_reward + touchdown_bonus +
             speed_penalty + orientation_penalty + fuel_cost)

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "touchdown_bonus": touchdown_bonus,
        "speed_penalty": speed_penalty,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
    }
    return float(total), components
```
