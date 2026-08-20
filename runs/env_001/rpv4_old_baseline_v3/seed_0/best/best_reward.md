Now I have all the evidence. Let me conduct the systematic analysis.

---

## Component-by-Component Analysis

### (1) Score assessment

Score is **35.63** (target 200, gap 164). Not zero/negative, so Rule (1) is not triggered — but the score is **volatile**: cp50%=+43, cp70%=−60, cp100%=+62. The reward structure is unstable, not just undertuned.

### (2) `descent_reward` — DEAD, Rule (2) triggered

- ep_sum_mean: 2.97, magnitude_share: 0.9%, active_rate: 75.8%
- **Trend is completely flat**: 1.91 → 2.29 → 2.67 → 2.67 → 2.75 → 2.75 → 2.75 → 2.66 → 2.95 → 2.93

This component never changes across checkpoints. The formula `max(0, y_prev-y) / (1+speed)` produces near-identical values regardless of policy. RL cannot optimize it. **Must be removed and replaced.**

### (3) `contact_bonus` + `landing_quality` — DOMINANT & DESTABILIZING, Rule (3) triggered

- Combined magnitude_share: **83.2%** but active_rate only **5.5%**
- History confirms instability: iter-2 (w=0.5/2.0) had contact=58.5, landing=77.4 with stable behavior. Iter-9 (w=2.0/5.0) has contact=131, landing=143 but with **wild checkpoint swings**: contact surges to 386 at cp50% then crashes to 28 at cp70%.
- The agent discovers contact → gets massive reward spike → gradient is dominated by this sparse event → overfits → loses the skill → score collapses. This cycle repeats.

**Must scale back to iter-2 levels** (contact=0.5, landing=2.0).

### (4) `approach_reward`, `proximity`, `vel_cost`, `orientation_cost`, `fuel_cost` — STABLE but WEAK

- These five components are well-behaved (smooth trends, stable active rates) but collectively account for only 16.8% of magnitude. They're drowned out by the sparse contact/landing spikes.
- `approach_reward` at 4.18 is too weak to drive the agent toward the pad against the physics of hovering.

### Root Cause

The reward has a **structural sparsity-imbalance**: two sparse components (contact, landing) dominate 83% of the reward landscape, creating a "lottery" where the agent occasionally wins big but can't learn consistently. The continuous guidance signals (approach, proximity) are too weak to provide reliable gradients. The dead `descent_reward` provides no descent pressure.

### Solution

1. **Remove** dead `descent_reward`; replace with **`aligned_descent`** — rewards descending proportionally to horizontal alignment, creating a strong descent gradient when the agent is above the pad
2. **Boost** `approach` from 5.0→10.0 to provide stronger directional pull toward the pad
3. **Scale back** `contact_bonus` (2.0→0.5) and `landing_quality` (5.0→2.0) to iter-2 levels — these become genuine bonuses, not dominant spikes
4. **Keep** `proximity`, `vel_cost`, `orientation_cost`, `fuel_cost` at iter-2 proven weights

This closes the hover exploit: approach only rewards movement toward the pad, aligned_descent only rewards descending when aligned — an agent hovering in place gets zero from both and goes net-negative from speed/orientation/fuel costs.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Balanced lander reward: strong approach + aligned-descent gradients,
    moderate contact/landing bonuses (iter-2 weights), stable costs.
    Hover exploit closed: approach=0 when stationary, aligned_descent=0 when not descending.
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
    # 1. Strong potential-based approach — doubled from iter-2 (5.0→10.0)
    #    Only rewards movement toward pad; zero when hovering.
    # ------------------------------------------------------------------
    w_app = 10.0
    approach = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Aligned descent — replaces dead descent_reward
    #    Rewards downward motion, gated by horizontal alignment and speed.
    #    Strongest when: directly above pad (x≈0), moving down slowly.
    #    Near-zero when: far horizontally OR moving fast (plummeting).
    # ------------------------------------------------------------------
    descent_amount = max(0.0, y_prev - y)
    w_desc = 8.0
    aligned_descent = w_desc * descent_amount / (1.0 + abs(x) + speed)

    # ------------------------------------------------------------------
    # 3. Proximity — iter-2 weight (0.05), modest hover-prevention
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 4. Per-foot contact bonus — iter-2 weight (0.5), not dominant
    # ------------------------------------------------------------------
    w_contact = 0.5
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 5. Landing quality — iter-2 weight (2.0), not dominant
    # ------------------------------------------------------------------
    both_feet = left * right
    w_land = 2.0
    landing_quality = w_land * both_feet / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 6. Velocity cost — iter-2 weight (0.1)
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 7. Orientation cost — iter-2 weights
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 8. Fuel efficiency — iter-2 weight (0.03)
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach + aligned_descent + proximity + contact_bonus +
        landing_quality + vel_cost + orientation_cost + fuel_cost
    )

    components = {
        "approach": approach,
        "aligned_descent": aligned_descent,
        "proximity": proximity,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```