# Response Record

Now I have all the data. Let me conduct the full component-by-component analysis.

---

## Component Analysis

### 1. `approach_reward` — **Dead signal (flatlined)**
- ep_sum_mean=5.95, magnitude_share=3.3%, active=100%
- Checkpoint trend: flat at 4-6 across all 10 checkpoints — zero learning progression
- **Root cause**: The coefficient (5.0) is too small. Over 850 steps the net progress is only ~1.2 distance units. The signal is always active but cancels itself out (positive when approaching, negative when retreating) and is drowned by other components.
- **Verdict**: Scale up massively — this should be the primary directional signal.

### 2. `proximity` — **Too shallow a gradient**
- ep_sum_mean=32.95, magnitude_share=15.4%, active=100%
- The `0.05/(1+dist)` form gives only 0.05 at the pad and 0.0025 at dist=19 — a tiny 20× range. There's almost no gradient to follow from far away.
- **Verdict**: Replace with exponential form `w * exp(-dist/scale)` for a 1000×+ gradient from start to pad.

### 3. `vel_cost` — **Undiscriminating, conflicts with approach**
- ep_sum_mean=-10.52, magnitude_share=4.9%, active=100%
- Penalizes all speed equally. The agent needs speed to descend and move horizontally. This creates a direct conflict with the approach goal.
- **Verdict**: Gate by proximity — penalize speed only near the pad, allow free movement far away.

### 4. `orientation_cost` — **Working correctly**
- ep_sum_mean=-2.62, magnitude_share=1.2%, active=100%
- Stable across all checkpoints (~-1.2 to -2.8). Not exploited, not dead.
- **Verdict**: Keep with slight relaxation to allow necessary tilting for horizontal control.

### 5. `contact_bonus` — **Sparse but useful**
- ep_sum_mean=58.50, magnitude_share=27.3%, active=8.4%
- Volatile: 34 at cp70% → **0 at cp80%** → 56 at cp100%. The agent finds then loses contact ability.
- **Verdict**: Keep moderate. The volatility is a symptom of weak continuous guidance, not this component's fault.

### 6. `landing_quality` — **Sparse-dominant (the core problem)**
- ep_sum_mean=77.37, magnitude_share=**36.1%** (largest), active=**5.4%** (rarest)
- Same collapse at cp80%: 52 at cp70% → **0 at cp80%** → 95 at cp100%
- **Verdict**: Together with contact_bonus, sparse rewards are 63.4% of magnitude but fire <10% of the time. This makes the reward function effectively sparse — the agent stumbles into landing, gets a reward spike, then forgets. The continuous signals must be strengthened so the agent can reliably reach this state.

### 7. `fuel_cost` — **Accumulates due to long episodes**
- ep_sum_mean=-25.11, magnitude_share=11.7%, active=98.4%
- Agent uses engines constantly because it never learns to land and terminate. Fix episodes, not this component.
- **Verdict**: Keep similar. Will naturally shrink when episodes shorten.

---

## Root Cause — Rule (1) applies

The reward function is **sparse in practice**: `landing_quality` + `contact_bonus` = 63.4% of magnitude but active <10% of steps. The continuous shaping signals (`approach_reward` 3.3%, `proximity` 15.4%) are too weak to reliably guide the agent to the pad. The agent occasionally stumbles into contact (cp50%, cp70%, cp100%) then loses the skill entirely (cp80%, cp90%). This is a fundamental framework problem — the continuous guidance must be rewritten as the dominant signal.

**Intervention**: Rewrite with exponential proximity as the primary continuous attractor, 4× stronger potential-based approach, proximity-gated velocity damping, and relaxed orientation.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: exponential proximity attraction as primary signal,
    strong potential-based approach shaping, proximity-gated velocity damping,
    relaxed orientation penalty, and moderate contact/landing bonuses.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 1. Exponential proximity: primary continuous attractor toward pad center
    # Range: ~1.5 at pad (dist=0), ~0.55 at dist=2.5, ~0.07 at dist=7.5, ~0.0005 at dist=20
    w_proximity = 1.5
    dist_scale = 2.5
    proximity_reward = w_proximity * (2.718281828 ** (-dist / dist_scale))

    # 2. Potential-based approach: strong directional shaping
    # Rewards every step that reduces distance to pad
    w_approach = 20.0
    approach_reward = w_approach * (dist_prev - dist)

    # 3. Proximity-gated velocity damping: penalise speed only near the pad
    # Far away: allow free movement for efficient descent. Near pad: enforce soft landing.
    near_pad = 2.718281828 ** (-dist / 1.5)
    w_vel_near = 1.0
    w_vel_far = 0.02
    vel_penalty = -(w_vel_near * near_pad + w_vel_far) * (abs(vx) + abs(vy))

    # 4. Orientation cost: relaxed to allow necessary tilting for horizontal control
    w_angle = 0.3
    w_angvel = 0.05
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 5. Per-foot contact bonus: moderate reward for ground contact
    w_contact = 0.5
    contact_bonus = w_contact * (left_contact + right_contact)

    # 6. Landing quality: bonus for stable two-foot contact
    both_feet = left_contact * right_contact
    w_landing = 2.0
    landing_quality = w_landing * both_feet / (1.0 + abs(angle) + speed)

    # 7. Fuel efficiency: light penalty for engine use
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity_reward + vel_penalty +
        orientation_cost + contact_bonus + landing_quality + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity_reward": proximity_reward,
        "vel_penalty": vel_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```
