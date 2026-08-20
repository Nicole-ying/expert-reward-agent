# Response Record

## Component-by-Component Analysis

### 1. Score fundamentally broken (–159, gap 359)
The score is deeply negative, all 20 episodes terminate (not truncate), and 3/20 end early with score < –50. The agent crashes or fails without ever landing. The current reward's only active positive signal (`progress_reward`) delivers just +0.1 per episode — far too weak to overcome whatever base penalty the environment imposes. **This is a root-cause failure requiring a full rewrite.**

### 2. Dead components: `contact_reward` and `vel_damping`
- **contact_reward**: 0.0 across all checkpoints, 0% active_rate. Requires `left_contact * right_contact` — both feet simultaneously. The agent never achieves this. The condition is too strict; we need per-foot rewards.
- **vel_damping**: 0.0 across all checkpoints. The gate `max(0, 1 – dist/1.0)` requires dist < 1.0, which the agent never reaches. The threshold is too tight relative to what the agent can achieve.

### 3. `progress_reward` dominates but is impotent
93.2% magnitude share, yet only +0.1 per episode. It's the only signal the agent follows. Checkpoint trend (–3.2 → +0.1) shows the agent *is* learning to move toward the pad, but the signal is too weak to close the deal. It needs a much stronger coefficient.

### 4. `orientation_penalty` is the only working component
Active 100%, improved from –2.57 → –0.03. The agent learns to stay upright. This mechanism is sound and should be retained (with adjusted weight).

### Intervention Strategy
1. **Stronger approach signal**: potential-based shaping with larger coefficient so moving toward the pad is significantly rewarded.
2. **Per-foot contact**: reward each foot individually, not just the AND of both.
3. **Landing quality bonus**: when both feet touch, reward stability (low speed + low angle) — this is the incentive to actually settle.
4. **Proximity term**: small continuous reward for simply being near the pad, so the agent doesn't drift away.
5. **Velocity cost**: always-active penalty on speed to encourage gentle descent.
6. **Fuel penalty**: light pressure to use engine sparingly, aligning with the task's secondary objective.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: potential-based approach shaping, per-foot contact,
    landing-quality bonus, orientation/velocity penalties, and fuel efficiency.
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

    # 1. Potential-based approach: reward every step that reduces distance to pad
    w_approach = 5.0
    approach_reward = w_approach * (dist_prev - dist)

    # 2. Proximity: small continuous reward for staying near the pad
    w_proximity = 0.05
    proximity = w_proximity / (1.0 + dist)

    # 3. Velocity cost: penalise speed at all times (gentle descent)
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # 4. Orientation cost: penalise tilt and angular velocity
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 5. Per-foot contact bonus: reward each foot touching ground individually
    w_contact = 0.5
    contact_bonus = w_contact * (left_contact + right_contact)

    # 6. Landing quality: bonus when both feet are down AND the lander is stable
    both_feet = left_contact * right_contact
    w_landing = 2.0
    landing_quality = w_landing * both_feet / (1.0 + abs(angle) + speed)

    # 7. Fuel efficiency: light pressure against unnecessary engine use
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity + vel_cost + orientation_cost +
        contact_bonus + landing_quality + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```
