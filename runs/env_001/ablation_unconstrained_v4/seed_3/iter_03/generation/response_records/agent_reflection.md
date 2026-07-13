# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x = obs[0]          # horizontal position relative to pad
    y = obs[1]          # vertical position relative to pad
    vx = obs[2]         # horizontal velocity
    vy = obs[3]         # vertical velocity
    angle = obs[4]      # body angle (rad)
    # obs[5] angular velocity (unused, but available)
    left_contact = obs[6]
    right_contact = obs[7]

    # --- Component A: goal-approach ---
    distance = (x**2 + y**2) ** 0.5
    goal_reward = 1.0 / (1.0 + 5.0 * distance)

    # --- Component B: height incentive (stronger gradient near pad) ---
    height_reward = 1.0 / (1.0 + 5.0 * abs(y))

    # --- Component C: continuous contact reward ---
    # Give partial credit for either leg touching, scaled by soft landing quality
    any_contact = max(left_contact, right_contact)
    # Quality factors: low vertical speed, small angle
    vy_quality = 1.0 / (1.0 + 3.0 * abs(vy))
    angle_quality = 1.0 / (1.0 + 1.0 * abs(angle))
    contact_reward = any_contact * vy_quality * angle_quality * 0.5  # small per-step reward

    # --- Component D: landing success bonus ---
    # Strong incentive to bring both legs down with low speed and right attitude
    if left_contact and right_contact and abs(vy) < 0.3 and abs(angle) < 0.3:
        landing_bonus = 2.0
    else:
        landing_bonus = 0.0

    # --- Component E: engine usage penalty (fuel saving) ---
    engine_penalty = -0.1 if action != 0 else 0.0

    total_reward = goal_reward + height_reward + contact_reward + landing_bonus + engine_penalty

    components = {
        'goal_reward': goal_reward,
        'height_reward': height_reward,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'engine_penalty': engine_penalty
    }

    return float(total_reward), components
```
