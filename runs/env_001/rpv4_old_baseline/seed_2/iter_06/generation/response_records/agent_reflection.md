# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Shaped descent reward: encourage vy to match a target speed that decreases with height ----
    # Target downward speed should reduce as the lander gets closer to the platform (y approaching 0)
    # Using target_vy = -0.8 * max(y, 0.02) to avoid division‑by‑zero and keep target bounded
    height_for_target = max(y, 0.02)
    target_vy = -0.8 * height_for_target
    vy_error = abs(vy - target_vy)
    descent_shaping = -0.5 * vy_error

    # ---- Horizontal position penalty (keep x near 0) ----
    horiz_penalty = -0.2 * abs(x)

    # ---- Horizontal speed penalty (discourage lateral drift) ----
    vx_penalty = -0.1 * abs(vx)

    # ---- Orientation and angular velocity penalties ----
    orient_penalty = -0.1 * abs(body_angle)
    angvel_penalty = -0.05 * abs(angvel)

    # ---- Contact reward for touching feet ----
    contact_reward = (left_contact + right_contact) * 2.0

    # ---- Soft‑landing bonus (relaxed thresholds to make it more attainable) ----
    contact_ok = float(left_contact > 0.5 and right_contact > 0.5)
    speed_ok = float(abs(vx) < 0.5 and abs(vy) < 0.5 and abs(angvel) < 0.2)
    angle_ok = float(abs(body_angle) < 0.2)
    landing_bonus = 200.0 * contact_ok * speed_ok * angle_ok

    # ---- Small per‑step penalty to discourage lingering ----
    time_penalty = -0.01

    total = (descent_shaping + horiz_penalty + vx_penalty +
             orient_penalty + angvel_penalty + contact_reward +
             landing_bonus + time_penalty)

    components = {
        "descent_shaping": descent_shaping,
        "horiz_penalty": horiz_penalty,
        "vx_penalty": vx_penalty,
        "orient_penalty": orient_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "time_penalty": time_penalty
    }
    return total, components
```
