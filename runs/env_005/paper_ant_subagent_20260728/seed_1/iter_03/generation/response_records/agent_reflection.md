# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    body_x_velocity = obs[13]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: body_up_z = 1.0 - 2*(qx^2 + qy^2), range ~[-1, 1]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Health gate factors ----------
    # Height factor: healthy in (0.25, 0.95), smooth decay outside
    height_low_danger = max(0.0, (0.25 - body_z) / 0.10)
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    height_high_danger = max(0.0, (body_z - 0.95) / 0.10)
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # Upright factor: healthy when body_up_z > 0.6, linear decay below
    upright_factor = min(1.0, max(0.0, (body_up_z - 0.2) / 0.4))

    # Combined health gate: both must be satisfied
    health_gate = min(height_factor, upright_factor)

    # ---------- Reward components ----------
    # Forward velocity with health-modulated interpolation (base + bonus)
    # Allows negative reward for backward movement, provides non-zero gradient everywhere.
    forward_velocity = float(body_x_velocity)
    reward_forward = 1.0 * forward_velocity * (0.3 + 0.7 * health_gate)

    # Uprightness hinge penalty (active when severely tilted)
    upright_error = max(0.0, 0.6 - body_up_z)
    reward_upright_penalty = -0.5 * upright_error

    # Vertical velocity penalty (discourage bouncing/launching)
    vz_magnitude = abs(float(body_z_velocity))
    vz_excess = max(0.0, vz_magnitude - 1.5)
    reward_z_velocity_penalty = -0.2 * vz_excess

    # ---------- Total reward ----------
    total_reward = reward_forward + reward_upright_penalty + reward_z_velocity_penalty

    components = {
        "forward_gated": reward_forward,
        "health_gate": health_gate,
        "upright_penalty": reward_upright_penalty,
        "z_velocity_penalty": reward_z_velocity_penalty
    }

    return float(total_reward), components
```
