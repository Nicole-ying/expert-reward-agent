def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]

    # ---------- Derived signals ----------
    # Uprightness: 1.0 = perfectly upright, -1.0 = upside-down
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Height gate (tightly coupled to termination bounds) ----------
    # Healthy zone (0.30, 0.90) -> factor=1.0
    # Linear decay to 0.0 at termination boundaries (0.20, 1.00)
    height_low_danger = max(0.0, (0.30 - body_z) / 0.10)   # 0.30→0.20
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    height_high_danger = max(0.0, (body_z - 0.90) / 0.10)  # 0.90→1.00
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # ---------- Reward components ----------
    # Forward velocity — only available when height is safe
    forward_reward = float(body_x_velocity) * height_factor

    # Upright bonus — gentle positive gradient, no penalty
    upright_bonus = 0.1 * body_up_z

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_bonus

    components = {
        "forward": forward_reward,
        "upright_bonus": upright_bonus,
        "height_factor": height_factor,
    }

    return float(total_reward), components