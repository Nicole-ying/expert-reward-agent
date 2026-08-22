def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]

    v7 design:
    - Restore v5's proven 30x Euclidean progress as the approach engine
    - Add descent_bonus for dense vertical progress (no hovering local optimum)
    - Replace sparse contact/landing with continuous soft_landing gradient
    - Drop fuel_cost entirely (v6's 63.7% magnitude killer)
    - Keep mild attitude penalty and crash guard
    """
    ox, oy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ── 1. Progress: Euclidean distance delta (v5's proven backbone) ──
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    progress_reward = 30.0 * (old_dist - new_dist)

    # ── 2. Descent bonus: reward downward movement ──
    # Dense per-step signal. Hovering or ascending gives zero — no exploit.
    dy = oy - ny
    descent_bonus = 5.0 * max(0.0, dy)

    # ── 3. Attitude penalty: keep it level ──
    attitude_penalty = 0.5 * (n_angle**2) + 0.1 * (n_av**2)

    # ── 4. Soft landing: continuous gradient for contact + low speed + upright ──
    # Uses multiplication: each factor (contact, speed, angle) gates the reward,
    # providing gradient even for partial fulfillment
    both_legs = n_left * n_right
    speed = (nvx**2 + nvy**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(n_angle))
    soft_landing = 25.0 * both_legs * speed_factor * angle_factor

    # ── 5. Crash guard: penalize high-speed ground contact ──
    any_contact = (n_left + n_right) > 0.0
    crash_penalty = speed if (any_contact and speed > 2.0) else 0.0

    total_reward = (
        progress_reward
        + descent_bonus
        - attitude_penalty
        + soft_landing
        - crash_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "descent_bonus": descent_bonus,
        "attitude_penalty": -attitude_penalty,
        "soft_landing": soft_landing,
        "crash_penalty": -crash_penalty,
    }

    return float(total_reward), components