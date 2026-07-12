def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_velocity = obs[2]          # use current velocity for forward reward
    vertical_velocity = obs[3]            # new signal for bounce penalty

    # ----- 1. Forward progress signal -----
    # Encourage positive horizontal speed (same as before)
    forward_speed = max(0.0, horizontal_velocity)

    # ----- 2. Stability gate (multiplicative, replaces additive upright_penalty) -----
    # Gate approaches 1 when angle and angvel are near zero,
    # drops toward 0 when instability grows, choking off forward reward.
    # Using exponential for smooth, bounded [0,1] output.
    # Tuned so that modest instability (angle~0.2 rad, angvel~1 rad/s) still gives gate~0.92,
    # while severe instability (angle~1.0 rad) gives gate~0.37.
    angle_sq = hull_angle * hull_angle
    angvel_sq = hull_angular_velocity * hull_angular_velocity
    stability_gate = 2.718281828 ** (-1.5 * angle_sq - 0.15 * angvel_sq)

    # ----- 3. Vertical bounce penalty (new) -----
    # Penalize vertical kinetic energy to discourage hopping.
    bounce_penalty = -0.15 * (vertical_velocity * vertical_velocity)

    # ----- 4. Action energy cost (retained, fractionally reduced) -----
    torque_sq_sum = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_cost = -0.008 * torque_sq_sum   # slightly smaller to keep penalty burden ~0.5x of main signal

    # Combine: main reward is gated forward speed, then add penalties.
    forward_speed_reward = 2.0 * forward_speed * stability_gate
    total_reward = forward_speed_reward + bounce_penalty + action_cost

    components = {
        "forward_speed_reward": forward_speed_reward,
        "bounce_penalty": bounce_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components