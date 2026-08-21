def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L1 micro-tune: keep iter_6's proven concave tanh saturation and the
    # energy_penalty that drove the 254 -> 299 improvement. Only scale the
    # forward amplitude 2.0 -> 2.05 (+2.5%) to nudge the mean score from
    # 299.43 over the 300 target, without re-opening the high-speed farming
    # channel (tanh still saturates at high speed).
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.05 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Asymmetric backward penalty: moving backward costs more than forward
        # gains, discouraging in-place oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Stability and vertical terms held at iter_4/iter_6's proven coefficients.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    # Keep iter_6's torque-energy regularization unchanged: it was the key
    # improvement this round (smooth gait, suppressed high-frequency control).
    energy_penalty = -0.02 * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    total_reward = (
        forward_progress
        + torso_stability
        + vertical_velocity_penalty
        + energy_penalty
    )

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components