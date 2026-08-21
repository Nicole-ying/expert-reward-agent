def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Forward progress: keep iter_4's proven concave tanh saturation.
    # iter_5's slope 0.62 was closer to saturation at the operating speed and
    # reduced the forward gradient (forward ep_sum 587 -> 565), slowing the gait.
    # Hold the slope at 0.60; this term is not the material change this round.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Asymmetric backward penalty: moving backward costs more than forward
        # gains, discouraging in-place oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Stability and vertical terms held at iter_4's proven coefficients.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    # NEW component: mild torque-energy regularization.
    # No historical reward used `action`; the environment lists energy cost and
    # smooth hip/knee coordination as secondary objectives. A small quadratic
    # torque penalty suppresses wasted high-frequency control (the likely source
    # of iter_5's volatile score trend) without becoming a competing objective.
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