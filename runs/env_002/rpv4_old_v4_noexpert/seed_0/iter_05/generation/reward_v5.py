def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L1 micro-tune: keep iter_4's proven concave tanh saturation, raise the
    # moderate-speed slope 0.6 -> 0.62 while keeping the 2.0 ceiling unchanged.
    # This nudges forward reward up ~2-3% in the mid-speed range (the operating
    # point) to push the mean score from 298.56 over 300, without re-opening the
    # high-speed farming channel that crashed iter_2.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.62 * v) = (e^(1.24*v) - 1) / (e^(1.24*v) + 1)
        e2v = 2.718281828 ** (1.24 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Keep iter_1's proven asymmetric backward penalty: moving backward is
        # punished harder than moving forward is rewarded, discouraging oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Hold stability and vertical terms at iter_1/iter_4's proven coefficients:
    # they were not the exploit source, and changing them risks trading away
    # forward progress (as seen in iter_3) or destabilizing the gait.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
    }

    return float(total_reward), components