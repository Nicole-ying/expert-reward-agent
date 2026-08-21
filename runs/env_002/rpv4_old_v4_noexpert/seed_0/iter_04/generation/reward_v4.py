def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # L2: replace the linear speed reward (coef * min(v, 2.0)) with a concave
    # tanh saturation. Evidence: iter_2 raised the linear forward proxy from
    # 505 -> 555 ep_sum but the external score DROPPED 296 -> 237 with a shorter
    # episode (1173 -> 1050): the agent exploited raw high speed until falling.
    # iter_3's linear 0.8/1.2 over-corrected it (forward ep_sum 333, score 192).
    # tanh keeps moderate-speed reward close to the proven iter_1 balance while
    # giving diminishing returns at high speed, so the proxy can no longer be farmed.
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.6 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Keep iter_1's proven asymmetric backward penalty: moving backward is
        # punished harder than moving forward is rewarded, discouraging oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Hold stability and vertical terms at iter_1's proven coefficients: they were
    # not the exploit source, and iter_3 showed that over-strengthening stability
    # trades away forward progress.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
    }

    return float(total_reward), components