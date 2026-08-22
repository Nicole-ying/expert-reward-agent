def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- extract next state signals -----
    body_z = next_obs[0]
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    body_z_vel = next_obs[15]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # ----- derived postural signals -----
    # body up direction: 1 = fully upright, 0 = horizontal, <0 = inverted
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))   # clamp to [0, 1] (inverted gives 0)

    # height health factor: 0 at lower bound 0.2, 1 at upper bound 1.0
    height_factor = (body_z - 0.2) / 0.8
    height_factor = max(0.0, min(1.0, height_factor))

    # upright factor: 0 at fully horizontal/inverted, 1 at fully upright
    upright_factor = body_up_z

    # ----- multiplicative progress reward with soft health gate -----
    # forward speed is multiplied by height and upright factors:
    #   - if body_z is low or robot is tilted, forward reward is scaled down
    #   - fully healthy posture enables the full forward speed reward
    # scaling constant controls the average per-step reward magnitude
    forward_reward = 3.0 * body_x_vel * height_factor * upright_factor

    # ----- additive stabilisation/constraint penalties -----
    # lateral drift penalty (suppress sideways motion)
    lateral_penalty = -1.0 * (body_y_vel ** 2)

    # angular velocity penalty (suppress excessive roll/pitch/yaw)
    angular_penalty = -0.2 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    # vertical velocity penalty (suppress bouncing, encourage smooth gait)
    vertical_penalty = -0.5 * (body_z_vel ** 2)

    # ----- total reward -----
    total_reward = forward_reward + lateral_penalty + angular_penalty + vertical_penalty

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty,
        'vertical_penalty': vertical_penalty,
    }

    return float(total_reward), components