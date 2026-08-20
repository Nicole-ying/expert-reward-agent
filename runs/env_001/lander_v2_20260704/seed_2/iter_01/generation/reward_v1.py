def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]           # relative horizontal distance to platform center
    y_pos = next_obs[1]           # relative vertical distance to platform support surface
    x_vel = next_obs[2]           # horizontal velocity
    y_vel = next_obs[3]           # vertical velocity
    body_angle = next_obs[4]      # tilt angle
    angular_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]    # left support leg contact (0 or 1)
    right_contact = next_obs[7]   # right support leg contact (0 or 1)

    # 1. Primary learning signal: continuous negative Euclidean distance to goal
    #    Guides the agent to move towards the landing platform center.
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Stability penalty: light constraint to suppress excessive speed, tilt, and spin
    #    Helps smooth approach and safe landing, but weights are low enough not to freeze exploration.
    w_vx = 0.15
    w_vy = 0.05   # vertical velocity penalty smaller to allow descent
    w_angle = 0.2
    w_angvel = 0.2
    stability_penalty = -(
        w_vx * abs(x_vel) +
        w_vy * abs(y_vel) +
        w_angle * abs(body_angle) +
        w_angvel * abs(angular_vel)
    )

    # 3. Soft landing proxy: multi‑condition bonus that signals near‑perfect landing posture
    #    Only triggers when the agent is close to the platform, has low speed, is nearly upright,
    #    and both legs are in contact. This gives a clear hint for the desired final state.
    prox_thresh = 0.5
    speed_thresh_x = 0.2
    speed_thresh_y = 0.3
    angle_thresh = 0.1
    soft_landing_bonus = 0.0
    if (distance_to_target < prox_thresh and
        abs(x_vel) < speed_thresh_x and
        abs(y_vel) < speed_thresh_y and
        abs(body_angle) < angle_thresh and
        left_contact == 1.0 and right_contact == 1.0):
        soft_landing_bonus = 0.5

    total_reward = distance_reward + stability_penalty + soft_landing_bonus

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components