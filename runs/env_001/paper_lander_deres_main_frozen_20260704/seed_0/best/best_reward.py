def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # parameters
    w_align = 0.8
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    landing_bonus = 0.5

    # velocity alignment reward (main learning signal)
    # encourage velocity directed toward the goal at (0,0)
    # dot product: -(x * vx + y * vy) -> positive when moving toward origin
    dot = -(next_obs[0] * next_obs[2] + next_obs[1] * next_obs[3])
    # clamp to [-1,1] to avoid huge values, then only positive part
    dot_clamped = max(min(dot, 1.0), -1.0)
    alignment_reward = w_align * max(0.0, dot_clamped)

    # stability penalty (quadratic form, lighter on small deviations)
    vx, vy = next_obs[2], next_obs[3]
    vel_penalty = w_vel * (vx**2 + vy**2)
    angle_penalty = w_angle * (next_obs[4]**2)
    angvel_penalty = w_angvel * (next_obs[5]**2)
    stability_penalty = -(vel_penalty + angle_penalty + angvel_penalty)

    # soft landing proxy bonus (multi‑condition completion signal)
    dist_to_goal = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    speed_mag = (vx**2 + vy**2) ** 0.5
    near_target = dist_to_goal < 0.3
    low_speed = speed_mag < 0.2
    stable_angle = abs(next_obs[4]) < 0.05
    both_contact = (next_obs[6] == 1.0 and next_obs[7] == 1.0)
    soft_landing_proxy = landing_bonus if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = alignment_reward + stability_penalty + soft_landing_proxy

    components = {
        "velocity_alignment_reward": alignment_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components