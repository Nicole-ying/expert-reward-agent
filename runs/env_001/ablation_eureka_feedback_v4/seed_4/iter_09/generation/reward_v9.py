def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    approach_delta = dist_curr - dist_next

    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    left_contact_curr = obs[6] > 0.5
    right_contact_curr = obs[7] > 0.5
    left_contact_next = next_obs[6] > 0.5
    right_contact_next = next_obs[7] > 0.5

    w_approach = 5.0
    w_vel_penalty = 0.5
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.1
    w_touchdown = 10.0

    approach_reward = w_approach * approach_delta

    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    prev_contact = left_contact_curr or right_contact_curr
    cur_contact = left_contact_next or right_contact_next
    touchdown_reward = 0.0
    if not prev_contact and cur_contact:
        speed_ratio = max(0.0, 1.0 - speed_next / safe_speed)
        touchdown_reward = w_touchdown * speed_ratio

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward + touchdown_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
        "touchdown_success": touchdown_reward,
    }
    return float(total_reward), components