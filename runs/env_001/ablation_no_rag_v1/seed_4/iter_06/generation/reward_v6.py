def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact_now = next_obs[6]
    right_contact_now = next_obs[7]
    left_contact_prev = obs[6]
    right_contact_prev = obs[7]

    w_x = 0.05
    w_y = 0.05
    w_vx = 0.05
    w_vy = 0.05
    w_angle = 0.05
    w_omega = 0.05
    w_landing = 1.0
    w_engine = 0.001

    target_proximity = -w_x * abs(x) - w_y * abs(y)
    velocity_penalty = -w_vx * (vx ** 2) - w_vy * (vy ** 2)
    orientation_penalty = -w_angle * (angle ** 2)
    angvel_penalty = -w_omega * (omega ** 2)

    # 首次从非双脚接触变为双脚接触时给予一次性奖励
    prev_both_contact = (left_contact_prev * right_contact_prev) > 0.5
    now_both_contact = (left_contact_now * right_contact_now) > 0.5
    landing_reward = w_landing * (1.0 if (not prev_both_contact and now_both_contact) else 0.0)

    engine_penalty = -w_engine * (1.0 if action != 0 else 0.0)

    total_reward = target_proximity + velocity_penalty + orientation_penalty + angvel_penalty + landing_reward + engine_penalty

    components = {
        "target_proximity": target_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_reward": landing_reward,
        "engine_penalty": engine_penalty
    }

    return float(total_reward), components