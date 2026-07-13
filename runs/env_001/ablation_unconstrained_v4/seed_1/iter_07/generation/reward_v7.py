def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # 距离进展（大幅提升权重，驱动向目标运动）
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    progress = prev_dist - next_dist
    c_progress = 10.0 * progress

    # 高度惩罚：鼓励接近目标垫高度 y=0
    c_altitude = -2.0 * abs(ny)

    # 腿部接触奖励（提高系数，降低触地尝试门槛）
    contact_count = n_left_contact + n_right_contact
    c_contact = 5.0 * contact_count

    # 姿态惩罚（保持稳定）
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.01 * (n_angvel**2)

    # 主引擎燃料惩罚：抑制无效悬停
    is_main_engine = (action == 2)
    c_fuel = -0.1 * float(is_main_engine)

    # 软着陆奖励（保留best结构，仅双腿同时接触时触发）
    two_legs = (n_left_contact > 0.5 and n_right_contact > 0.5)
    if two_legs:
        dist = next_dist
        speed_mag = (nvx**2 + nvy**2) ** 0.5
        angle_mag = abs(n_angle)
        angvel_mag = abs(n_angvel)
        score_dist = max(0.0, 1.0 - dist / 0.3)
        score_speed = max(0.0, 1.0 - speed_mag / 0.3)
        score_angle = max(0.0, 1.0 - angle_mag / 0.2)
        score_angvel = max(0.0, 1.0 - angvel_mag / 0.3)
        c_landing = 200.0 * score_dist * score_speed * score_angle * score_angvel
    else:
        c_landing = 0.0

    total_reward = c_progress + c_altitude + c_contact + c_angle + c_angvel + c_fuel + c_landing

    components = {
        'distance_progress': c_progress,
        'altitude_penalty': c_altitude,
        'contact_reward': c_contact,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
        'fuel_penalty': c_fuel,
        'landing_bonus': c_landing,
    }
    return (float(total_reward), components)