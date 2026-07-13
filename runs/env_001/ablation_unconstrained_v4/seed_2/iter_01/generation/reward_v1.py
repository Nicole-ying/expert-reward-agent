def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    
    w_progress = 1.0
    w_vel = 0.5
    w_att = 0.5
    w_contact = 1.0
    d_close = 2.0
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2
    d_land = 1.0
    
    # 1. 向目标移动的稠密进展奖励 (improvement_delta)
    progress_reward = w_progress * (dist_old - dist_new)
    
    # 2. 接近目标时速度约束 (hinge penalty with gate)
    gate_speed = max(0.0, 1.0 - dist_new / d_close)
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * gate_speed * (excess_vx**2 + excess_vy**2)
    
    # 3. 接近目标时姿态稳定性约束 (hinge penalty with gate)
    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * gate_speed * (excess_angle**2 + excess_angvel**2)
    
    # 4. 软着陆接触奖励 (continuous closeness × contact average)
    closeness_land = max(0.0, 1.0 - dist_new / d_land)
    avg_contact = (left_contact + right_contact) / 2.0
    contact_bonus = w_contact * avg_contact * closeness_land
    
    total = progress_reward + vel_penalty + att_penalty + contact_bonus
    components = {
        'progress_reward': progress_reward,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty,
        'contact_bonus': contact_bonus
    }
    return float(total), components