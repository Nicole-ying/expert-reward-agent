def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ================== main progress：减少到目标垫的欧氏距离 ==================
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_old - dist_new                    # 距离缩小为正奖励
    w_progress = 1.0

    # ================== 姿态/角速度稳定约束（hinge 软惩罚）===================
    body_angle = abs(obs[4])
    ang_vel = abs(obs[5]) if abs(obs[5]) > 1e-6 else 0.0

    angle_penalty = max(0.0, body_angle - 0.3)        # 角度超过0.3 rad 才惩罚
    ang_vel_penalty = max(0.0, ang_vel - 1.0)         # 角速度超过1.0 rad/s 才惩罚

    w_angle = 0.05
    w_ang_vel = 0.02

    # ================== 登陆完成软代理（joint_condition_proxy）=================
    # 用 next_obs 判断着陆条件
    proximity = 1.0 / (1.0 + 5.0 * dist_new)           # 越近越接近1
    leg_contact = next_obs[6] * next_obs[7]            # 双腿都接触=1.0，否则0
    speed = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * speed)           # 低速接近1

    # 几何平均缓解乘积塌缩
    proxy = (proximity * leg_contact * speed_factor) ** (1.0/3.0)
    w_proxy = 0.05   # <--- 从 0.3 降低至 0.05

    # ================== 总奖励 ==================
    total_reward = (w_progress * progress
                    - w_angle * angle_penalty
                    - w_ang_vel * ang_vel_penalty
                    + w_proxy * proxy)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": -w_angle * angle_penalty,
        "ang_vel_penalty": -w_ang_vel * ang_vel_penalty,
        "landing_proxy": w_proxy * proxy
    }

    return float(total_reward), components