def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 姿态门控：线性铰链，替换指数衰减
    # k=2.0, 在|angle|=0.5时归零，与best的exp(-10*0.5²)=0.082行为边界相近
    # 恒定梯度鼓励持续姿态修正，同时在中小角度区域比指数更宽松
    k_angle = 2.0
    posture_factor = max(0.0, 1.0 - k_angle * abs(torso_angle))

    # 主学习信号：前进速度 × 姿态因子
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：抑制躯干角速度
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty
    }

    return float(total_reward), components