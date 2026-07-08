def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取下一时刻的状态信息
    x = next_obs[0]   # 相对目标垫的水平位置
    y = next_obs[1]   # 相对目标垫的垂直位置（高度）
    vx = next_obs[2]  # 水平速度
    vy = next_obs[3]  # 垂直速度
    angle = next_obs[4]        # 机身倾角
    angular_vel = next_obs[5]  # 角速度

    # 主信号：鼓励接近目标垫
    distance = (x**2 + y**2)**0.5
    proximity_reward = -distance

    # 稳定约束：惩罚偏离竖立的倾角
    tilt_penalty = -0.5 * abs(angle)

    # 稳定约束：惩罚线速度大小，促使最终稳定
    speed = (vx**2 + vy**2)**0.5
    velocity_penalty = -0.1 * speed

    # 稳定约束：惩罚快速旋转
    rotation_penalty = -0.1 * abs(angular_vel)

    total_reward = proximity_reward + tilt_penalty + velocity_penalty + rotation_penalty

    components = {
        "proximity_reward": proximity_reward,
        "tilt_penalty": tilt_penalty,
        "velocity_penalty": velocity_penalty,
        "rotation_penalty": rotation_penalty
    }
    return total_reward, components