def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 核心进步信号：鼓励沿前进方向快速移动
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # 姿态稳定约束：惩罚躯干倾角偏离直立
    hull_angle = next_obs[0]
    w_angle = 1.0
    angle_penalty = -w_angle * (hull_angle ** 2)

    # 垂直稳定约束：抑制跳跃或剧烈起伏
    vertical_velocity = next_obs[3]
    w_vert = 0.5
    vert_penalty = -w_vert * (vertical_velocity ** 2)

    # 能耗约束：机械功率惩罚，力矩×关节角速度的绝对值之和
    # 更准确地反映实际能量消耗，区分推蹬发力与等长维持
    joint_velocities = [obs[5], obs[7], obs[10], obs[12]]  # hip1, knee1, hip2, knee2 角速度
    mechanical_power = sum(abs(action[i] * joint_velocities[i]) for i in range(4))
    w_energy = 0.08
    energy_penalty = -w_energy * mechanical_power

    total_reward = forward_reward + angle_penalty + vert_penalty + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'vert_penalty': vert_penalty,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components