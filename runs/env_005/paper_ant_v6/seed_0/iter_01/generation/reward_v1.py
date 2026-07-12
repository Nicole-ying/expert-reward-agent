def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号
    body_z = next_obs[0]         # 身体高度
    quat_x = next_obs[2]         # 四元数虚部 x
    quat_y = next_obs[3]         # 四元数虚部 y
    quat_z = next_obs[4]         # 四元数虚部 z
    vx = next_obs[13]            # 世界系前向速度
    vy = next_obs[14]            # 世界系侧向速度

    # 1. 前向速度奖励（主学习信号）：只奖励非负的前向速度
    forward_reward = 1.0 * max(0.0, vx)

    # 2. 侧向速度惩罚：抑制横向漂移
    lateral_penalty = -0.5 * (vy ** 2)

    # 3. 身体高度安全约束：只在高度越出 [0.2, 1.0] 时进行线性惩罚
    height_low = 0.2
    height_high = 1.0
    height_exceed = max(0.0, height_low - body_z) + max(0.0, body_z - height_high)
    height_penalty = -10.0 * height_exceed

    # 4. 姿态直立约束：惩罚四元数虚部分量平方和，推动 quat_w 趋近 1
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    pose_penalty = -5.0 * pose_error

    # 总奖励
    total_reward = forward_reward + lateral_penalty + height_penalty + pose_penalty

    # 组件字典
    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "height_penalty": height_penalty,
        "posture_penalty": pose_penalty
    }

    return float(total_reward), components