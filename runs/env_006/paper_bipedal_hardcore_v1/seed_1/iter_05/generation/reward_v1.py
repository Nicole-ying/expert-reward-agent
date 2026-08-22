def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle = next_obs[0]          # body pitch angle
    horizontal_speed = next_obs[2]   # forward (assumed positive) velocity
    vertical_speed = next_obs[3]     # vertical velocity, bouncing indicator

    # 主进展信号：水平速度，鼓励持续前进（每步有梯度）
    progress_raw = horizontal_speed

    # 身体稳定性门控：倾角越接近安全上限（0.6 rad），门控值越接近0
    angle_threshold = 0.6
    angle_gate = max(0.0, 1.0 - abs(hull_angle) / angle_threshold)

    # 垂直弹跳门控：垂直速度越小越好，门控线性衰减
    vert_threshold = 2.0
    vertical_gate = max(0.0, 1.0 - abs(vertical_speed) / vert_threshold)

    # 综合健康门控，乘积作用于主奖励，避免硬惩罚
    health_gate = angle_gate * vertical_gate

    # 总奖励：在身体状态良好时充分奖励前进，恶化时自动衰减激励
    total_reward = progress_raw * health_gate

    # 组件记录，便于调试
    components = {
        'progress_raw': progress_raw,
        'angle_gate': angle_gate,
        'vertical_gate': vertical_gate,
        'health_gate': health_gate,
    }

    return float(total_reward), components