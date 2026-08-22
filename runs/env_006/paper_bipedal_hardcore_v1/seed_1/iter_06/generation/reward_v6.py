def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle = next_obs[0]          # body pitch angle
    horizontal_speed = next_obs[2]   # forward velocity
    vertical_speed = next_obs[3]     # vertical velocity

    # 主进展信号：水平速度（保持每步有梯度）
    progress_raw = horizontal_speed

    # 身体倾角门控：倾角越接近安全上限（0.6 rad），门控越接近0
    angle_threshold = 0.6
    angle_gate = max(0.0, 1.0 - abs(hull_angle) / angle_threshold)

    # 垂直弹跳门控：垂直速度越小越好
    vert_threshold = 2.0
    vertical_gate = max(0.0, 1.0 - abs(vertical_speed) / vert_threshold)

    # 综合健康门控 —— 使用几何平均替代裸乘积，避免塌缩为0
    # 加微小 epsilon 防止数值异常
    health_gate = ((angle_gate * vertical_gate) + 1e-8) ** 0.5

    # 总奖励：在身体状态良好时充分奖励前进，恶化时平缓衰减
    total_reward = progress_raw * health_gate

    components = {
        'progress_raw': progress_raw,
        'angle_gate': angle_gate,
        'vertical_gate': vertical_gate,
        'health_gate': health_gate,
    }

    return float(total_reward), components