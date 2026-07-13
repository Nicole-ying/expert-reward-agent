def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ======== 观测解包 ========
    x_pos = obs[0]          # 水平偏移
    y_pos = obs[1]          # 垂直偏移
    x_vel = obs[2]          # 水平速度
    y_vel = obs[3]          # 垂直速度
    angle = obs[4]          # 姿态角
    ang_vel = obs[5]        # 角速度
    left_contact = obs[6]   # 左接触
    right_contact = obs[7]  # 右接触
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ======== 组件A: 主学习信号 - 接近目标 (approach_to_target) ========
    # 使用欧氏距离的负线性形式，提供每步梯度
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # 使用 improvement_delta: 距离减少即正奖励
    approach_reward = distance - next_distance  # 正表示靠近目标
    # 缩放至合理范围，避免被其他组件主导
    A = 5.0 * approach_reward

    # ======== 组件B: soft_health_gate - 姿态与速度健康门控 ========
    # 在接近目标时，如果姿态或速度不合适，衰减主奖励
    # 使用倒数衰减形式，当姿态角或速度过大时门值降低
    
    # 姿态因子: 角度越接近0越好
    angle_factor = 1.0 / (1.0 + 10.0 * abs(angle))
    
    # 速度因子: 总速度越接近0越好，但只在接近目标时激活
    total_speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * total_speed)
    
    # 门控因子: 乘积形式，两个条件都需要满足
    # 使用几何平均避免乘积塌缩
    health_gate = (angle_factor * speed_factor) ** 0.5
    
    # 只在飞行器接近目标地面时应用门控 (y_pos 较小)
    # 使用平滑过渡，避免硬阈值
    height_proximity = 1.0 / (1.0 + 10.0 * abs(y_pos))
    
    # 组合门控: 当接近地面时，health_gate生效；远离时gate=1.0
    effective_gate = height_proximity * health_gate + (1.0 - height_proximity) * 1.0
    
    # 将门控应用到主奖励上
    # 但保持正向 progress 不受完全抑制，避免"不敢动"
    B = A * (effective_gate - 1.0)  # 附加项：当health良好时≈0，恶化时惩罚主奖励
    B = min(B, 0.0)  # 只衰减，不增加

    # ======== 组件C: 速度惩罚 - 轻量约束 ========
    # 对高速运动给予温和惩罚，防止翻滚和撞击
    # 使用二次惩罚，但保持权重较小
    speed_penalty = -0.1 * total_speed ** 2
    
    # 角速度惩罚: 防止过快旋转
    ang_vel_penalty = -0.05 * ang_vel ** 2
    
    C = speed_penalty + ang_vel_penalty

    # ======== 组件D: 姿态约束 - 铰链形式 ========
    # 只在角度较大时惩罚，允许小幅摆动
    angle_abs = abs(angle)
    angle_threshold = 0.3  # 约17度，在此范围内不惩罚
    D = -0.2 * max(0.0, angle_abs - angle_threshold)

    # ======== 总奖励组装 ========
    total_reward = A + B + C + D

    components = {
        'approach_reward': A,
        'health_gate_modulation': B,
        'speed_penalty': C,
        'angle_hinge_penalty': D
    }

    return float(total_reward), components