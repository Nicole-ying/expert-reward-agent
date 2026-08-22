def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- 从 obs 中提取关键信号 -------------------
    body_z      = obs[0]      # 身体高度
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]     # 世界 x 方向前进速度
    body_y_vel  = obs[14]
    body_z_vel  = obs[15]
    roll_vel    = obs[16]
    pitch_vel   = obs[17]
    yaw_vel     = obs[18]

    # ------------------- 派生信号 -------------------
    # body_up_z: 身体在世界坐标系中的“上方向”分量，1 表示完全直立
    # 直接使用环境卡片提供的公式进行计算
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ------------------- 1. 主学习信号：前进速度 -------------------
    # 使用 linear dense_state_signal，每步都有梯度
    forward_reward = body_x_vel   # 正值鼓励前进，负值惩罚后退

    # ------------------- 2. 姿态约束：直立 -------------------
    # 二次惩罚 body_up_z 偏离 1 的程度（越小越好）
    # 在接近 1 时梯度很小，允许适当的身体摆动；倾斜越大惩罚越强
    upright_error = 1.0 - body_up_z
    upright_penalty = upright_error ** 2
    upright_reward = -upright_penalty

    # ------------------- 3. 高度安全约束：hinge penalty -------------------
    # 只在身体高度接近危险区（<0.3 或 >0.9）时施加惩罚
    # 安全区 (0.3~0.9) 内不惩罚，避免持续抑制正常的运动变化
    height_margin_low  = 0.3
    height_margin_high = 0.9
    below_margin = max(0.0, height_margin_low - body_z)
    above_margin = max(0.0, body_z - height_margin_high)
    height_penalty = below_margin + above_margin
    height_reward = -height_penalty

    # ------------------- 组合 -------------------
    w_forward = 1.0
    w_upright = 0.2
    w_height  = 5.0

    total_reward = (
        w_forward * forward_reward +
        w_upright * upright_reward +
        w_height  * height_reward
    )

    components = {
        'forward_reward': w_forward * forward_reward,
        'upright_reward': w_upright * upright_reward,
        'height_reward':  w_height  * height_reward
    }

    return float(total_reward), components