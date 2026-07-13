def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Hyperparameters
    w_progress = 2.0          # 靠近目标的权重
    w_x = 0.2                 # 横向偏离惩罚权重
    w_engine = 0.05           # 引擎点火惩罚权重
    landing_bonus = 20.0      # 成功着陆一次性奖励

    # 安全门控参数
    safe_vy_threshold = 0.5   # 垂直速度上限（绝对值），超过此值视为危险
    safe_height_threshold = 1.0   # 安全高度，低于此高度时激活坠落门控
    k_fall_gate = 10.0        # 坠落门控陡峭度
    k_angle_gate = 5.0        # 姿态门控陡峭度

    # 着陆判断阈值
    contact_thresh = 0.5      # 接触判断（0/1）
    vx_landing_max = 0.5
    vy_landing_max = 0.5
    angle_landing_max = 0.1

    # 计算与目标的距离
    dist_now = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 基础进度奖励：一步间靠近目标
    progress_reward = w_progress * (dist_now - dist_next)

    # 安全门控因子
    # 1) 坠落抑制：垂直速度向下(vy < 0)且离地很近时削弱
    vy = next_obs[3]
    height = next_obs[1]
    fall_danger = max(0.0, -(vy + safe_vy_threshold)) * max(0.0, safe_height_threshold - height)
    fall_gate = 1.0 / (1.0 + k_fall_gate * fall_danger)

    # 2) 姿态门控：倾角越大削弱越强
    angle_gate = 1.0 / (1.0 + k_angle_gate * abs(next_obs[4]))

    safety_gate = fall_gate * angle_gate

    # 着陆奖励：双腿触地且速度/角度都在安全范围内
    left_contact = next_obs[6] > contact_thresh
    right_contact = next_obs[7] > contact_thresh
    low_speed_vx = abs(next_obs[2]) < vx_landing_max
    low_speed_vy = abs(next_obs[3]) < vy_landing_max
    small_angle = abs(next_obs[4]) < angle_landing_max

    if left_contact and right_contact and low_speed_vx and low_speed_vy and small_angle:
        landing_reward = landing_bonus
    else:
        landing_reward = 0.0

    # 横向位置惩罚
    lateral_penalty = -w_x * abs(next_obs[0])

    # 引擎使用惩罚
    engine_penalty = -w_engine * (1.0 if action != 0 else 0.0)

    # 总奖励
    total_reward = progress_reward * safety_gate + landing_reward + lateral_penalty + engine_penalty

    components = {
        "progress_reward": progress_reward * safety_gate,
        "landing_reward": landing_reward,
        "lateral_penalty": lateral_penalty,
        "engine_penalty": engine_penalty
    }
    return float(total_reward), components