def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact

    # 1. 主学习信号：距离改善 × 姿态门控（尺度修复：系数 25.0）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = 25.0 * progress_raw * gate     # 系数从1.0→25.0

    # 2. 安全约束：角速度轻量惩罚
    ang_vel         = next_obs[5]
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 3. 着陆质量奖励：低速+姿态良好才给显著奖励
    left  = next_obs[6]
    right = next_obs[7]
    both_contact = left * right                      # 仅当两腿同时接触时非零

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    velocity_gate = 1.0 / (1.0 + 2.0 * speed)        # 低速→1，高速→0

    angle_gate = 1.0 / (1.0 + 5.0 * angle_abs)       # 直立→1，倾斜→0

    landing_bonus = 10.0 * both_contact * velocity_gate * angle_gate

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return float(total_reward), components