def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact

    # 1. 主学习信号：距离改善 × 姿态门控（系数下调至5.0，防止rush-and-crash）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = 5.0 * progress_raw * gate       # 从25.0降至5.0

    # 2. 安全约束：角速度轻量惩罚
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆奖励：转移事件 —— 仅首次触地瞬间触发
    prev_both = obs[6] * obs[7]                       # 上一步双腿接触
    curr_both = next_obs[6] * next_obs[7]             # 当前步双腿接触
    landing_transition = max(0.0, curr_both - prev_both)  # 0→1 上升沿

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    velocity_gate = 1.0 / (1.0 + 2.0 * speed)         # 低速→1，高速→0
    angle_gate = 1.0 / (1.0 + 5.0 * angle_abs)        # 直立→1，倾斜→0

    landing_bonus = 100.0 * landing_transition * velocity_gate * angle_gate

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return (float(total_reward), components)