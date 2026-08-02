# Response Record

# 设计理由
- **重建模式**：历史两轮使用相同骨架（progress + landing_proxy + stability_penalty + failure_penalty），虽然 score 从 -166 提升到 -6，但平均得分仍远低于目标 200，且 length 仍较长（455），说明 agent 在大量 episode 中缺乏高效、可靠的着陆策略。累积迭代记录未显示预判❌（因为仅两轮），但该骨架已连续两轮未突破，且本次反馈中 failure_penalty 触发率为 0%，暗示失败检测覆盖不全（可能有远处静止或软坠毁未被捕获）。因此切换骨架是合理的。
- **新骨架核心算子选择**：
  - 保留 `improvement_delta`（progress 距离减少）作为主进展信号，权重适度增加以鼓励更快接近。
  - 引入 `joint_condition_proxy` 构造**软着陆塑造**（soft_landing），用位置、速度、角度的连续乘积 factor 提供稠密的“接近成功”梯度，替代原先的弱 proximity_reward，使 agent 在学习稳定悬停/着陆时获得密集奖励。
  - 引入 `action_efficiency` 惩罚（离散动作中消耗燃料的动作），轻微约束不必要的引擎使用。
  - 扩展失败检测：在原有边界和 crash 基础上新增 `stall_detection`（远处低速静止），给予每步小惩罚，避免 agent 在远离目标处陷入停滞，覆盖原先 0% 触发率对应的缺失信号。
  - 仍保留稀疏 `success_reward`（100）用于奖励精确着陆，与 soft_landing 形成互补。
- **与已尝试路径的本质区别**：先前骨架依赖 progress + proximity（或 landing_proxy 内的简单接近奖励）+ 大额成功奖励，但缺乏中间密度引导（soft_landing 的连续乘积因子）和对远处停滞的负反馈。新骨架用 joint_condition_proxy 塑造着陆条件，利用乘积因子使 agent 在降速、回正角度的同时接近原点时获得递增奖励，填补“从接近到稳定”之间的梯度空白；同时增加 action 惩罚促进燃料节约。这有望显著提高成功率和平均 score。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========== 超参数 ==========
    PROGRESS_WEIGHT = 3.0
    FAIL_BOUNDS = -10.0
    FAIL_CRASH = -10.0
    FAIL_STALL_PER_STEP = -0.2

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.05
    ACTION_PENALTY = 0.05

    SOFT_LANDING_WEIGHT = 0.5
    SOFT_POS_THRESH = 0.3
    SOFT_VEL_THRESH = 0.5          # 绝对值之和阈值
    SOFT_ANGLE_THRESH = 0.2

    SUCCESS_REWARD = 100.0
    SUCCESS_X_THRESH = 0.1
    SUCCESS_Y_THRESH = 0.1
    SUCCESS_VEL_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1

    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.2
    CRASH_ANGLE = 0.8
    CRASH_VEL = 1.5
    STALL_DIST_THRESH = 0.5
    STALL_VEL_THRESH = 0.1

    # ========== 1. 主进展信号 (improvement_delta) ==========
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ========== 2. 失败惩罚（terminal_event 扩展：边界/坠毁/停滞）==========
    # 边界
    out_of_bounds = abs(x_next) > X_BOUNDARY
    boundary_penalty = FAIL_BOUNDS if out_of_bounds else 0.0

    # 坠毁（有脚接触 + 近地 + 姿态/速度异常）
    crash = False
    if (left_contact == 1.0 or right_contact == 1.0):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True
    crash_penalty = FAIL_CRASH if crash else 0.0

    # 远处低速停滞（补充缺失的失败信号）
    stall = False
    if (dist_next > STALL_DIST_THRESH and 
        abs(x_vel_next) < STALL_VEL_THRESH and 
        abs(y_vel_next) < STALL_VEL_THRESH):
        stall = True
    stall_penalty = FAIL_STALL_PER_STEP if stall else 0.0

    failure_penalty = boundary_penalty + crash_penalty + stall_penalty

    # ========== 3. 姿态稳定惩罚 ==========
    stability_penalty = (-ANGLE_PENALTY * (body_angle_next ** 2) 
                         - ANG_VEL_PENALTY * (ang_vel_next ** 2))

    # ========== 4. 动作效率惩罚（离散动作） ==========
    action_penalty = 0.0
    if action != 0:  # 非 no_engine 动作消耗燃料
        action_penalty = -ACTION_PENALTY

    # ========== 5. 软着陆塑造 (joint_condition_proxy, 连续) ==========
    # 位置因子
    pos_factor = max(0.0, 1.0 - dist_next / SOFT_POS_THRESH)
    # 速度因子
    vel_abs_sum = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - vel_abs_sum / SOFT_VEL_THRESH)
    # 角度因子
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / SOFT_ANGLE_THRESH)
    # 乘积（几何平均的简单形式）
    soft_landing = SOFT_LANDING_WEIGHT * pos_factor * vel_factor * angle_factor

    # ========== 6. 成功着陆奖励（稀疏大额） ==========
    landed = (abs(x_next) < SUCCESS_X_THRESH and
              abs(y_next) < SUCCESS_Y_THRESH and
              abs(x_vel_next) < SUCCESS_VEL_THRESH and
              abs(y_vel_next) < SUCCESS_VEL_THRESH and
              abs(body_angle_next) < SUCCESS_ANGLE_THRESH and
              (left_contact == 1.0 or right_contact == 1.0))
    success_reward = SUCCESS_REWARD if landed else 0.0

    # ========== 汇总 ==========
    total_reward = (progress + failure_penalty + stability_penalty +
                    action_penalty + soft_landing + success_reward)

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'soft_landing': soft_landing,
        'success_reward': success_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失信号：远处静止未被惩罚，导致 failure_penalty 触发率 0%；着陆引导从接近到稳定间梯度稀疏，用 joint_condition_proxy 弥补。
- **behavior**: agent 能偶尔成功着陆（score 上限 273）但平均仍低（-6），存在大量远处停滞或缓慢移动的低效 episode。
- **signal**: progress 主信号过弱（每步约 0.003），且缺乏慢速/静止的负向信号；硬着陆的前置条件梯度不足。
- **level**: Level 3（重建骨架）
- **hypothesis**: 加入软着陆塑造因子和停滞惩罚后，agent 将被连续奖励引导快速降速、回正姿态并接近原点，同时被惩罚驱离远处静止状态，平均成功率将显著提升，拉高平均 score。
- **risk**: stall_penalty 可能迫使 agent 在远处做无谓机动（浪费燃料）而非直飞目标；soft_landing 阈值若过于宽松，可能奖励在原点附近悬停但不真正着陆的行为，需后续迭代校准。
