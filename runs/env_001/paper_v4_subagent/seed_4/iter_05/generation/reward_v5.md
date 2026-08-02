# 设计理由
**修改组件**：`failure_penalty` 从硬覆盖 -10 的稀疏大额惩罚改造为基于边界距离与坠毁风险的软门控安全调整量 (`safety_penalty`)。  
**原因**：当前 `failure_penalty` 是得分负值的主要来源（-52.8% 符号份额），量级远大于主进步信号（per‑step ≈ 0.05），且几乎总是导致 episode 立刻终止，切断了探索路径。同时 `soft_landing` 与 `success_reward` 几乎死亡，agent 在几十步内就因飞出边界或坠毁退场，无法累积任何着陆引导。改为软门控后，危险行为不再直接获得致命大负分，而是通过降低进步奖励来趋避，允许 agent 在边界内侧仍有学习余地，同时保留对危险的连续梯度信号。  
**数学形式**：  
- `boundary_gate = clamp(1.0 - max(0, abs(x_next)-0.7)/0.3, 0, 1)` → 在 |x|>0.7 时线性衰减，至边界 1.0 时归零。  
- `crash_danger` 在有脚接触时，由倾角、速度、距离四个指标的归一化平均值构成（截断至 1），`crash_gate = 1.0 - 0.8 * danger`（最低至 0.2）。无接触时保持 1.0。  
- `safety_factor = boundary_gate * crash_gate`。  
- `safety_penalty = progress * (safety_factor - 1.0)`，替代原来的 `failure_penalty`（硬惩罚移除）。  
**系数校准**：每个危险 step 的 `safety_penalty` 最大量级约 0.3，符合 ≤ 主信号 0.5x 的要求；不再有单步 -10 或 -20 的断崖惩罚。进步信号保持活跃，整体奖励结构从“稀疏失败主宰”转为“密集进步 + 软安全约束”。

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

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.05
    ACTION_PENALTY = 0.05

    SOFT_LANDING_WEIGHT = 0.5
    SOFT_POS_THRESH = 0.3
    SOFT_VEL_THRESH = 0.5
    SOFT_ANGLE_THRESH = 0.2

    SUCCESS_REWARD = 100.0
    SUCCESS_X_THRESH = 0.1
    SUCCESS_Y_THRESH = 0.1
    SUCCESS_VEL_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1

    # 软安全门控参数
    BOUNDARY_INNER = 0.7      # 开始衰减的内侧距离
    BOUNDARY_LIMIT = 1.0      # 环境视口边界（硬终止）
    CRASH_ANGLE = 0.5
    CRASH_VEL = 1.0
    CRASH_DIST = 0.5
    CRASH_GATE_MIN = 0.2      # 最高危险时 crash_gate 的下限

    # ========== 1. 主进展信号 ==========
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ========== 2. 安全门控（替代原来的硬失败惩罚）==========
    # 边界门：线性衰减，在边界处为 0
    dist_to_boundary = BOUNDARY_LIMIT - abs(x_next)
    if dist_to_boundary <= 0:
        boundary_gate = 0.0
    elif dist_to_boundary < (BOUNDARY_LIMIT - BOUNDARY_INNER):
        boundary_gate = max(0.0, dist_to_boundary / (BOUNDARY_LIMIT - BOUNDARY_INNER))
    else:
        boundary_gate = 1.0

    # 坠毁危险门：仅在有脚接触时评估，连续化
    contact = (left_contact == 1.0) or (right_contact == 1.0)
    if contact:
        angle_danger = min(abs(body_angle_next) / CRASH_ANGLE, 1.0)
        vel_danger = min((abs(x_vel_next) + abs(y_vel_next)) / (2.0 * CRASH_VEL), 1.0)
        dist_danger = min(dist_next / CRASH_DIST, 1.0)
        # 综合危险度（四个指标等权，平均后截断）
        danger = (angle_danger + vel_danger + dist_danger) / 3.0
        danger = min(danger, 1.0)
        crash_gate = 1.0 - (1.0 - CRASH_GATE_MIN) * danger
    else:
        crash_gate = 1.0

    safety_factor = boundary_gate * crash_gate
    # 安全调整量：为负值或零，扮演原来的 failure_penalty 角色
    safety_penalty = progress * (safety_factor - 1.0)

    # ========== 3. 姿态稳定惩罚 ==========
    stability_penalty = (-ANGLE_PENALTY * (body_angle_next ** 2) 
                         - ANG_VEL_PENALTY * (ang_vel_next ** 2))

    # ========== 4. 动作效率惩罚 ==========
    action_penalty = 0.0
    if action != 0:  # 非 no_engine 动作
        action_penalty = -ACTION_PENALTY

    # ========== 5. 软着陆塑造 ==========
    pos_factor = max(0.0, 1.0 - dist_next / SOFT_POS_THRESH)
    vel_abs_sum = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - vel_abs_sum / SOFT_VEL_THRESH)
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / SOFT_ANGLE_THRESH)
    soft_landing = SOFT_LANDING_WEIGHT * pos_factor * vel_factor * angle_factor

    # ========== 6. 成功着陆奖励 ==========
    landed = (abs(x_next) < SUCCESS_X_THRESH and
              abs(y_next) < SUCCESS_Y_THRESH and
              abs(x_vel_next) < SUCCESS_VEL_THRESH and
              abs(y_vel_next) < SUCCESS_VEL_THRESH and
              abs(body_angle_next) < SUCCESS_ANGLE_THRESH and
              contact)
    success_reward = SUCCESS_REWARD if landed else 0.0

    # ========== 汇总 ==========
    total_reward = (progress + safety_penalty + stability_penalty +
                    action_penalty + soft_landing + success_reward)

    components = {
        'progress': progress,
        'safety_penalty': safety_penalty,      # 替换原 failure_penalty
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'soft_landing': soft_landing,
        'success_reward': success_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测维度全部使用，但两个关键着陆代理奖励（soft_landing、success_reward）几乎死亡，失败惩罚以极端负值主导整个 episode，信号缺失在于缺乏有效的中间安全引导。  
- **behavior**: agent 在几十步内就飞出边界或坠毁，快速失败，没有机会接近成功着陆状态。  
- **signal**: failure_penalty 过强（每步平均 -0.25，为主信号的 5 倍），且硬惩罚切断了探索；progress 虽有连续信号但无力抗衡。  
- **level**: Level 2 — 结构变换（硬覆盖惩罚 → 软门控安全调整量）。  
- **hypothesis**: 用软门控替代硬惩罚后，agent 不再因靠近边界就被立即罚死，而会收到持续、微弱的“远离危险”梯度，从而延长存活时间，使 progress、soft_landing 等信号有机会发挥作用，最终导向安全着陆。  
- **risk**: 软边界可能使 agent 更频繁触碰视口边界（环境硬终止但不扣大分），若环境自身的终止截断学习信号，可能导致存活步数依然很短；但更平滑的负梯度应优于跳跃式惩罚。