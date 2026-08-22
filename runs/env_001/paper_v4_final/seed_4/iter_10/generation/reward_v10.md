# 设计理由
本轮修改 **failure_penalty** 组件（将其中的 `distant_stall` 移除，改为一个新的 **stall‑gate** 作用在 progress 上），并保留 out‑of‑bounds 与 crash 的一次性惩罚。  
- 当前 `failure_penalty` 的 `distant_stall` 条件要求 `cur_angle < 0.1`，太窄，导致 agent 在远离目标垫且倾角稍大时发生环境 `body_not_awake_or_settled` 而终止，完全不受惩罚（active_rate=0%）。  
- 新设计：当 `dist_next > 0.15`、`cur_speed < 0.2` 且双脚无接触时，认定 agent “停滞”，对 `progress` 施加 0.1 倍的门控，大幅削减去往目标的驱动力值，迫使 agent 保持移动。  
- `out_of_bounds` 和 `crash` 仍各自给予一次性的 -20。  
- 系数校准：主 progress 信号约 6/step，gate 生效后降至 0.6/step，形成的负向压力小于主信号量级，不会导致训练崩溃。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0
    SURVIVAL_PENALTY = -0.08
    FAIL_PENALTY = -20.0               # 出界 / 坠毁 一次性惩罚
    STALL_GATE = 0.1                   # 远离目标且停滞时 progress 乘以此值

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功着陆检测参数
    SUCCESS_DIST_THRESH = 0.15
    SUCCESS_SPEED_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1
    LANDING_SUCCESS_BONUS = 150.0

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号（带 stall-gate） ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    cur_speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5

    # 判断是否停滞在远离目标处（移除了角度限制）
    is_stall = (
        (dist_next > SUCCESS_DIST_THRESH) and
        (cur_speed < 0.2) and
        (left_contact < 0.5 and right_contact < 0.5)
    )
    gate = STALL_GATE if is_stall else 1.0

    progress = PROGRESS_WEIGHT * (dist_cur - dist_next) * gate

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 一次性成功着陆奖励 ----
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    prev_angle = abs(obs[4])
    prev_contact = max(obs[6], obs[7]) > 0.5

    cur_angle = abs(body_angle_next)
    cur_contact = max(left_contact, right_contact) > 0.5

    prev_success = (prev_dist < SUCCESS_DIST_THRESH and prev_speed < SUCCESS_SPEED_THRESH and
                    prev_angle < SUCCESS_ANGLE_THRESH and prev_contact)
    cur_success = (dist_next < SUCCESS_DIST_THRESH and cur_speed < SUCCESS_SPEED_THRESH and
                   cur_angle < SUCCESS_ANGLE_THRESH and cur_contact)

    landing_success_bonus = LANDING_SUCCESS_BONUS if (cur_success and not prev_success) else 0.0

    # ---- 6. 终止事件推断（仅出界与坠毁） ----
    out_of_bounds = abs(x_next) > X_BOUNDARY

    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 远离目标的停滞不再给一次性惩罚，已用 gate 处理
    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash) else 0.0

    # 合并奖励
    total_reward = (progress + survival + stability + fuel +
                    landing_success_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_success_bonus': landing_success_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号覆盖齐全，但 `distant_stall` 触发条件过紧（要求角度 < 0.1）导致僵尸组件；所有 episode 均以非成功 settled 终止却无惩罚，得分崩坏。
- **behavior**: agent 在远离目标垫处以一定倾角 settled（75 步终止），既不 crash 也不出界，从而避开了所有失败惩罚，仅靠微薄 progress 支撑但依然负分。
- **signal**: `distant_stall` 缺位，未能警告“非目标区域的静止”。
- **level**: Level 2
- **hypothesis**: 将静止检测放宽并与 progress 门控耦合，会强制 agent 持续移动向目标，消除当前安全 settled 的 exploit，使成功率与得分回升。
- **risk**: 在合理悬停阶段（接近目标垫时减速）可能被误伤，但 `dist_next <= 0.15` 时 gate 不生效，安全；高速飞越时不会触发，训练初期可能产生大量负向压力但小于主信号，可维持。