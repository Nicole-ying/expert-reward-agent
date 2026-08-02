# Response Record

# 设计理由
改动组件：`failure_penalty`。原有逻辑只检测显式的 `out_of_bounds` 和 `crash`，但 active_rate=0% 意味着这些条件几乎从未触发。然而在评估中 agent 有 50% 的 episode 不是成功着陆（landing_bonus 平均 100 而非 200），却在非成功终止时没有受到任何惩罚，说明它学会了在远离目标垫时提前悬停/静止来终止 episode、避开失败惩罚。这暴露了一个严重的信号缺口：**缺失对“非成功终止”的持续 discouragement**，导致 agent 可以通过停滞来 exploit 当前奖励结构。

本次 Level‑2 变换：将 `failure_penalty` 从二值终端事件改为**带 early‑stop 门控的连续惩罚**。新增 `early_stop` 条件：当到目标垫的欧氏距离 > 0.5 且合速度 < 0.3 时，给予随距离线性增长的惩罚（系数 0.1）。这样，agent 在远离目标时减速停滞会立即受到负反馈，必须继续接近目标。同时保留了原来的 `crash`/`out_of_bounds` 一次性的 -30 惩罚，作为硬覆盖（若未来触发）。该组件现在既捕捉“过早静止”也捕捉严重事故，closed gap。

系数校准：主信号 per‑step ≈ 0.23，`early_stop` 在触发时最大约 -0.05，符合 ≤0.3x 的惩罚负担限制。整体惩罚负担仍在可控范围。

# 诊断摘要
- **audit**: 信号缺失 — 未惩罚在非目标位置过早静止导致的终止，agent 利用该缺口逃避失败惩罚，致使成功率仅约 50%。
- **behavior**: agent 在 109 步终止，一半成功着陆，另一半在远处无惩罚静止终止，平均得分被拉低。
- **signal**: 缺失对“远离目标且速度过低”的持续 discouragement，使得 agent 没有动力在非成功条件下继续前进。
- **level**: Level 2
- **hypothesis**: 加入 early‑stop 惩罚后，agent 在远离目标时必须保持移动，不能在远处“赖着”终止，从而被迫前往目标垫，提高成功率并缩短时间。
- **risk**: 惩罚系数稍大可能导致 agent 在接近目标时过度加速、撞垫 crash，需观测 len 是否暴跌或 failure_penalty 突然激活。

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
    SUCCESS_BONUS = 200.0
    FAIL_PENALTY = -30.0               # 出界/坠毁一次性惩罚

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功阈值
    LAND_DIST_THRESHOLD = 0.2
    LAND_SPEED_THRESHOLD = 0.2
    LAND_ANGLE_THRESHOLD = 0.15
    LAND_CONTACT_REQUIRED = True

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # 新增：early-stop 惩罚阈值
    EARLY_STOP_DIST_THRESH = 0.5
    EARLY_STOP_SPEED_THRESH = 0.3
    EARLY_STOP_PENALTY_COEF = 0.1

    # ---- 1. 进展信号 ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 终止事件推断 ----
    # 成功着陆条件
    dist_to_pad = dist_next
    speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    angle_ok = abs(body_angle_next) < LAND_ANGLE_THRESHOLD
    contact_ok = (left_contact > 0.5) or (right_contact > 0.5)
    success = (dist_to_pad < LAND_DIST_THRESHOLD and
               speed < LAND_SPEED_THRESHOLD and
               angle_ok and
               contact_ok)

    # 出界
    out_of_bounds = abs(x_next) > X_BOUNDARY

    # 坠毁（原有逻辑）
    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 出界或坠毁的一次性惩罚
    terminal_fail_penalty = FAIL_PENALTY if (out_of_bounds or crash) else 0.0

    # 新增：过早停滞惩罚（连续 discouragement）
    early_stop = 0.0
    if dist_to_pad > EARLY_STOP_DIST_THRESH and speed < EARLY_STOP_SPEED_THRESH:
        early_stop = -EARLY_STOP_PENALTY_COEF * (dist_to_pad - EARLY_STOP_DIST_THRESH)

    # 最终失败惩罚由两部分组成
    failure_penalty = terminal_fail_penalty + early_stop

    # 成功奖励
    landing_bonus = SUCCESS_BONUS if success else 0.0

    # 合并
    total_reward = (progress + survival + stability + fuel +
                    landing_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_bonus': landing_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components
```
