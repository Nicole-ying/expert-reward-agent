# Response Record

# 设计理由

## 审计发现
本轮训练反馈显示 `failure_penalty` 的 `active_rate = 0.0%`，即从未触发。同时 `landing_bonus` 的 `episode_sum_mean = 100.0` 且 `active_rate = 0.5%`，说明约一半的 episode 成功着陆，另一半以某种失败方式终止但没有受到任何惩罚。

环境允许的终止条件包括：出界、与地面高速/大角度碰撞、以及**“远离目标时的静止”（body_not_awake_or_settled）**。现有失败检测只覆盖了前两种情况，遗漏了第三种。缺失的失败信号导致 agent 可以在远处缓慢停住、安全地结束 episode，既不受惩罚也不因存活时间获得正向奖励，最终拉低了平均得分。

## 修改内容
**组件**：`failure_penalty`  
**变换**：扩展失败类型，从仅判断 `out_of_bounds` 和 `crash` 改为同时捕获**远处“假成功”静止着陆**。  
**数学形式**：新增条件
- 非成功状态，
- 垂直高度已贴地（`y_next < GROUND_Y_CLOSE`），
- 到目标垫距离仍超出成功门限（`dist > LAND_DIST_THRESHOLD`），
- 速度极低、倾角正常且无支撑接触，
此时给出一次性 `FAIL_PENALTY = -30.0`。

这样设定的理由：
1. 这些特征恰好是 agent 降落到远处地面并触发 `body_not_awake_or_settled` 的典型末态。
2. 增加该惩罚后，agent 在远离目标时主动停止将受到严厉惩罚，从而被迫保持向目标运动，直至到达成功区域。
3. 惩罚量级与原有 failure_penalty 保持一致，不引入新的超参数。

## 系数校准
- 主进度奖励 per‑step ≈ `24.96 / 109.45 ≈ 0.228`。
- 存活惩罚 + 姿态惩罚 + 燃料惩罚 per‑step ≈ `-0.08 - 0.015 - 0.007 = -0.102`，约为主进度奖励的 0.45x，在安全限度内（≤0.5x）。
- 失败一次性惩罚仅作用于少数步，不影响逐步负担结构。

## 风险控制
- 可能误判：agent 在目标上方减速准备着陆时，`x` 距离可能暂时 > 0.2，但 `y` 尚未贴地。本设计中`y_next < 0.15` 的约束将这类空中减速排除在外，只在已经触地时生效。
- 保留原有 crash/out‑of‑bounds 逻辑，不改动其他组件。

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
    FAIL_PENALTY = -30.0               # 出界 / 坠毁 / 远距离静止 一次性惩罚

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

    # 高速/大角度碰撞（原有坠毁逻辑）
    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # ★ 新增：远离目标的静止着陆（脚未触垫，但已贴地且速度/倾角均正常，触发 body_not_awake_or_settled）
    distant_stall = (
        (not success) and
        (y_next < GROUND_Y_CLOSE) and
        (dist_next > LAND_DIST_THRESHOLD) and
        speed < LAND_SPEED_THRESHOLD and
        angle_ok and
        (not contact_ok)
    )

    # 任意一种失败即触发一次性惩罚
    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash or distant_stall) else 0.0

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

# 诊断摘要
- **audit**: 信号缺失 — 远离目标的静止终止未被惩罚，导致约半数 episode 在未受罚的情况下失败。
- **behavior**: agent 能成功着陆部分 episode，但在其余 episode 中利用“远距离静止”终止来规避惩罚，整体得分在 14 附近停滞。
- **signal**: 缺少对 `body_not_awake_or_settled` 发生在非目标区的失败检测，现有 `failure_penalty` 未覆盖该情况。
- **level**: Level 2
- **hypothesis**: 加入远离目标的静止失败惩罚后，agent 将不能再安全地结束于远处，必须努力到达目标垫，从而提升成功率和平均得分。
- **risk**: 正常下降过程中的短暂减速可能被误判，但 `y_next < 0.15` 的约束将其限制在已贴地时刻，误伤概率低；若仍有误伤，可将 `dist_next` 阈值上调至 0.3 或加入 `ang_vel` 补充条件。
