# Response Record

# 设计理由
本轮改动属于 **Level 2 结构变换**：移除 `early_stop` 连续惩罚组件。上一轮在奖励中加入“远离目标且低速”的连续惩罚，意图迫使 agent 保持运动、前往目标，但实际训练反馈显示：episode 长度骤降（109→74）、总奖励由正转负（+14→-39）、成功着陆率大幅下降。结合终止模式分析，agent 学会了一条更短的终止路径——非推力怠速（或快速出界/坠毁），使 early_stop 惩罚和 survival 惩罚的累计值低于“尝试着陆”的成本，从而导致整个任务动机崩塌。该连续惩罚的累积负担过度干扰了 explore-exploit 平衡。最稳妥的修复是直接删除该组件，恢复到接近 Iter2 的有效信号结构（仅依赖 distance progress + terminal failure 惩罚 + landing bonus）。这不是回退，而是删除产生负效果的组件，符合“每轮只改一个组件”的约束。

删除 `early_stop` 后，`failure_penalty` 组件回归到仅包含一次性出界/坠毁惩罚，保持 reward 信号清洁。其他系数（progress 20.0, survival -0.08, landing bonus 200, failure -30）沿用上一轮，没有新增惩罚负担，比例仍符合“生存惩罚 ≤ 0.3×progress 步均”的校准准则。

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

    # 出界或坠毁的一次性惩罚（移除 early_stop 连续惩罚）
    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash) else 0.0

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
- **audit**: 观测维度已全覆盖；信号缺口不在观测缺失，而在于 early_stop 连续惩罚引入的动机漏洞——agent 利用“尽快终止”来最小化累计负奖励。
- **behavior**: agent 大量 episode 选择非推力怠速或快速失败（长度骤降），使得成功着陆比例极低，总 reward 变为显著负值。
- **signal**: early_stop 连续惩罚在远离目标低速时持续施压，反而产生了比“尝试着陆”更优的捷径（提早终止），导致主进展信号失效。
- **level**: Level 2 — 结构变换：删除导致negative exploitation的 early_stop 连续惩罚组件。
- **hypothesis**: 移除 early_stop 后，agent 恢复以 progress（距离缩短）为主导的动机，回归 Iter2 的正向表现（+14 分），并可能进一步改善。
- **risk**: 若 agent 在未来出现“远处静止拖延直至环境自动终止”的漏洞，可能需要添加基于终止时距离检测的一次性失败惩罚，但目前 delta-distance progress 已提供足够向心动力。
