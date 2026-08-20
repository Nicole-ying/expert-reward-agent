# 设计理由
当前奖励函数已成功驱动agent实现稳定着陆并超过目标分数，但存在一个微小改进点：

- 组件 `angle_penalty` 的 active_rate 高达 99.9%，但 episode_sum_mean 仅为 -0.032，形式上的惩罚几乎为零，说明 agent 依靠其他组件已经将角度控制在极低水平，该组件已成为“空转”代码。
- 观测维度 `angvel`（角速度）未被任何组件使用，而角速度是着陆阶段姿态稳定的重要动态信号。**第 0 步审计**发现信号缺口：angvel 未被奖励函数覆盖，而它能够为旋转阻尼提供直接梯度信息。
- 当前 agent 长度 386 步、终止率 100%，行为是快速稳定着陆，整体表现优秀，因此改动应当保守。

**修改内容**：将无实际贡献的 `angle_penalty` 替换为 `angular_velocity_penalty`，使用 `next_obs[5]`（`angvel_n`），数学形式为 `-0.01 * (angvel_n ** 2)`。  
- 系数 0.01 与原 `angle_penalty` 一致，保持量级微小，且满足惩罚负担 ≤ 主信号 0.3x 的约束（主信号 per‑step ≈ 0.19，角速度惩罚 per‑step 预期远低于 0.057）。  
- 该变换属于 **Level 2 结构变换**（组件职责替换），为 agent 提供平滑的旋转减速梯度，可能进一步降低着陆瞬间的振荡。

其余组件（progress_reward、soft_landing、contact_stability、success_bonus）保持不变，因为它们已协调驱动了良好的着陆行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: replace zero‑contribution angle_penalty with angular_velocity_penalty
        to directly damp rotational motion using the previously unused angvel observation.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.05
    ANGVEL_PENALTY_WEIGHT = 0.01   # replaces angle_penalty
    CONTACT_WEIGHT = 0.1
    PROXIMITY_THRESHOLD = 0.5
    SUCCESS_DIST_THRESHOLD = 0.3
    SUCCESS_SPEED_THRESHOLD = 0.3
    SUCCESS_ANGLE_THRESHOLD = 0.2
    SUCCESS_SCALE = 1.0

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    speed = (x_v_n ** 2 + y_v_n ** 2) ** 0.5
    speed_bonus = 1.0 / (1.0 + speed)
    soft_landing = LANDING_WEIGHT * proximity * speed_bonus

    # ---------- 3) angular velocity penalty (replaces angle_penalty) ----------
    angular_velocity_penalty = -ANGVEL_PENALTY_WEIGHT * (angvel_n ** 2)

    # ---------- 4) contact stability reward ----------
    contact_flag = max(left_n, right_n)
    angle_bonus = 1.0 / (1.0 + abs(angle_n))
    contact_stability = (
        CONTACT_WEIGHT * proximity * contact_flag * speed_bonus * angle_bonus
    )

    # ---------- 5) success bonus (dense continuous factor) ----------
    proximity_factor = max(0.0, 1.0 - R_next / SUCCESS_DIST_THRESHOLD)
    speed_factor = max(0.0, 1.0 - speed / SUCCESS_SPEED_THRESHOLD)
    angle_factor = max(0.0, 1.0 - abs(angle_n) / SUCCESS_ANGLE_THRESHOLD)
    contact_factor = max(left_n, right_n)
    success_factor = (proximity_factor * speed_factor *
                      angle_factor * contact_factor)
    success_bonus = SUCCESS_SCALE * success_factor

    # ---------- aggregate ----------
    total_reward = (
        progress_reward + soft_landing + angular_velocity_penalty +
        contact_stability + success_bonus
    )

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angular_velocity_penalty": angular_velocity_penalty,
        "contact_stability": contact_stability,
        "success_bonus": success_bonus,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测 `angvel` 未被使用，而 `angle_penalty` 因角度常零已无实际约束力；对姿态的阻尼信号存在缺失。
- **behavior**: agent 学会了快速（len=386）稳定着陆，终止率 100%，分数已超目标。
- **signal**: 缺少对旋转速度的直接阻尼，削弱了着陆末段的姿态精细控制。
- **level**: Level 2
- **hypothesis**: 增加轻量角速度惩罚能提供旋转减速的明确梯度，进一步减少着陆振荡，微小提升分数或保持高分。
- **risk**: 系数极小，几乎不会影响正常飞行的快速旋转；最差情况是不改变任何行为，但不会退化已收敛的策略。