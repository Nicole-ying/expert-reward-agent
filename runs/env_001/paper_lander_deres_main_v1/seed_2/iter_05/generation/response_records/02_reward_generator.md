# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 主学习信号: 距离减少奖励 (progress_reward) ---
    # 当前距离
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # 下一步距离
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_current - dist_next  # 正值 = 靠近目标

    # --- 稳定/安全约束: 姿态惩罚 ---
    # 惩罚角度偏离水平（obs[4] 接近 0 为好）
    orientation_penalty = -0.3 * (obs[4] ** 2)

    # --- 任务完成近似信号 (landing_bonus) ---
    # 当满足多重着陆条件时给予额外奖励
    # 条件: 接近中心, 高度极低, 速度很小, 姿态水平, 双腿同时触地
    landing_bonus = 0.0
    if (abs(next_obs[0]) < 0.5 and
        next_obs[1] < 0.3 and
        abs(next_obs[2]) < 0.5 and
        abs(next_obs[3]) < 0.5 and
        abs(next_obs[4]) < 0.2 and
        next_obs[6] == 1.0 and
        next_obs[7] == 1.0):
        landing_bonus = 5.0

    # --- 总奖励 ---
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 组件与角色

- **progress_reward（主学习信号）**：每步奖励飞行器相对于平台中心的欧氏距离减小量。该信号密集且与任务目标（到达平台）直接相关，提供清晰的梯度方向。
- **orientation_penalty（稳定约束）**：轻量惩罚当前姿态偏离水平（角度平方）。引导飞行器在降落时保持水平，防止侧翻，但不严重影响机动阶段。权重 0.3 经过缩放任其不会主导主信号。
- **landing_bonus（任务完成近似信号）**：当下一步观测同时满足近中心、低高度、低速、水平且双腿触地时，给予固定正奖励。该 proxy 用多个连续条件组合避免误触，激励飞行器完成稳定着陆。权重设定为 5.0，足以与单步 progress 拉开差距，但不会压制动过程。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因

环境显式成功/失败标志不可用（`explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，`info` 为空字典）。强行伪造 `info['success']` 或基于终止条件硬编码会违反接口契约，故不采用。

## 未纳入的效率/动作代价与后续迭代

- 燃料消耗（动作代价）在 v1 中未使用。当前阶段首要目标是学会接近并稳定着陆，效率优化在 agent 掌握基本行为后以小权重加入。
- 时间惩罚和复杂门控暂不引入，避免过早约束导致探索停滞。

## 训练后需观察的 failure mode

- **目标附近震荡**：progress_reward 可能诱导飞行器在平台上方左右摆动以持续获得正奖励，而无法收敛到零速度。应监控 `landing_bonus` 的触发频率，若长时间未触发，说明需要加强完成信号或引入速度惩罚。
- **高奖励但未成功**：如果累计 reward 很高但 episode 不以满足着陆条件结束，可能 agent 学会了不停接近后退的循环。此时需降低 progress_reward 或加入对震荡的惩罚。
- **姿态失控**：若 orientation_penalty 权重偏大将导致不敢倾斜以平移，可能使水平移动过慢，需要微调权重。
- **着陆不完整**：landing_bonus 条件苛刻，可能难以初次触发，需密切观察是否随训练触发率上升；若始终为零，可考虑放宽阈值或增加中间引导。
