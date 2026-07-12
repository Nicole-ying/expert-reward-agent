# Response Record

# 设计理由
前进速度奖励当前为线性 `body_x_velocity`，但 score 停留在 260.75，且平均每步速度约 0.62，表明 agent 已学会稳定前进但缺乏追求更高速度的动力。根据 Formula switching guide，“线性正奖励 → 凸化”，故在保留线性项基础上增加二次项 `0.5 * max(0, body_x_velocity)**2`，即 `forward_reward = body_x_velocity + 0.5 * max(0, v)**2`。此结构让低速段保持原始梯度（线性项主导），高速段获得超线性激励（二次项加速拉开），推动 agent 突破当前速度平台。系数 `0.5` 使当前平均速度下的奖励从 0.64 升至约 0.84，量级变化温和，不会破坏已有生存行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控（独立正向生存奖励） ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 主学习信号：凸化前进速度 ----------
    forward_reward = 1.0 * body_x_velocity + 0.5 * max(0.0, body_x_velocity)**2

    # ---------- 直立姿态约束（hinge 惩罚，系数降低） ----------
    upright_error = max(0.0, 0.7 - body_up_z)
    upright_penalty = -0.3 * upright_error

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + height_reward + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止前兆（高度）、目标梯度（前进速度）、效率信号（动作惩罚）均已覆盖，无缺口，无僵尸组件。
- **behavior**: agent 已学会稳定中速行走（平均每步 0.64 前进奖励，80% 终止率，无早停），但缺乏向更高速度突破的激励。
- **signal**: 前进速度奖励是线性的，高速度带来的额外收益不足，导致 agent 安于当前中等速度。
- **level**: Level 2
- **hypothesis**: 凸化前进奖励使高速步态获得超线性回报，在不破坏现有生存策略的前提下，逐步推高 agent 的前进速度，从而提升总得分。
- **risk**: 高速步态可能加剧力矩使用和姿态抖动，action_cost 与 upright_penalty 会相应增大，若惩罚占比过高可能抑制速度提升；但当前惩罚负担 0.5x 门限，应有余地。
