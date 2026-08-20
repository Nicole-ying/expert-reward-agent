# 设计理由
当前奖励函数的问题核心在于 forward 奖励系数过高（2.0），导致 agent 为了最大化速度而采取激进策略，忽视了高度维持的稳定性，最终频繁触发高度越界终止（terminated rate 80%）。  
历史证据：iter4 使用 1.0 系数时获得了 +69.24 分和 989 步存活；iter5 将系数加倍至 2.0 后分数暴跌至 -664.53；iter6 虽调整了直立惩罚但仍承受该高系数的负面影响（-231.68）。  
本次修改将 `forward_reward` 系数从 2.0 降为 1.0，使其回归到 iter4 的有效水平，恢复“前进激励”与“稳定存活”之间的平衡。其余组件（高度门控、横向/垂直惩罚）保持不变，直立惩罚虽为僵尸但不造成伤害，留待后续处理。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Height gate ----------
    height_low_danger = max(0.0, (0.30 - body_z) / 0.10)
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    height_high_danger = max(0.0, (body_z - 0.90) / 0.10)
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # ---------- Reward components ----------
    # Reduced forward multiplier to 1.0 (was 2.0) to restore stability
    forward_reward = 1.0 * body_x_velocity * height_factor

    # Upright penalty (currently dead but harmless; keep for now)
    upright_penalty = -0.5 * max(0.0, 0.0 - body_up_z)

    lateral_deviation = abs(body_y_velocity)
    lateral_penalty = -0.1 * max(0.0, lateral_deviation - 0.3)

    vertical_deviation = abs(body_z_velocity)
    vertical_penalty = -0.1 * max(0.0, vertical_deviation - 0.5)

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_penalty + lateral_penalty + vertical_penalty

    components = {
        "forward": forward_reward,
        "upright_penalty": upright_penalty,
        "lateral_penalty": lateral_penalty,
        "vertical_penalty": vertical_penalty,
        "height_factor": height_factor,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止前兆信号（高度）已存在但强度不足以阻止高频终止；直立惩罚为僵尸组件，横向/垂直惩罚贡献微小，缺少低高度独立恐惧信号。
- **behavior**: agent 学会了向前移动（forward 均值 380/回合），但因激进速度追求导致频繁高度越界（terminated 80%），总得分被压低至负值。
- **signal**: forward 激励过强（系数 2.0）打破了稳定与前进的平衡，使 agent 为最大化奖励而牺牲高度安全。
- **level**: Level 1
- **hypothesis**: 将 forward 系数降至 1.0 可以重现 iter4 的稳定特性，减少越界终止，使存活步长和总得分回升至正值。
- **risk**: 极低——该系数在 iter4 已被证明有效，不会引入新的不稳定因素；可能短期前进速度略降，但存活时间延长带来的总奖励增长可弥补。