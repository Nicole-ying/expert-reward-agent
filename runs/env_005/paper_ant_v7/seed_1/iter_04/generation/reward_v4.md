# 设计理由
当前奖励函数中 `upright_factor` 直接取 `body_up_z`（范围 0~1），这意味只要身体有一点点前倾（例如从 1.0 跌到 0.9），gate 立刻下降到 0.9，前进奖励同比例衰减。四足机器人高速运动必然伴随显著的身体俯仰和倾斜，这种过于严格的 upright gate 会扼杀所有快速步态的尝试，迫使 agent 停留在最安全但几乎静止的状态。  

因此对 `upright_factor` 做 **hinge 重建**：将映射改为 `(body_up_z + 0.5) / 1.5`，并用 `max(0, min(1, ...))` 夹断。这样在 `body_up_z = 0`（水平）时仍有约 0.33 的 gate 值，在正常动态倾斜区（如 0.5~0.8）能得到较高因子，同时完全倒立时依然归零，避免完全失控。这是唯一改动的组件，属于 **Level 2 结构变换**。  

系数校准：goal per‑step 前进奖励约 1.1，修改后 gate 在常见倾斜范围内会略微放大，不会造成奖励爆炸；侧向惩罚和动作惩罚均未调整，总惩罚负担仍远低于主信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation signals
    body_z = obs[0]                     # body height
    vx = obs[13]                        # forward velocity (world x)
    vy = obs[14]                        # lateral velocity (world y)
    quat_x = obs[2]
    quat_y = obs[3]

    # ---- 1. Survival gate: height factor ----
    low_safe  = max(0.0, min(1.0, (body_z - 0.2) / 0.2))
    high_safe = max(0.0, min(1.0, (1.0 - body_z) / 0.2))
    height_factor = min(low_safe, high_safe)

    # ---- 2. Survival gate: uprightness factor (HINGE relax) ----
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, min(1.0, (body_up_z + 0.5) / 1.5))

    survival_gate = height_factor * upright_factor

    # ---- 3. Main progress component (gated) ----
    w_progress = 1.0
    forward_gated = w_progress * vx * survival_gate

    # ---- 4. Lateral drift penalty (gated) ----
    w_lateral = 0.2
    lateral_penalty = -w_lateral * (vy ** 2) * survival_gate

    # ---- 5. Energy efficiency term ----
    w_energy = 0.01
    action_penalty = -w_energy * sum(a**2 for a in action)

    # Total reward
    total_reward = forward_gated + lateral_penalty + action_penalty
    components = {
        'forward_gated': forward_gated,
        'lateral_penalty': lateral_penalty,
        'action_penalty': action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止前兆（高度、倾斜）均已覆盖，但倾斜约束过严，抑制了高速探索必需的身体动态。
- **behavior**: agent 学会在安全区间内保持极小速度以维持 gate=1，存活长但移动极慢，导致外部评估得分极低。
- **signal**: upright_factor 太敏感，需放宽以允许高速步态带来的自然倾斜。
- **level**: Level 2
- **hypothesis**: 放宽 upright gate 会解除对前倾的惩罚，让 agent 敢于尝试更快的步态，从而提升实际前进速度与外部评估分数。
- **risk**: 过度倾斜可能导致跌倒，但 hinge 最低处仍保留 0 关门，同时 height gate 继续保护高度，风险可控。