# Response Record

# 设计理由
在 iter2 中，agent 因加入全局侧向漂移惩罚 `-0.2 * vy^2` 导致整体评价大幅下降（score 从 2.91 暴跌至 -416），尽管多数 episode 的组件和仍为正（~216），但约 3/20 的 episode 极早期终止（<150 步且 score<-50），说明在摔倒过程中侧向惩罚继续堆积大量负值，形成死亡螺旋，拉低了全局得分。  
当前奖励框架的信号覆盖已齐全（前进激励、高度/直立存活门控、能量惩罚、侧向惩罚），主要问题是侧向惩罚在 agent 姿态恶化时未减轻。  
因此本轮只修改一个组件：将 `lateral_penalty` 从全局惩罚改为**门控惩罚**，乘以 `survival_gate`，使得当 agent 接近边界（高度或直立度恶化）时侧向惩罚自动减弱，从而降低死亡 spiral 的风险，帮助 agent 从危险状态恢复。系数保持不变，期望 per-step 惩罚负担不高于主信号的 0.3 倍。

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

    # ---- 2. Survival gate: uprightness factor ----
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, min(1.0, body_up_z))

    survival_gate = height_factor * upright_factor

    # ---- 3. Main progress component (gated) ----
    w_progress = 1.0
    forward_gated = w_progress * vx * survival_gate

    # ---- 4. Lateral drift penalty (now gated) ----
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
- **audit**: 所有终止条件均有前兆软信号，但侧向惩罚在姿态恶化时未减轻，导致早期终止 episode 的负奖励雪崩。
- **behavior**: agent 在多数 episode 中保持存活并获得正向组件和，但因约 15% 的 episode 极早期摔倒而总评严重为负。
- **signal**: lateral_penalty 全局惩罚过重，缺少在生存概率降低时的自动衰减。
- **level**: Level 2
- **hypothesis**: 将 lateral_penalty 乘以 survival_gate 可在 agent 接近边界时减弱侧向惩罚，减少死亡螺旋，提高早期终止 episode 的奖励底限，从而提升总体平均 score。
- **risk**: 门控化可能轻微减弱对健康状态下侧向漂移的抑制，但由于 survival_gate 在安全区域接近 1，影响有限；若效果不佳，可后续调整 w_lateral。
