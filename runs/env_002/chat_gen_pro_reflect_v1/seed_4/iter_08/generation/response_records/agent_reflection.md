# Response Record

# 设计理由
当前奖励骨架（forward_reward + gate + energy_penalty）在多次迭代中已经到达过 297.68，但最近三轮尝试用额外惩罚或凸化速度奖励均导致崩塌（score < 282）。说明在此骨架内继续添加新组件治标不治本。

审计发现 gate 部件在 `|hull_angle| < 0.25` 时产生 **>1 的放大效应**（平均 gate_factor ≈ 1.86），这意味着 agent 花费大量时间维持极度直立姿态以获得额外奖励，可能抑制了更大步幅与速度的探索。正常行走中适当的前倾/后倾是必要的，不应被过度惩罚或奖励。

因此本轮的 Level 2 修改：**将 gate_factor 裁剪到 [0, 1]**，使其成为纯粹的安全衰减器，不再奖励过度直立。其他信号保持不变。这应当让速度梯度更为主导，帮助突破当前 ~282 的平台，回归并超越 297 的 best。

**系数校准**：无新系数引入，不增加额外惩罚，不会引发 len 崩塌。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (linear main signal)
    # ============================================================
    forward_reward = 1.0 * max(0.0, horizontal_vel)

    # ============================================================
    # Component 2: soft_health_gate (pure attenuation, no boost)
    #   gate = 1.0 for |angle| ≤ 0.25, decays to 0.0 at |angle| ≥ 0.5.
    #   Clipped to max 1.0 to remove exaggerated upright reward.
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = min(1.0, gate_raw)   # <--- the only structural change
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (unchanged)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty

    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: gate 组件会在极度直立时产生高达 2× 的奖励放大，导致过度奖励直立姿态，可能挤占速度信号的优化空间。
- **behavior**: agent 趋于保守直立、速度难再提升，得分在 282 附近停滞。
- **signal**: 主速度奖励被 gate 的直立放大效应部分扭曲，需要恢复以速度为核心的梯度。
- **level**: Level 2
- **hypothesis**: 裁剪门控到纯衰减后，速度信号将获得更纯净的主导权，推动 agent 探索更快的步态，从而超越此前 297 分。
- **risk**: 门控放大移除后初期稳定性可能轻微下降，但衰减机制依然存在，预计最终能保持平衡并提速。
