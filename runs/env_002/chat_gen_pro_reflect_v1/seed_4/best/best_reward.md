# 设计理由
**尺度修复**：上一轮引入的 `energy_penalty`（系数 -0.2）使每步惩罚约为前进奖励的 80%，严重压制了动作探索，导致 agent 偏好输出极小力矩而无法维持平衡，长度从 901 骤降至 86。实际惩罚均值虽只有约 -0.06/步，但梯度信号仍被惩罚主导，迫使 policy 趋向零动作。本次将系数从 -0.2 降至 -0.02，使每步惩罚约为主信号的 3%，远低于 30% 上限，让前进速度信号重新主导优化方向。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # ============================================================
    forward_reward = 1.0 * horizontal_vel

    # ============================================================
    # Component 2: soft_health_gate (safety: attenuates forward reward
    #   when balance degrades; gate = 1.0 for |angle| ≤ 0.25 rad)
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (scaling fix: coefficient -0.02)
    #   Reduction from -0.2 to -0.02 ensures per-step penalty ~0.006
    #   which is only ~3% of main signal, restoring action exploration.
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + energy_penalty

    # ============================================================
    # Components dict
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gate_factor': gate_factor,
        'gated_forward': gated_forward,
        'energy_penalty': energy_penalty,
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少摔倒前兆的软信号（如角度过大惩罚），但当前急迫问题是能量惩罚过强导致训练崩溃；先修复尺度再考虑信号补齐。
- **behavior**: agent 在加入 -0.2 动作惩罚后迅速收敛到极小动作，无法维持平衡，几乎每回合立即摔倒（len=86, term=100%）。
- **signal**: 能量惩罚梯度远超前进奖励，导致 policy 抑制所有动作，前进和平衡信号均被侵蚀。
- **level**: Level 1
- **hypothesis**: 将能量惩罚系数降至 -0.02 可使前进速度信号重新主导梯度，agent 恢复有效步态探索，len 和 score 应迅速回升。
- **risk**: 能量消耗可能较无惩罚时略高，但极小的系数不会明显拖累外部评分。