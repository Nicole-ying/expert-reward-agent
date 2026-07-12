# 设计理由
本轮基于累积迭代记录和训练反馈，诊断出 iter 6 将前进奖励从线性改为平方速度奖励（`2.0 * max(0, vel)^2`）导致灾难性崩溃：len 从 1032 暴跌至 136，score 从 297.68 跌至 -59.50，全部 episode 因摔倒终止。平方奖励引入了无界且边际递增的激励，使得 agent 为追求极高速度而忽视姿态约束（gate 乘性压制不足以平衡），最终快速摔倒。  
修改策略：将 `forward_reward` 从二次方形态恢复为**线性**（系数 1.0，与成功迭代 iter 4 一致），保留 gate 和能量惩罚不变。这是一个 Level 2 结构变换，属于“极端值支配 reward”场景下的数学形态修正（凸化→线性），每步奖励振幅重新受控，让 gate 能有效导引安全步态。  
系数校准：linear `vel` per‑step 在 iter 4 平均约 0.3 m/s 时产生约 0.3 奖励，gate 乘积后约 0.3–0.5，与能量惩罚（‑0.036）相加，前进信号仍主导。当前代码其他系数未变，因此预期回归到 iter 4 的稳定行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (linear main signal)
    #   Simple proportional reward for forward speed, bounded by
    #   the environment's natural velocity limits.
    # ============================================================
    forward_reward = 1.0 * max(0.0, horizontal_vel)

    # ============================================================
    # Component 2: soft_health_gate
    #   Attenuates forward reward when tilt approaches fall threshold.
    #   gate = 1.0 for |angle| ≤ 0.25, decays to 0.0 at |angle| ≥ 0.5.
    # ============================================================
    gate_lower = 0.25
    gate_upper = 0.5
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw
    gated_forward = forward_reward * gate_factor

    # ============================================================
    # Component 3: energy_penalty (light action smoothness)
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.02 * action_power

    # ============================================================
    # Total reward (angle_hinge_penalty removed – never triggered)
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
- **audit**: 当前 reward 缺少对成功到达地形末端的奖励，但核心崩溃成因是前进奖励的平方形态造成速度激进与摔倒。  
- **behavior**: agent 为追逐高二次方奖励做出极端步态，快速摔倒（len 136, terminated 20/20）。  
- **signal**: forward_reward 无界且凸性占据主导，gate 无法有效约束。  
- **level**: Level 2  
- **hypothesis**: 回归线性速度奖励后，agent 将重新学习在 gate 约束下稳定行走，恢复 ~297 的 baseline 分数。  
- **risk**: 可能仍然略低于 300 分门槛，后续需探索更精细的进度奖励（如 LIDAR 或步态效率），但本修改可先摆脱崩盘。