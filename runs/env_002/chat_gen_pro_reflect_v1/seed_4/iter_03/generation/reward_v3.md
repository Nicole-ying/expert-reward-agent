# 设计理由
## 正常模式 — 结构变换：用能量消耗惩罚替换僵尸平衡惩罚

当前奖励函数中 `angle_penalty` 和 `angle_deviation` 的 active_rate 均为 0%，说明 hinge 逻辑从未被触发——agent 已经稳定保持身体竖直，不会进入惩罚区域。同时 `angular_vel_penalty` 的幅度极小（episode_sum_mean ≈ -0.05），对学习几乎无贡献。这些组件已成为**僵尸代码**。

另一方面，环境任务明确要求“最小化能量消耗”，但当前奖励函数完全没有能量相关信号。动作空间为 4 维力矩，直接注入能量惩罚可以引导 agent 学习更有效率的步态，进而提高融合了能量指标的外部得分（我们差距仅 12 分，很可能就是能量项的贡献）。

### 改动
1. **删除**所有基于 `hull_angle` 和 `hull_angular_vel` 的惩罚项（`angle_deviation`、`angle_penalty`、`angular_vel_penalty`、`balance_penalty`）。
2. **新增** `energy_penalty = -0.2 * sum(action_i^2)`，作为唯一的约束项。
3. 保持 `forward_reward`、`gate_factor`、`gated_forward` 不变，`gate_factor` 继续在身体偏离过大时衰减前进奖励，作为安全网。

### 系数校准
- 主信号 per-step ≈ `gated_forward`/len ≈ 949/901.5 ≈ 1.05。
- 目标惩罚 per-step 设在 ≈ 0.3×1.05 ≈ 0.31。
- 假设典型动作幅度 ~0.5，则 `sum(action²)` ≈ 1.0，系数 -0.2 给出 -0.2，偏低但安全；若未来需要更强约束可升至 -0.3。
- 惩罚负担占主信号 < 20%，远低于 0.5 上限，不会破坏探索。

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
    # Component 3: energy_penalty (replaces balance_penalty)
    #   Penalize squared joint torques to encourage efficiency.
    #   Coefficient -0.2 yields ~ -0.2 per step under normal operation.
    # ============================================================
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.2 * action_power

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
- **audit**: 缺少能量消耗约束；平衡约束组件已全部僵尸（active_rate 0%）
- **behavior**: agent 保持良好平衡，以中等速度稳定行走；所有 episode 均正常终止（likely reaching end of terrain）
- **signal**: 奖励完全由前进速度支配，无效率引导；僵尸惩罚不产生梯度
- **level**: Level 2
- **hypothesis**: 加入动作幅度惩罚会迫使 agent 探索更节能的步态，从而拉动外部得分中能量项的提升，逼近 300 分
- **risk**: 若地形极崎岖，动作幅度天生较大，惩罚可能压制必需力矩，降低速度并拖累前进得分；可后续按需调整系数或引入与接触状态的联动