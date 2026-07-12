# Response Record

# 设计理由

## 信号覆盖审计（第 0 步）
- **终止→前兆**：`body_fallen_over` 的前兆是 `hull_angle` 过大。当前 `gate_factor` 在角度 0.25 rad 开始衰减，0.5 rad 完全置零，已有软信号。额外的 `angle_hinge_penalty`（阈值 0.4 rad）从未触发（`active_rate = 0%`），说明当前策略的角度始终落在安全区，该惩罚是僵尸组件，既无梯度贡献，又可能轻微扰动训练动态。
- **目标→进度**：前进速度、距离、最小化能量。主驱动 `forward_reward` 是线性的 `horizontal_vel`，在最佳分数 297 时步均速度约 0.3–0.5。线性奖励对“更快”的边际激励恒定，无法提供突破性驱动力。
- **效率信号**：动作维度 4，已有弱能量惩罚，无额外动作惩罚需求。
- **僵尸组件**：`angle_hinge_penalty` 应删除。
- **一句话结论**：当前 reward 缺的是对更高速度的凸状激励，且携带一个无用的僵尸惩罚。

## 行为诊断
**agent 在做什么？** 稳定行走，速度中等，len 在 900–1000 步，分数在 281–297 之间波动。agent 已经掌握了安全步态，但无法进一步加速以突破 300 分。

**干预哪个目标？** 主目标：前进速度。当前线性奖励对加速的边际收益不变，换成凸函数（平方）能提供逐渐增强的加速度梯度。

**这个方向还值得继续吗？** 迭代记录显示：尝试施加惩罚（iter3 能量惩罚过强、iter5 角度 hing e）导致分数下降；而调低惩罚、保留主信号（iter4）则带来历史最佳。因此，不应该再加惩罚，而应当增强前进正信号的梯度强度。同方向（增强 forward_reward）尚未尝试过凸化，属于可行新方向。

## 干预层级：Level 2（结构变换）
将 `forward_reward` 由线性改为平方形式，属于 “有界凸化” 结构变换（对应公式切换指南：`dense_state_signal` 凸化），同时删除无触发的 `angle_hinge_penalty`，保持 reward 干净。

- **数学形式**：`forward_reward = 2.0 * max(0, horizontal_vel) ** 2`  
  系数 2.0 保证在典型速度 ~0.5 m/s 时量级与旧线性奖励相等（0.5 vs 0.5）；速度越低惩罚越重（0.3 → 0.18），速度越高收益急剧增加（0.8 → 1.28），形成“加速更有价值”的引导。

- **系数校准**：
  - 主信号 per‑step ≈ `forward_reward` 在 0.5–1.0 之间（取决于速度），能量惩罚 per‑step ≈ 0.02×平均动作平方（~1.33）≈0.027，占比 <5%，远低于 30% 上限。
  - `gate_factor` 不受影响，仍提供从安全到危险的连续衰减，其均值 1.6 表明大部分时间角度很小，gate 近乎 1.0，不会意外塌缩。
  - 删除 `angle_hinge_penalty` 后，总惩罚负担仅能量惩罚一项，满足 0.5 倍主信号的约束。

- **为什么预料能改善**：凸状奖励增加加速的边际收益，push agent 探索更高速度但依然稳定的步态，从而在相同 episode 长度内走得更远、速度更高，外部得分有望从 ~297 提升至 300+。同时移除僵尸组件消除噪音。

- **风险**：过度加速可能微幅增加能耗或偶尔加大角度，但 `gate_factor` 会在角度变大时立即衰减主奖励，形成平衡约束。最坏情况是步态不稳定导致几百分的波动，但不会像之前能量惩罚那样导致完全崩溃（len 暴跌）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    horizontal_vel = obs[2]
    hull_angle = obs[0]

    # ============================================================
    # Component 1: forward_progress (convexified main signal)
    #   Squared positive velocity gives increasing marginal reward
    #   for faster walking, pushing the agent above the 0.5 m/s plateau.
    #   Coefficient 2.0 matches the linear reward at typical speed.
    # ============================================================
    forward_reward = 2.0 * max(0.0, horizontal_vel) ** 2

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
- **audit**: 缺少对更高速度的凸激励，且存在未触发的僵尸惩罚；gate 已覆盖安全前兆。
- **behavior**: agent 在安全速度平台上稳定行走，无法加速突破 300 分。
- **signal**: 前进速度线性的边际收益不足以驱动更快步态；角度 hinge 惩罚从未触发，无实际信号。
- **level**: Level 2
- **hypothesis**: 平方速度奖励提供更强的“更快更好”梯度，配合 gate 约束，可推动步态加速并守住平衡，从而将外部得分从 ~297 提升至 300。
- **risk**: 可能出现短暂加速/姿势波动，但 gate 会压制极端情况；能量惩罚极轻，不会像 iter3 那样导致崩溃。
