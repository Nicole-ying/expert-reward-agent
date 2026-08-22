# 设计理由

## 为什么以前都失败了
回顾迭代历史：
- **iter 1** 骨架 `action_penalty + forward_gated` 取得 2.91 分（best），是一个温和的起点。
- **iter 3** 在侧向惩罚上加了 survival gate 后达到 **13.55 分的历史最高**，说明 gate 在终止前衰减惩罚/奖励的方向有效。
- 但随后 **iter 4** 放宽 upright gate 导致分数崩溃（-464），说明姿态约束是必需的。
- **iter 5 完全移除所有 gate**，只保留 forward_progress + lateral_penalty，结果 agent 在 1000 步全存活下拿到了 **-1815** 的极低外部评分——这揭示了一个关键事实：**无 gate 的前进奖励会被 exploit**（agent 可能跌倒滑行或做出高能效动作刷 reward，但外部环境惩罚巨额能量/接触成本）。
- **iter 6** 重新加入 gate，但对 gate 下界做了 `max(0.1, ...)` 的硬性抬升，导致 agent 即使在严重偏离健康姿态时仍能获得 10% 的主奖励。切断了 gate 的“切断”功能，前进变成了无风险的低水平稳态，最终得分 -174。

**核心教训**：主前进奖励必须被一个能在“接近终止边界”时迅速衰减至近零的 gate 所乘，同时需要引入轻量的动作效率惩罚来抑制内部隐藏的能量/接触代价。

## 新骨架选了什么算子
基于 §2 完整公式算子库，选择以下框架（与已尝试的 `forward_progress + lateral_penalty` 族有本质不同）：

| 组件 | 算子 | 数学形式 | 理由 |
|---|---|---|---|
| 前进奖励 | dense_state_signal（凸化） | `2.0 * max(0, body_x_vel) ** 2` | 提供足够的正向梯度 |
| 健康 gate | soft_health_gate (min-of-hinge) | `min(height_factor, upright_factor)`，每个因子在安全区为 1，在终止边界前线性衰减至 ~0 | 确保前进只在健康姿态下生效，越接近危险奖励越低 |
| 动作效率 | quadratic_penalty | `-0.02 * sum(action_i ** 2)` | 抑制高扭矩、降低隐藏能量惩罚 |
| 侧向约束 | quadratic_penalty | `-0.1 * body_y_vel ** 2` | 减少侧向漂移，提升前进效率 |

**目前已尝试的 failed 方向**：
- 无 gate（iter 5）
- gate 下界固定 0.1（iter 6）
- 乘积 gate + lateral 惩罚单独使用（iter 2-4 的部分形态）

**本次新骨架的关键不同**：
- 使用 **min-of-hinge** 替代乘积 gate：min 确保只要任一维度（高度或直立）进入危险区，整个 gate 就快速衰减，不会像乘积那样允许“一个维度略低、另一个很高”时 gate 仍然偏高。
- hinge 形式只在安全边界附近生效（衰减区间设在终止边界的 60-80%），安全区内 gate ≡ 1，不做无谓惩罚。
- 重新引入被过早放弃的 action penalty，以对抗隐藏能量代价——这是历史最佳 iter 3（13.55 分）中存在的组件，之后被移除。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 信号提取 ──
    body_z_next      = next_obs[0]
    qx_next          = next_obs[2]
    qy_next          = next_obs[3]
    body_x_vel_next  = next_obs[13]
    body_y_vel_next  = next_obs[14]

    # ── 健康因子：高度 ──
    # 终止边界：[0.2, 1.0]；安全区 [0.3, 0.9]
    if body_z_next < 0.3:
        height_factor = (body_z_next - 0.2) / 0.1   # 在 z=0.2 时为 0
    elif body_z_next > 0.9:
        height_factor = (1.0 - body_z_next) / 0.1   # 在 z=1.0 时为 0
    else:
        height_factor = 1.0
    height_factor = max(0.0, min(1.0, height_factor))  # 允许到 0

    # ── 健康因子：直立度 ──
    body_up_z = 1.0 - 2.0 * (qx_next**2 + qy_next**2)  # 1=完全直立, -1=倒立
    # 安全阈值 0.3（约 17° 倾斜），在 0.3→-0.2 线性衰减
    upright_factor = (body_up_z + 0.2) / 0.5           # 在 -0.2 时为 0
    upright_factor = max(0.0, min(1.0, upright_factor))

    # ── 健康门控（min-of-hinge） ──
    health_gate = min(height_factor, upright_factor)

    # ── 前进奖励（gated） ──
    forward_speed = max(0.0, body_x_vel_next)
    progress_reward = 2.0 * (forward_speed ** 2) * health_gate

    # ── 动作效率惩罚 ──
    action_penalty = -0.02 * sum(a_i ** 2 for a_i in action)

    # ── 侧向漂移惩罚 ──
    lateral_penalty = -0.1 * (body_y_vel_next ** 2)

    total_reward = progress_reward + action_penalty + lateral_penalty

    components = {
        'progress': progress_reward,
        'action_penalty': action_penalty,
        'lateral_penalty': lateral_penalty,
        'health_gate': health_gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 缺少动作效率信号，且 gate 下界 0.1 切断了惩罚梯度，使 unhealthy 姿态仍能得到可观奖励
- **behavior**: agent 在勉强维持姿态并通过 low-effort 动作刷 forward_progress，但外部隐藏惩罚拉低总分
- **signal**: 需要恢复在危险边界将 gate 压至近零的能力，同时引入 action penalty 抑制高扭矩 exploit
- **level**: Level 3（重建骨架）
- **hypothesis**: min-of-hinge gate 确保前进奖励严格依赖于健康姿态，action penalty 压低隐藏能耗代价，应大幅提升外部评分
- **risk**: gate 在早期较差姿态时可能使主奖励接近消失，导致学习信号稀疏；但 hinge 缓冲区和凸化前进速度提供的剩余梯度应能缓解，如出现停滞可考虑在 gate 中引入 0.05 的最小梯度底线（当前未设）