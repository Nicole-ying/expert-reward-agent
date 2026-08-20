# Response Record

# 设计理由

## 信号覆盖审计

经过逐项审计，我发现当前奖励函数存在三个关键的信号缺口：

- **能耗信号缺失**：动作空间维度为8，且当前奖励函数完全没有动作惩罚，导致能量消耗未被考虑。从训练反馈看，11/20的episode因失败而终止，这可能与动作过大导致不稳定有关。
- **侧向漂移抑制缺失**：虽然任务核心是前进，但没有对侧向速度obs[14]施加任何约束，导致可能出现侧向偏移浪费能量。
- **高度惩罚形态不当**：当前高度惩罚使用的线性误差是"硬约束惩罚"，但它没有在接近边界时提供渐进式的前兆信号。这种情况更适合用高度门控因子，而不是作为独立的惩罚项。

## 行为诊断

从训练反馈分析：

- **Agent在做什么**：forward_upright_product为正值(281.6)，说明agent确实在尝试向前移动，但实际得分为负且终止率高(55%)，表明控制不稳定。agent可能通过高扭矩、高风险的动作模式短暂获得高速度奖励，但代价是频繁越界终止。
- **干预目标**：基于信号覆盖审计，应该优先引入动作惩罚，因为：(1)动作维度≥6，能耗惩罚有明显必要性；(2)降低扭矩可以提高控制稳定性，间接改善高度和终止率；(3)该惩罚是全覆盖信号，不依赖状态阈值，鲁棒性强。
- **方向价值**：这是第一轮修正，当前骨架`forward_upright_product`信号方向正确，不需要推倒重来，只需通过添加辅助组件改善训练稳定性。

## 选择的干预层级：Level 2 结构变换

**添加组件**：动作惩罚 `torque_penalty = -w_action * sum(action^2)/8`

**变换原因**：
- 动作维度≥6且当前代码无动作惩罚，属于信号缺口
- 添加此组件可以提供能耗梯度，引导agent学习高效、稳定的步态
- 平方惩罚比线性惩罚更稳定，对大幅值扭矩施加更强抑制
- 通过除以8实现归一化，使惩罚尺度与动作维度解耦

## 系数校准

- 主信号 per-step 估算：`forward_upright_product / len = 281.6 / 647.8 ≈ 0.43`
- 动作平方和归一化后范围[0, 1]，均值约为0.2-0.4
- 目标动作惩罚 per-step：`0.03 * 0.3 ≈ 0.009`，仅为主信号的~2%
- 系数0.03确保惩罚温和，避免阻碍探索，但提供足够的梯度引导节能行为
- 总惩罚负担远低于0.5x主信号，满足设计校准要求

这个修改是**可验证的**：如果成功，我们应该看到score提升（因为减少无谓的能量消耗），同时terminated率下降（因为更稳定的动作模式），且len保持或提升。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from observation
    body_z = obs[0]                # torso height
    qx = obs[2]                    # quaternion x component
    qy = obs[3]                    # quaternion y component
    body_x_vel = obs[13]           # forward (world‑x) velocity

    # ----- uprightness (projection of body z-axis onto world z-axis) -----
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))   # numerical safety clamp

    # ----- core learning signal: forward velocity modulated by uprightness -----
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * body_up_z

    # ----- safety hinge penalty on torso height -----
    w_h = 0.5
    low_thresh = 0.35
    high_thresh = 0.85
    low_error = max(0.0, low_thresh - body_z)
    high_error = max(0.0, body_z - high_thresh)
    height_penalty = -w_h * (low_error + high_error)

    # ----- action penalty for energy efficiency -----
    w_action = 0.03
    torque_penalty = -w_action * sum(action**2) / 8.0

    total_reward = forward_reward + height_penalty + torque_penalty

    components = {
        "forward_upright_product": forward_reward,
        "height_survival_penalty": height_penalty,
        "torque_penalty": torque_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失动作惩罚(维度8≥6)、侧向漂移抑制，高度惩罚应改为门控而非独立罚项
- **behavior**: agent在尝试前进但控制不稳定，高扭矩行为导致55%的终止率
- **signal**: 缺少动作能耗信号和侧向漂移惩罚，动作维度过高未受约束
- **level**: Level 2
- **hypothesis**: 添加动作惩罚将抑制高扭矩操作，引导学习更稳定、节能的步态，预期可降低终止率并提高前进效率
- **risk**: 惩罚过强可能使agent贪图静止而不探索前进，但系数0.03/8温和且主signal≥0.02x保证了前进激励占主导
