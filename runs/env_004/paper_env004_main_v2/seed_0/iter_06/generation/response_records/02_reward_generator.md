# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主信号：前向加速度，鼓励每一步增加前向速度
    acceleration = next_obs[5] - obs[5]
    accel_reward = 2.0 * acceleration

    # 稳定性约束：惩罚躯干倾斜
    posture_penalty = -0.1 * (next_obs[1] ** 2)

    # 稳定性约束：惩罚躯干角速度（抑制翻滚）
    ang_vel_penalty = -0.001 * (next_obs[7] ** 2)

    # 平滑约束：轻微惩罚垂直速度幅度，避免过激的上下抖动
    vertical_penalty = -0.01 * (next_obs[6] ** 2)

    total_reward = accel_reward + posture_penalty + ang_vel_penalty + vertical_penalty

    components = {
        "acceleration_reward": accel_reward,
        "posture_penalty": posture_penalty,
        "ang_vel_penalty": ang_vel_penalty,
        "vertical_penalty": vertical_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 设计动机：与历史尝试的本质差异
此前的最高分结构采用 **前向速度与姿态因子的乘积耦合**（`forward_vel * exp(-k*angle^2)`），并在其上附加角速度惩罚。尽管乘法耦合在早期达到了 2926 分，但后续的系数调整、加法解耦、加入垂直速度奖励或能量惩罚均未突破瓶颈。这表明乘积形式的姿态约束可能过于严厉，限制了跳跃所必需的自然躯干摆动，且与奖励垂直速度产生了冲突。

本次设计提出一个 **结构上彻底不同的假设**：以 **前向加速度** 作为主学习信号，将所有姿态和运动惩罚解耦为独立的轻量约束。核心思想是：
- 加速度奖励替代速度绝对值，赋予每一步清晰、连续的梯度，且其累积效果与终端速度等价，但每步都有学习信号；
- 姿态、角速度和垂直速度以平方惩罚独立施加，不再与前进奖励相乘，从而允许策略在跳跃时产生合理的瞬时倾斜和摆动，同时防止极端行为。

## 各组件角色

| 组件 | 角色 | 说明 |
|---|---|---|
| `acceleration_reward` | **主学习信号** | 奖励前向速度的增加量 `next_obs[5] - obs[5]`。agent 每步只要略微加速即获得正奖励，逐步累积出持续前移的策略。 |
| `posture_penalty` | 稳定约束 | 惩罚躯干角度平方，鼓励直立。采用平方形式比绝对值更平滑，且对小幅倾斜容忍度更高。 |
| `ang_vel_penalty` | 稳定约束 | 惩罚躯干角速度平方，抑制翻滚。权重极小，只在角速度异常剧烈时才产生作用。 |
| `vertical_penalty` | 平滑约束 | 惩罚垂直速度平方，抑制上下抖动过大。允许跳跃所需的垂直运动，但惩罚过度波动，避免策略陷入无效的弹跳模式。 |

## 关键设计决策与风险预判

- **不使用 `terminal_success_reward` / `terminal_failure_penalty`**：环境未提供显式的成功或失败标志，`info` 为空，无法实现。
- **无 `alive_bonus`**：函数签名无 `done` 信号，无法判断存活。
- **效率/能耗代价留到后续迭代**：本次专注于建立前向移动的基本能力，未加入动作惩罚或时间代价，避免 agent_afraid_to_move。
- **可能的 failure mode**：agent 可能通过高频小幅加速-减速循环（抖动）来获取累积的加速度奖励，此时观察 `ang_vel_penalty` 和 `vertical_penalty` 是否足够抑制这种模式，若不足，后续迭代可引入 `action_smoothness_penalty`（需记录上一步动作）或增大垂直/姿态惩罚权重。
- **后续迭代方向**：若加速度信号导致策略周期性冲刺而非平稳跳跃，可尝试混合小权重的速度绝对值奖励；若步伐不够有力，可考虑基于高度变化的引导信号。
