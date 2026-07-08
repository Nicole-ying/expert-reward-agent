# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant next observations
    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    horizontal_velocity = next_obs[2]
    vertical_velocity = next_obs[3]

    # Weights
    w_forward = 1.0
    w_angle = 1.0
    w_angvel = 0.2
    w_vert = 1.0

    forward_reward = w_forward * horizontal_velocity
    stability_penalty = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_velocity)
        - w_vert * (vertical_velocity ** 2)
    )

    total_reward = forward_reward + stability_penalty
    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 使用的奖励组件及其角色

- **forward_reward**（主学习信号）：直接奖励沿前进方向的速度（`next_obs[2]`）。给予每步稠密的梯度，引导机器人获得向前移动的策略。若速度为负（后退），该分量自然为负，抑制倒退行为。
- **stability_penalty**（稳定/安全约束）：惩罚身体倾斜角（`next_obs[0]`）、高角速度（`next_obs[1]`）和不必要的垂直速度（`next_obs[3]`）。其作用是让机器人学习在前进的同时保持躯干直立、动作平滑，降低摔倒风险（fast‑then‑fail）。该组件将多个稳定相关信号合并为一个，避免过度拆分。

V1 不引入额外的存活奖励、能耗惩罚或步态约束，因为核心目标只需要“稳定地前进”，更复杂的优化留给后续迭代。

## 2. 未使用 terminal_success_reward / terminal_failure_penalty 的原因

- `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，环境没有提供任何可用的成功或失败标志（`info` 为空字典）。
- 即使知道“到达地形尽头”或“身体倒地”会终止 episode，我们也无法在奖励函数中可靠区分这两种终止原因，强行引入终止后奖励/惩罚会引入错误的梯度或无法正确解释终止信号。
- 因此 V1 完全依赖稠密的运动信号，通过稳定性惩罚间接抑制导致倒地的行为，不依赖稀疏的终止事件。

## 3. 未纳入 V1 但后续迭代可考虑的部分

- **energy_penalty**：惩罚动作扭矩的大小，降低能耗。V1 阶段优先学会稳定行走，过早加入能耗约束可能导致 agent 不敢动。
- **action_smoothness_penalty**：惩罚动作的突变，需要访问历史动作。当前接口未提供上一步动作，需在后续阶段由外部 wrapper 提供或使用差分 `action` 与 `obs` 的变化，复杂度较高，留待 V2/ V3。
- **步态对称性 / contact heuristic**：利用 `leg1_contact`、`leg2_contact` 鼓励交替支撑，例如惩罚双脚同时离地。此信号可作为 V2 的额外稳定/效率信号。
- **地形预适应（lidar）**：通过前方雷达读数引导避障或抬腿，V1 不引入以免过早复杂化。
- **gated / dynamic curriculum**：多阶段或自适应权重，明显超出 V1 范围。

## 4. 预期训练后需要观察的 failure mode

- **fast‑then‑fail**：如果 `forward_reward` 的权重相对稳定性惩罚过大，机器人可能倾向于快速前冲然后摔倒。此时应增大 `w_angle` 或 `w_vert`，或调整稳定性惩罚的数学形态（如使用乘积形式将速度奖励与直立因子耦合）。
- **hover / stand_still**：如果稳定性惩罚过强（尤其是对角速度的惩罚），机器人可能选择原地站立不动以获取零惩罚但几乎没有正向奖励。此时应适当减小 `w_angvel`，或确保 `forward_reward` 足够有吸引力。
- **过度跳跃**：若 `w_vert` 太小，机器人可能通过小幅跳跃获得额外前向速度，导致能耗高且步态不自然。可动态调整 `w_vert`。
- **后退逃逸**：由于 `forward_reward` 允许负值，策略暂时后退也能获得负奖励，但整体应导向前进。若出现长期后退，可能需要改为 max(0, velocity) 的形式，但 V1 先保持线性以便梯度更平滑。

所有权重目前为初始建议值，实际使用需根据环境的物理量级和训练曲线进行调整。