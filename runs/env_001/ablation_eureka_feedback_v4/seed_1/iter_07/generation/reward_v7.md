# 设计理由
将无条件的 contact_bonus 替换为与距离、速度、姿态耦合的 success_reward，使用指数衰减因子和接触标志的乘积。这样仅在腿接触且机体稳定靠近中心时才给予大额奖励，防止 agent 在远距离或高速下通过短时接触刷分。速度惩罚、朝向惩罚和线性距离惩罚保持不变，保持整体梯度连续。权重校准：成功状态 per-step 约 +20，显著强于距离惩罚（per-step 约 -0.5*dist），确保学会到达目标后稳定停靠。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # 接近奖励（线性距离惩罚，避免平方过陡）
    proximity_reward = -0.5 * dist

    # 速度惩罚（远处允许高速，近处要求减速）
    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    # 姿态惩罚
    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    # 成功状态奖励：双接触 + 稳定（距离近、速度小、角度小）
    contact = lc * rc
    success_factor = (2.718281828 ** (-dist / 0.5)) * (2.718281828 ** (-speed / 0.3)) * (2.718281828 ** (-abs(angle) / 0.2))
    success_reward = 20.0 * contact * success_factor

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + success_reward

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'success_reward': success_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: contact_bonus 均值高达 398 但 episode 平均得分仅 34.67，全部 20 episode 提前终止，表明 agent 可能通过不稳定接触刷分但 crash 失败。
- **behavior**: agent 频繁获得腿部接触奖励，但位置较远、速度较高，导致 proximity 惩罚较大，最终在短时间内 crash 终止，未能稳定停靠。
- **signal**: contact_bonus 作为独立稀疏奖励缺少稳定性条件，使 agent 学会在任何状态下追求短时接触而忽视安全着陆。
- **level**: Level 3（相同骨架多次迭代未稳定超越 best，且做过 Level 2 修改，触发重建）。
- **hypothesis**: 将接触奖励与距离、速度、姿态的连续衰减耦合，仅在机体稳定靠近目标时才给予强信号，能够引导稳定停靠而非 risky 接触。
- **risk**: 成功奖励在未接触时很小，初期可能仍依赖 proximity 和速度惩罚学习；若探索不足可能陷入局部最优，但速度惩罚和距离 shaping 应能提供足够梯度。