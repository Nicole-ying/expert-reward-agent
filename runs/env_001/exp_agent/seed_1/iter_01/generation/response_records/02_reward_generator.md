# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress delta（到目标垫中心的距离减少量）
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    d_prev = (x_prev**2 + y_prev**2) ** 0.5
    d_next = (x_next**2 + y_next**2) ** 0.5
    progress_delta = d_prev - d_next

    # 稳定约束：惩罚速度、角度与角速度（作用于 next_obs）
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.1
    w_angle = 0.5
    w_angvel = 0.1
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # 任务完成软代理：接近中心 + 低速 + 水平 + 双脚着地
    near = (d_next < 0.1)
    slow = (abs(x_vel) < 0.1 and abs(y_vel) < 0.1)
    level = (abs(body_angle) < 0.1)
    both_feet = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_bonus_raw = 1.0 if (near and slow and level and both_feet) else 0.0
    soft_landing_bonus = 0.5 * soft_landing_bonus_raw

    total_reward = progress_delta - stability_penalty + soft_landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色
- **progress_delta**：主学习信号。基于每一步到目标垫中心距离的减少量，给出连续、密集的引导。该信号是推动主体靠近目标的核心动力。
- **stability_penalty**：稳定/安全约束。惩罚过高的线速度、姿态倾斜与角速度，促使主体在接近目标时平滑减速并保持水平，为安全着陆创造条件。权重设置较轻，以避免过度束缚探索。
- **soft_landing_bonus**：任务完成软代理。仅在满足“中心附近、低速、角度水平、双脚接触”四条条件时触发一个较小的奖励，向智能体传递“最终着陆状态”的信号，但不会伪造成功标志（不能单独作为成功判断）。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因
环境提供的 `info` 字典为空，没有显式的 `success` 或 `failure` 标志。按照“如果 explicit_success/failure_flag_available=false，不将 terminal 奖励作为 v1 核心项”的原则，本版本不依赖任何外部终止原因。

## 留到后续迭代的组件
- **energy/action penalty**：当前 v1 未加入，以免早期训练中智能体因害怕动作代价而不敢移动。待 progress 和 stability 训练稳定后，可逐步引入以优化燃料消耗。
- **time penalty**：在后期可加入轻微的时间惩罚，鼓励更快的完成。
- **terminal success/failure reward**：一旦环境包装器在 info 中暴露确切的成功/失败标志，可添加一次性的成功奖励和失败惩罚，进一步强化终点行为。
- **curriculum 或 gated reward**：若出现高频不安全行为（如高速撞地），可考虑安全门控，当前版本保持简单。

## 训练后应观察的 failure modes
1. **悬停在目标上方但不下降**：progress_delta 可能在垂直方向表现为负值（高度降低）而水平方向进步很快时智能体可能过早获得高奖励，需要检查垂直速度是否被有效抑制。
2. **高速撞击目标垫**：若 stability_penalty 权重不足，智能体可能虽然接近中心但未减速，导致接触时速度过大，建议增大稳定性惩罚或调整阈值。
3. **利用 soft_landing_bonus 的条件漏洞**：例如通过一侧脚接触、另一侧悬空、或反复略微进出阈值来刷奖励。需要监控 near/slow/level/feet 的触发频率，若异常可收紧阈值或降低 bonus 权重。
4. **完全不敢使用主引擎**：若后期加入能量惩罚，可能出现过度保守。在 v1 中暂不涉及，但作为未来考量。
