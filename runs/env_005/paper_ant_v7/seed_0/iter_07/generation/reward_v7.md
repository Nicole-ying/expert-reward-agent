# 设计理由
改动了 `upright_penalty` 组件，将其从 **二值 hinge**（仅当 `body_up_z < 0.7` 时才激活）变为 **连续的二次惩罚**  
`-0.5 * (1.0 - body_up_z)**2`。  
- 原来 `max(0, 0.7 - body_up_z)` 只在较大倾斜时才给出梯度，active_rate 仅 7.3%，大量轻度不稳的步态完全没有姿态纠正信号，导致高速奔跑时倾斜逐渐累积，最终跌倒 terminated。  
- 新形式对每一个偏差都给出梯度，且越倾斜惩罚非线性加重，引导 policy 始终维持高度直立。同时系数 0.5 确保平均步均惩罚 ≪ 主前进信号的 0.3×，不会压制高速步态。  
- `(1 - body_up_z)` 不会引入死区，在完全直立时精确为 0，在小倾斜时惩罚极轻但梯度存在，预期可降低 terminated 率而不牺牲已获得的速度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控（独立正向生存奖励） ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 高度边界前兆惩罚（防止突然终止） ----------
    low_penalty = -0.3 * max(0.0, (0.25 - body_z) / 0.05)
    high_penalty = -0.3 * max(0.0, (body_z - 0.95) / 0.05)
    height_boundary_penalty = low_penalty + high_penalty

    # ---------- 主学习信号：凸化前进速度 ----------
    forward_reward = 1.0 * body_x_velocity + 0.5 * max(0.0, body_x_velocity)**2

    # ---------- 直立姿态连续惩罚（代替 hinge）----------
    upright_error = 1.0 - body_up_z
    upright_penalty = -0.5 * (upright_error ** 2)

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = (forward_reward + height_reward + height_boundary_penalty
                    + upright_penalty + action_cost)

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "height_boundary_penalty": height_boundary_penalty,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 高度边界惩罚几乎僵尸（active_rate 0.4%），但保持无害；关键缺口是缺少对每一次微小姿态偏差的持续梯度。  
- **behavior**: agent 已习得高速奔跑（~2.5 m/s），但频繁摔倒（terminated 65%），原因是姿态摇晃逐渐累积未得到纠正。  
- **signal**: 缺少对直立偏差的连续梯度，原 hinge 太稀疏。  
- **level**: Level 2  
- **hypothesis**: 每步都给出姿态偏差的微小惩罚，能使策略主动维持高度直立，减少倾斜累积导致的终止，从而提高存活率同时保持速度。  
- **risk**: 系数 0.5 的二次惩罚在平均姿态下极轻（~0.001/步），可能不足以显著改变行为；若效果不显，后续可适度提高系数或改用线性形式。