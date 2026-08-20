# 设计理由
我将 `forward` 中的直立门控从 `max(0, 1 - 2 * upright_error)`（即要求 `body_up_z > 0.5` 才有奖励）替换为连续的 `max(0, body_up_z)`。原门控激活区间过窄，导致机器人在稍有倾斜时前向奖励立即归零，直立学习的梯度被切断。新权重在 `body_up_z ∈ (0, 1]` 时连续给出与直立程度成比例的奖励，而倒立时不奖励也不惩罚，保持单位量级不变。高度惩罚维持原阈值，以集中干预姿态引导问题。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]

    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)   # 连续权重，替换原 upright_gate
    
    w_forward = 1.0
    forward = w_forward * body_x_vel * upright_factor

    w_height = 5.0
    height_pen_low = max(0.0, 0.25 - body_z)
    height_pen_high = max(0.0, body_z - 0.95)
    height_pen = -w_height * (height_pen_low + height_pen_high)

    total_reward = forward + height_pen

    components = {
        "forward": forward,
        "height": height_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated 12/20（多数失败），score=-521，forward sum mean 56.4 但 upright 门控导致大量步无梯度。
- **behavior**: 机器人频繁跌倒，少数 episode 能存活但前进效率低，整体奖励极差。
- **signal**: 直立门控激活区间过窄（body_up_z > 0.5 才有奖励），切断了倾斜恢复过程的梯度，使 agent 难以学会稳定直立和前进。
- **level**: Level 2（数学结构变换：稀疏/窄门控→连续有界因子）
- **hypothesis**: 用连续直立权重替换窄门控后，agent 在倾斜状态下仍能获得比例奖励，梯度连续，能更快学会在保持直立的前提下提高前向速度。
- **risk**: 若 body_up_z 长期较低，forward 奖励会很小，可能陷入低速度平台；但不会比原来完全无奖励更差，且随直立改善会自然提升。