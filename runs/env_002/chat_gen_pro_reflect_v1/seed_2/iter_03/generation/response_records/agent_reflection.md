# Response Record

# 设计理由
当前 agent 已稳定走完全程并超过目标分，但奖励函数缺乏能量消耗约束，可能导致关节力矩过高、步态低效。根据环境事实，能量最小化是附属优化目标，可进一步提升任务表现。我选择添加一个动作能耗惩罚项，用连续有界形式（动作平方和）提供每步梯度，系数设为 0.05，使预期惩罚约占主前进信号（～0.417/步）的 10%，符合校准比例，避免过度削弱前进激励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度的凸化奖励
    horizontal_vel = obs[2]
    forward_progress_reward = 1.3 * (horizontal_vel ** 2)

    # 平衡约束（角度与角速度的软惩罚，保持原有形态但当前未触发）
    hull_angle = obs[0]
    hull_angular_vel = obs[1]

    angle_threshold = 0.3
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold) ** 2

    angular_vel_threshold = 2.0
    angular_vel_penalty = -0.1 * max(0.0, abs(hull_angular_vel) - angular_vel_threshold) ** 2

    balance_penalty = angle_penalty + angular_vel_penalty

    # 健康门控：身体角度偏离越大，主奖励衰减越强
    gate_factor = max(0.0, min(1.0, 1.0 - abs(hull_angle) / 0.5))
    gated_forward_reward = forward_progress_reward * gate_factor

    # 新增加的能量消耗惩罚（Level 2 干预）
    action_power = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.05 * action_power

    total_reward = gated_forward_reward + balance_penalty + energy_penalty

    components = {
        'forward_progress_reward': forward_progress_reward,
        'gated_forward_reward': gated_forward_reward,
        'balance_angle_penalty': angle_penalty,
        'balance_angular_vel_penalty': angular_vel_penalty,
        'gate_factor': gate_factor,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=20/20, len=885.1, score_range=301~307；balance 惩罚 active_rate 0%；forward_progress 持续激活，gate_factor 均值 0.945，角度稳定。
- **behavior**: agent 已稳定走完地形并达到目标分，平衡良好，但可能采用高力矩步态。
- **signal**: 缺少能量消耗负反馈，无法诱导节能步态。
- **level**: Level 2，添加一个连续有界惩罚组件，并校准系数。
- **hypothesis**: 轻量动作能耗惩罚（约 10% 主信号）能推动步态优化，降低关节力矩，同时不破坏已习得的稳定行进行为。
- **risk**: 若惩罚系数过大，可能降低前进速度或导致步态僵硬；需通过后续验证微调。
