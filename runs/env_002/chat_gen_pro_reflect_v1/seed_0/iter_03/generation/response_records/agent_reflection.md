# Response Record

# 设计理由
代理已稳定走完全程并达到外部目标得分，但当前奖励中没有任何能量消耗相关的信号，而任务目标明确要求“最小化能量消耗”。证据：angle_penalty 已不再激活（active_rate=0%），平衡约束足够强；episode 均以成功终止结束。现新增一个基于动作扭矩平方和的轻微惩罚（action_penalty），鼓励使用更小的关节力完成行走，从而提升能量效率。系数 0.03 经校准后每步预期惩罚约为主前进信号的 5%~8%，不会破坏现有行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号 - 鼓励向前行走
    # ============================================================
    forward_velocity = obs[2]
    forward_reward = 1.0 * forward_velocity

    # ============================================================
    # 2. balance_maintenance: 健康约束 - 保持身体直立
    #    angle_penalty: hinge 惩罚，阈值 0.2 rad
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]

    angle_threshold = 0.2
    angle_penalty = -1.0 * max(0.0, abs(hull_angle) - angle_threshold)

    angular_vel_penalty = -0.1 * hull_angular_vel ** 2

    balance_penalty = angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. soft_health_gate: 保持原状
    # ============================================================
    gate_k = 3.0
    health_gate = 1.0 / (1.0 + gate_k * abs(hull_angle))
    gated_forward_reward = forward_reward * health_gate

    # ============================================================
    # 4. action_penalty: 能量消耗最小化 - 惩罚大扭矩
    # ============================================================
    action_penalty = -0.03 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ============================================================
    # 5. 组合
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty + action_penalty

    components = {
        'forward_reward': forward_reward,
        'health_gate': health_gate,
        'gated_forward_reward': gated_forward_reward,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty,
        'action_penalty': action_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=20/20, len=818, score=306.04；angle_penalty active_rate=0%，平衡已完美；无能量消耗信号。
- **behavior**: 代理在每轮均平稳行走至地形终点，无失败终止。
- **signal**: 缺少能量消耗最小化引导，主任务目标中明确要求“最小化能量消耗”。
- **level**: Level 2 – 新增一个基于动作平方和的连续惩罚组件。
- **hypothesis**: 轻微的动作惩罚将引导策略使用更小关节扭矩维持相同前进速度，从而降低实际能量消耗，改善外部任务表现。
- **risk**: 如果惩罚过强，可能会迫使幅度过小导致前进速度下降；当前系数保守，风险较低。
