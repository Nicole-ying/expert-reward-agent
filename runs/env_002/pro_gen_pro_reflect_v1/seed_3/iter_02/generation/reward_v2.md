# 设计理由
修改 `forward_progress` 组件，将其从单纯的线性速度奖励转变为**姿态健康门控**形式：`forward_vel * health`。这样前进奖励的大小直接受姿态健康度调制——只有当本体保持在近直立状态时，速度才能充分转化为奖励。原独立的 `tilt_penalty` 保留，形成双重稳压（门控降增益 + 惩罚负反馈），但门控作为主信号的乘法调制，能更直接地约束速度驱动力向稳定方向收敛。

**数学形式**：`health = 1.0 / (1.0 + 10.0 * hull_angle²)`，在零倾斜角时恒为 1，随倾角增大平滑衰减到 0，避免乘积塌缩的同时提供连续梯度。系数 `10.0` 使倾角约 0.3 rad 时 `health ≈ 0.5`，对应于终止（摔倒）边界的 60‑80% 附近开始显著压制 reward。

**系数校准**：保持 `forward_progress` 系数 `1.0`，与上一轮零倾角等效；`tilt_penalty` 系数不变。因为门控会降低偏离直立时的整体奖励尺度，但不会超过原始量级，不会带入新的缩放爆炸风险。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity, gated by posture health
    forward_vel = next_obs[2]                 # positive = moving forward
    hull_angle = next_obs[0]                  # radians, 0 = vertical
    health = 1.0 / (1.0 + 10.0 * (hull_angle ** 2))   # posture gate
    forward_progress = 1.0 * forward_vel * health

    # Stability: keep torso upright (secondary penalty, kept as safety net)
    tilt_penalty = -5.0 * (hull_angle ** 2)

    # Stability: suppress rotational jitter
    hull_angvel = next_obs[1]                 # rad/s
    angular_velocity_penalty = -1.0 * (hull_angvel ** 2)

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + tilt_penalty + angular_velocity_penalty + action_cost

    components = {
        'forward_progress': forward_progress,
        'tilt_penalty': tilt_penalty,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前漏掉了通过姿态直接约束主进度信号的机制，且对摔倒前兆（较大倾斜）的惩罚过弱，占比仅 1.3%。
- **behavior**: agent 以较高步长（1010 步）运行，主要利用前进速度最大化回报，但偶尔摔倒导致平均分数无法突破 217。
- **signal**: 缺少一个将“保持直立”与“获得速度奖励”强耦合的信号，现有独立惩罚太轻，不足以抑制探索中摔倒的风险。
- **level**: Level 2（结构变换：将绝对速度 reward 改造为 `forward_vel * health_gate`，引入连续有界调制因子）
- **hypothesis**: 将前进奖励与姿态健康直接乘积耦合，迫使 agent 为获得高速奖励而主动维持直立，从而在速度与稳定性之间自动平衡，减少摔倒，提升平均得分。
- **risk**: 门控过强可能过早收敛到保守步态，速度提升受限；若倾斜惩罚仍保持原有强度，可能出现双重压制，必要时后续可调降 `tilt_penalty` 系数。