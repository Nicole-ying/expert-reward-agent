# 设计理由
本轮仅修改一个组件：将 `angular_penalty` 从全时二次惩罚改为 **软 hinge 惩罚**，保留原有的结构（`angular_penalty + forward_reward + lateral_penalty`），不做其他变更。

**为什么改这个组件？**  
- **问题根源**：当前 `angular_penalty = -0.05 * (roll_vel² + pitch_vel² + yaw_vel²)` 对任意大小的角速度都施加惩罚，而四足行走过程中正常的摆动也会产生非零角速度。这导致负奖励持续累积，使得总奖励长期为负（score = -388），严重抑制了对前进的探索。  
- **证据**：累积记录显示 iter 8（移除角速度惩罚）取得了历史最佳 score = -6.63，而 iter 9 重新加入二次角速度惩罚后分数暴跌至 -388，说明无界的二次惩罚是拖累分数的关键因素。  
- **为什么现在不改其他组件**：前进奖励 `forward_reward` 量级合适，活性高；`lateral_penalty` 影响很小。优先修复最大的负贡献源。

**数学形式**：  
将角速度惩罚改为 **hinge** 形式，仅惩罚超出阈值（0.3 rad/s）的角速度平方：

```
roll_penalty  = max(0.0, abs(roll_vel)  - 0.3) ** 2
pitch_penalty = max(0.0, abs(pitch_vel) - 0.3) ** 2
yaw_penalty   = max(0.0, abs(yaw_vel)   - 0.3) ** 2
angular_penalty = -0.1 * (roll_penalty + pitch_penalty + yaw_penalty)
```

- **阈值 0.3 rad/s**：设定在正常步态摆动与剧烈晃动的分界处。当前角速度平均大小约 1-2 rad/s，阈值 0.3 保留了大部分必要的阻尼，但去除了对小幅度自然旋转的惩罚。  
- **系数 0.1**：按校准规则，惩罚 per‑step 应 ≤ 0.3× 主信号 per‑step（主信号 per‑step ≈ 0.91）。0.1 系数在角速度超出阈值 0.5 rad/s 时贡献约 –0.025，在恶劣条件下不会淹没前进奖励。

**预期效果**：  
agent 在保持姿态稳定时不会受到无谓惩罚，总奖励将由负转正或至少显著提高；同时 hinge 依然阻止剧烈翻滚，不会导致稳定性崩溃。

**风险**：  
阈值若过低，可能仍惩罚大部分步态，收益有限；若过高，则失去阻尼效果。本次选 0.3 为保守下限，可根据下一轮反馈进一步调优。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    body_z = obs[0]          # torso height
    quat_x = obs[2]          # quaternion x component
    quat_y = obs[3]          # quaternion y component
    body_x_vel = obs[13]     # forward velocity
    body_y_vel = obs[14]     # lateral velocity
    roll_vel  = obs[16]      # body angular velocities
    pitch_vel = obs[17]
    yaw_vel   = obs[18]

    # Compute uprightness
    uprightness = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    uprightness = max(-1.0, min(1.0, uprightness))

    # Soft health gate – height
    height_gate = (body_z - 0.25) / (0.35 - 0.25)
    height_gate = max(0.0, min(1.0, height_gate))

    # Soft health gate – uprightness
    upright_gate = (uprightness - 0.5) / (0.9 - 0.5)
    upright_gate = max(0.0, min(1.0, upright_gate))

    # Combined gate
    forward_gate = min(height_gate, upright_gate)

    # Main signal: gated forward velocity
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * forward_gate

    # Stabilisation: suppress lateral drift (unchanged)
    w_lat = 0.1
    lateral_penalty = -w_lat * (body_y_vel**2)

    # Stabilisation: damp excessive body rotation (CHANGED to hinged penalty)
    angular_thresh = 0.3
    roll_penalty  = max(0.0, abs(roll_vel)  - angular_thresh) ** 2
    pitch_penalty = max(0.0, abs(pitch_vel) - angular_thresh) ** 2
    yaw_penalty   = max(0.0, abs(yaw_vel)   - angular_thresh) ** 2
    angular_penalty = -0.1 * (roll_penalty + pitch_penalty + yaw_penalty)

    total_reward = forward_reward + lateral_penalty + angular_penalty

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失高度/直立度的前兆硬惩罚，且 angular_penalty 全时二次惩罚过重。
- **behavior**: agent 能行走（平均 len ~700，前进奖励活跃），但负奖励压倒正奖励，导致整体回报为负，抑制加速优化。
- **signal**: angular_penalty 对正常摆动也惩罚，过多负信号阻碍前进；缺少对即将跌倒的明确梯度。
- **level**: Level 2（结构变换：无界二次 → 软 hinge）
- **hypothesis**: 将角速度惩罚改为 hinge 后，小幅度旋转不再受罚，总奖励将显著上升，agent 能更专注于最大化前进速度，同时剧烈晃动仍被抑制，避免增加终止率。
- **risk**: 若阈值 0.3 过低，惩罚减少不够，分数仍负；若过高则失去稳定作用。需观察下一轮 angular_penalty active_rate 和分数变化。