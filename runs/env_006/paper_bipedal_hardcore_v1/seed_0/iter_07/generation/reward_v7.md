# 设计理由

**第 0 步审计结论**：终止模式以摔倒为主（terminated 18/20，有 5 个早夭 episode），但本轮分数和长度暴跌的根因不在摔倒，而在 `contact_transition_reward` 与前进速度的**耦合**。引入 `0.25 * forward_speed` 后，早期探索或低速阶段步态切换奖励近乎为零，正向激励消失，导致整体奖励坍缩、学习退化（分数从 -42.42 跌至 -62.76，长度从 406 降至 359）。未使用的 `hull_angular_velocity` 和 LIDAR 是后续可用的信号缺口，但当前最优先的干预是**恢复 contact 奖励的独立性**，让步态激励不再被速度压制。

**修改的组件**：`contact_transition_reward`（Component C）。  
**变换类型**：Level 2 结构变换——**去掉与前进速度的乘法耦合**，将单脚支撑的步态切换奖励恢复为固定值 `0.25`，双脚离地惩罚保持 `-0.2`，双脚同时触地仍不奖不罚。这一改动撤回了 iter 6 的失败改动，回到 iter 5 已验证有效的独立步态奖励形式。同时保留 `forward_reward_gated` 的健康门控，继续抑制原地踏步 exploit（因为无前进速度时 forward 奖励被抑制，agent 无法通过原地踢腿获得有意义的总奖励）。

**系数校准**：  
- `contact_reward` 单脚支撑奖励 `+0.25` per 切换事件，约为前进奖励 (`2.0 * 0.2 * 0.5 = 0.2`) 的 1.25×，属于合理水平。  
- 双脚离地惩罚 `-0.2` 适中，不会超过主信号 per-step。  
- 总惩罚负担仅来自 `action_cost`（约 -0.016 per-step），远小于前进奖励，满足 ≤0.3× 约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取信号
    hull_angle = obs[0]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]
    next_leg1_contact = next_obs[12]
    next_leg2_contact = next_obs[13]

    # Component A: 前进速度奖励乘以健康姿态门控（未变）
    forward_speed = max(0.0, horizontal_speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    vert_factor = 1.0 / (1.0 + 2.0 * abs(vertical_speed))
    health_gate = angle_factor * vert_factor
    forward_reward = 2.0 * forward_speed * health_gate

    # Component B: 动作代价（未变）
    action_cost = 0.01 * sum(a ** 2 for a in action)

    # Component C: 接触过渡奖励 —— 恢复为独立于前进速度的固定奖励
    contact_reward = 0.0
    contact_change = (leg1_contact != next_leg1_contact) or (leg2_contact != next_leg2_contact)
    if contact_change:
        if next_leg1_contact == 0 and next_leg2_contact == 0:
            contact_reward = -0.2   # 双脚离地惩罚
        elif next_leg1_contact == 1 and next_leg2_contact == 1:
            contact_reward = 0.0    # 双脚同时触地，不奖不罚
        else:
            # 单脚支撑的正常步态切换，不再与前进速度耦合
            contact_reward = 0.25

    total_reward = forward_reward - action_cost + contact_reward
    components = {
        "forward_reward_gated": forward_reward,
        "action_cost": -action_cost,
        "contact_transition_reward": contact_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止多为摔倒，但本轮退化主因是 contact 奖励与速度耦合导致奖励坍缩；LIDAR/角速度信号缺口有待后续利用。
- **behavior**: agent 因步态切换奖励依赖前进速度，在低速/探索期失去激励，分数和长度大幅倒退。
- **signal**: contact 奖励的乘积耦合损伤了早期步态学习信号，应回归独立正向激励。
- **level**: Level 2
- **hypothesis**: 恢复 contact_reward 的独立性后，步态切换激励回归，配合 forward_reward 引导前进，分数应回升至 iter 5 水平附近，同时健康门控可抑制原地踏步 exploit。
- **risk**: 无速度门控时，agent 可能在原地踢脚获取 contact 奖励，但 `forward_reward_gated` 会强力抑制无前进的状态，大概率不会形成 exploit；若出现，后续可加入角速度 gate 加强约束。