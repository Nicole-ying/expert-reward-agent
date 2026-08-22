# Response Record

# 设计理由

**审计发现**：当前骨架（iter 5）使 episode length 从 160+ 跃升至 406.5，证明 contact_transition_reward 在维持存活方面的确有正向作用。但 final score 仍在 -42.42，说明 agent 可能学会了“高频率原地交替踏步”以获得每步 0.1 的接触切换奖励，而 forward speed 始终不高——这解释了为何存活变长但任务完成度未提升。10 个 LIDAR 和 hull_angular_velocity 仍未被使用，但 LIDAR 属于地形预览而非当前 exploit 的直接解药；angular_velocity 可在摔倒预判中起作用，然而已存活的 agent 对它的需求不强。**结论：信号齐全但 contact_reward 与前进目标未对齐——走 Level 2 结构变换，改造 contact_transition_reward。**

**改造内容**：将 `contact_transition_reward` 从“只要发生单脚支撑切换即奖励 0.1”改为“单脚支撑切换的奖励乘以当前 forward_speed”，使原地踏步（speed≈0）不再产生正收益。同时保留双脚离地的惩罚和双脚同时触地的不奖惩。这样 agent 必须真实前进才能获得步态激励，从根本上消除 contact exploit。

**系数校准**：
- 旧 contact 每步约 0.097，新系数 0.25 在典型 speed 0.3~0.5 时奖励 0.075~0.125，量级可比较且不超过主信号 `forward_reward_gated`（per-step ~0.34）的 0.5 倍。
- `action_cost` 极小（per-step ~0.01），无需调整。
- 双脚离地惩罚 -0.2 保持不变，发生频率低，风险可控。

**数学形式**：
```
contact_reward = 0.25 * forward_speed * I[single stance transition] - 0.2 * I[double flight]
```
其中 forward_speed = max(0, horizontal_speed)，单步态切换定义为接触状态改变且落地脚数为 1。

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

    # Component C: 接触过渡奖励 —— 与前进速度耦合，消除原地踏步 exploit
    contact_reward = 0.0
    contact_change = (leg1_contact != next_leg1_contact) or (leg2_contact != next_leg2_contact)
    if contact_change:
        if next_leg1_contact == 0 and next_leg2_contact == 0:
            contact_reward = -0.2   # 双脚离地惩罚
        elif next_leg1_contact == 1 and next_leg2_contact == 1:
            contact_reward = 0.0    # 双脚同时触地，不奖不罚
        else:
            # 单脚支撑的正常步态切换，奖励与前进速度成正比
            contact_reward = 0.25 * forward_speed

    total_reward = forward_reward - action_cost + contact_reward
    components = {
        "forward_reward_gated": forward_reward,
        "action_cost": -action_cost,
        "contact_transition_reward": contact_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: contact_transition_reward 缺少与前进目标的耦合，agent 通过原地交替踏步获利，导致存活延长但 final score 仍为负；LIDAR 等未用信号非当前核心瓶颈。
- **behavior**: agent 频繁切换支撑脚获得每步约 0.1 的奖励，前进速度提升有限，形成 exploit 循环。
- **signal**: contact_reward 与 forward_speed 独立，造成激励不对齐。
- **level**: Level 2
- **hypothesis**: 将 contact_reward 乘以 forward_speed 后，agent 只有在真正前进时才能获得步态激励，迫使其将交替步态与移动目标统一，从而提升 forward speed 和最终得分。
- **risk**: 早期探索阶段 forward_speed 较低，contact_reward 会减少，学习初期可能变慢；但 forward_reward_gated 仍提供基础前进信号，风险可控。
