# reward_v1.py

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

    # Component A: 前进速度奖励乘以健康姿态门控
    forward_speed = max(0.0, horizontal_speed)  # 只奖励向前移动
    angle_factor = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    vert_factor = 1.0 / (1.0 + 2.0 * abs(vertical_speed))
    health_gate = angle_factor * vert_factor
    forward_reward = 2.0 * forward_speed * health_gate

    # Component B: 动作代价（抑制过度用力）
    action_cost = 0.01 * sum(a ** 2 for a in action)

    # Component C: 接触过渡奖励（鼓励正常的交替步态，惩罚双脚离地）
    contact_reward = 0.0
    if leg1_contact != next_leg1_contact or leg2_contact != next_leg2_contact:
        # 发生了接触状态变化
        if next_leg1_contact == 0 and next_leg2_contact == 0:
            contact_reward = -0.2   # 双脚离地（跳跃/摔倒前兆）
        elif next_leg1_contact == 1 and next_leg2_contact == 1:
            contact_reward = 0.0    # 双脚同时触地，不奖不罚
        else:
            contact_reward = 0.1    # 单脚支撑的正常步态切换

    total_reward = forward_reward - action_cost + contact_reward
    components = {
        "forward_reward_gated": forward_reward,
        "action_cost": -action_cost,
        "contact_transition_reward": contact_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

**task_family / dynamics_subtype**  
`locomotion_continuous_control` / `planar_bipedal_gait`。双足机器人在崎岖地面上持续前进，避免摔倒。

**selected reward roles**  
1. **主学习信号（progress_reward）**：使用门控后的前进速度奖励，替代简单的线性速度奖励。  
2. **稳定/健康约束（stability/health_gate）**：通过 `health_gate` 将身体倾角和垂直速度转化为速度奖励的衰减因子，使 agent 在学习前进的同时必须保持平稳姿态，无需额外施加惩罚项。  
3. **效率约束（action_cost）**：动作幅度二次惩罚，轻量抑制关节力矩浪费。  
4. **步态规范（contact_transition_reward）**：基于双足触地信号的切换事件，奖励单脚支撑的正常步态转换，惩罚双脚离地（跳跃），避免步态失调。

**role_to_signal_mapping**  
- `progress_reward` → `horizontal_speed`（obs[2]），裁剪至非负值。  
- `health_gate` → `hull_angle`（obs[0]）和 `vertical_speed`（obs[3]），用 bounded 形式 1/(1+k·|x|) 构造因子。  
- `action_cost` → `action` 各维度的 L2 范数。  
- `contact_transition_reward` → 二值触地信号 `leg_1_ground_contact`（obs[12]）和 `leg_2_ground_contact`（obs[13]），通过 `next_obs` 的对应维度检测接触变化。

**formula operators**  
- 前进奖励采用 `dense_state_signal`（线性正奖励）与 `soft_health_gate` 结合：`w * signal * gate`，其中 `gate` 由两个 `bounded_signal` 的乘积构成。  
- 动作代价使用 `quadratic_penalty`（`-w * sum(a_i**2)`）。  
- 接触过渡奖励是一个离散事件奖励，根据新接触状态使用条件逻辑，避免使用复杂的门控。

**excluded roles 及原因**  
- `terminal_success_reward`：环境未提供显式 success flag，info 为空，且奖励函数内无法感知 episode 终止，故无法实现。  
- `terminal_failure_penalty` 同理被排除；用软门控替代硬惩罚，使 agent 在接近危险时自然被抑制。  
- `energy_efficiency_penalty` 仅以极小动作代价形式出现，避免与主任务冲突；更积极的能耗优化留到后续迭代。  
- LIDAR 相关的 `preview_conditioned_reward` 暂不引入，因为 v1 先聚焦基本步态稳定性，利用前方地形属于高级阶段。

**为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
info 字段完全为空，且 `compute_reward` 签名中无法获取 done 信号。任何基于终止条件的奖励都无法在每步计算中可靠施加，因此强制排除。

**留到后续迭代的职责**  
- 准确的任务完成奖励（到达终点）或基于 episode 长度的推断。  
- 基于 LIDAR 的预判门控，使 agent 提前调整步态应对障碍。  
- 更精细的能耗优化和关节平滑性约束。

**训练后应观察的 failure modes**  
- 机器人可能学会极小步伐“溜冰”前进，以保持低倾角和高门控系数 —— 需关注触地切换奖励是否能鼓励正常步幅。  
- 过度依赖门控导致前进速度过低（health_gate 灵敏度参数可能需要调整）。  
- 若接触过渡奖励的 0.1 幅度相对于动作代价为主导，可能诱导 agent 故意抖动触地以刷分 —— 需观察接触切换频率是否异常。  
- 垂直速度门控可能抑制必要的摆动腿抬升，需留意行走姿态是否僵硬。