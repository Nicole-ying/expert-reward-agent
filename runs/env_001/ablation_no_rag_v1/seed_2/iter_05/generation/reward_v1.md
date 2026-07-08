# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar-lander-like task.
    
    Args:
        obs, next_obs: shape (8,) arrays with:
            0: x_position (relative to landing pad)
            1: y_position (relative to landing pad height)
            2: x_velocity
            3: y_velocity
            4: body_angle
            5: angular_velocity
            6: left_support_contact (0/1)
            7: right_support_contact (0/1)
        action: discrete 0..3 (unused in this version)
        info: empty dict (do not use)
        training_progress: unused
    """
    # -- Extract state variables --
    # Current state
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    # Next state
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]

    # -- Component 1: Potential-based progress + stability (main learning signal) --
    # Potential combines distance to target and velocity magnitude.
    # Weight on velocity makes the agent prefer slow, safe approach.
    w_dist = 1.0
    w_vel  = 0.3   # scaling for linear velocity relative to distance
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next

    # Positive when potential decreases (agent moves closer to target and/or slows down)
    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty (stability constraint) --
    # Penalise large body angle and angular velocity to promote upright landing.
    k_angle = 0.5
    k_angvel = 0.1
    # Use squared terms for smooth gradient
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Total reward --
    total_reward = potential_diff + angle_penalty

    # -- Components dict (only the two that are summed) --
    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty)
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件（2个）

1. **`potential_diff`** — 主学习信号（兼具 progress 与 stability）  
   - 通过势能差值塑造：`potential = distance_to_target + w_vel * (|vx|+|vy|)`。  
   - 每步奖励势能的减少，即同时鼓励 **靠近目标** 和 **减小速度**。  
   - 这是一个统一的稠密信号，避免了“先冲过去再减速”的冲突，也规避了独立 progress 与 stability 项之间的权重争夺。  
   - `w_vel = 0.3` 使速度项与距离项量级匹配（距离约 1~5 单位，速度约 0~3 单位）。

2. **`angle_penalty`** — 稳定/安全约束  
   - 惩罚非零的机体倾角与角速度，鼓励平稳姿态。  
   - 使用平方项提供平滑梯度，权重较小（`k_angle=0.5`, `k_angvel=0.1`），以免压制飞行机动。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境 **没有显式成功/失败标志**（`info` 为空），无法可靠检测任务完成。  
- 根据设计原则，当 `explicit_success_flag_available=false` 时不使用终端成功奖励，当 `explicit_failure_flag_available=false` 时不使用终端失败惩罚。

## 与之前 v1 结构的本质区别
- 之前结构：`progress_reward + stable_landing_reward`（分离式），已被证明停滞在负分。
- 新设计将 progress 与 stability 融合进一个 **potential‑based shaping** 信号，其数学机制不同于简单的加和：它通过势能差的定义，强制 agent 在接近目标的同时减速，而不是作为两个独立可分的目标分别优化。  
- 角度惩罚仅作为轻量辅助，不做为主信号的一部分。

## 哪些组件留到后续迭代
- **接触/着陆代理奖励**：当前版本未使用，因为缺乏终止反馈时持续给予接触奖励易被利用（原地不动刷分）。若后续获得终止信号或环境可动态调整，可加入基于多条件连续组合的 soft proxy。  
- **动作效率/燃料代价**：尚未加入，待任务成功后再优化节能。  
- **门控/动态课程**：留到后续更复杂阶段。

## 训练后应观察的 failure mode
- **低速徘徊**：agent 因速度惩罚过重而不敢靠近目标，长期悬停远处；可适当降低 `w_vel`。  
- **高速撞击**：角度惩罚或速度项不足，agent 仍以高速颠倒姿态冲向平台；需调高 `w_vel` 或 `k_angle`。  
- **稳步但不落地**：agent 可能在平台上方不断下降又突然上升（离散动作导致），因为缺少落地激励。此时可考虑引入接触代理。  
- **奖励尺度崩塌**：`potential_diff` 与 `angle_penalty` 量级悬殊，导致一项主导；需根据训练日志调整权重，确保两者对总奖励贡献大致可比。