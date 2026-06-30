好的，我们来根据上一轮的反馈和诊断，对奖励函数进行一次有针对性的修订。

## 修订分析

1.  **上一轮结构**：`progress_delta_reward` + `distance_anchor_reward` + `landing_quality_shaping`。
2.  **训练反馈**：外部评估奖励为 -105.06，平均回合长度 70.7 步。诊断结果为 `early_failure_or_crash`。
3.  **问题诊断**：尽管上一轮已经削弱了 `landing_quality_shaping`，但训练结果仍然显示智能体过早失败。这说明：
    - `progress_delta_reward` 提供的导航信号可能不够强，或者被其他惩罚项抵消。
    - `landing_quality_shaping` 即使在削弱后，其惩罚性可能仍然在早期阶段（当智能体远离目标时，`proximity_weight` 很小，但速度/角度惩罚仍然存在）对智能体造成了负面影响，导致其无法稳定探索。
    - `distance_anchor_reward` 的 -0.1 系数可能过小，无法提供有效的持续引导。
4.  **修订策略**：
    - **保留**：`progress_delta_reward` 作为核心的密集导航信号。
    - **削弱/删除**：删除 `landing_quality_shaping`。在智能体能够稳定地接近目标之前，任何对姿态和速度的惩罚都可能加剧早期失败。我们将在后续迭代中，当智能体能够可靠地到达目标附近时，再重新引入。
    - **修改**：将 `distance_anchor_reward` 的系数从 -0.1 提高到 -0.5，使其成为一个更强的持续引导信号，鼓励智能体保持接近目标的状态，而不是漫无目的地探索。
    - **新增**：新增一个 `smooth_landing_bonus`。这是一个条件性的、平滑的奖励，用于在智能体接近目标时（`next_distance < 0.5`）提供正向激励，奖励其低速度和稳定姿态。这比惩罚更温和，能引导智能体学习更好的着陆行为，而不会在早期阶段造成惩罚主导。
    - **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：环境接口中仍然没有提供明确的成功/失败信号，因此不能使用。
5.  **下一轮观察重点**：
    - 外部评估奖励是否从 -105 提升到 -50 或更高。
    - 平均回合长度是否从 70 步增加到 150 步以上，表明智能体存活时间更长。
    - `smooth_landing_bonus` 的触发频率和平均值，以判断其是否有效。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取位置信息（相对于目标着陆台）
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # 提取速度信息
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # 提取姿态信息
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 计算当前位置到目标的距离
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # ========== 组件1: 主学习信号 - progress_delta_reward ==========
    # 保留：奖励每一步更接近目标，提供密集的导航引导
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # ========== 组件2: 持续引导 - distance_anchor_reward ==========
    # 修改：提高系数，使其成为一个更强的持续引导信号
    distance_anchor_reward = -0.5 * next_distance
    
    # ========== 组件3: 平滑着陆奖励 - smooth_landing_bonus ==========
    # 新增：当智能体接近目标时，奖励其低速度和稳定姿态
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    
    # 仅在接近目标时激活
    if next_distance < 0.5:
        # 奖励低速度、小角度和低角速度
        speed_bonus = max(0.0, 1.0 - speed)
        angle_bonus = max(0.0, 1.0 - angle_penalty)
        angular_vel_bonus = max(0.0, 1.0 - angular_vel_penalty)
        smooth_landing_bonus = 0.5 * (speed_bonus + angle_bonus + angular_vel_bonus)
    else:
        smooth_landing_bonus = 0.0
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + distance_anchor_reward + smooth_landing_bonus
    
    # 构建组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "distance_anchor_reward": distance_anchor_reward,
        "smooth_landing_bonus": smooth_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```