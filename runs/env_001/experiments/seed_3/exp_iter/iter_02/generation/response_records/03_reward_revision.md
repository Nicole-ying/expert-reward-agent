# Response Record

好的，我们来根据上一轮的训练反馈和专家诊断，对奖励函数进行一次有针对性的修订。

## 设计说明

- **保留**：`progress_delta_reward` 作为主要的密集导航信号，其系数保持不变。
- **削弱/删除**：删除了 `soft_landing_bonus`。上一轮其触发率仅为 0.5%，几乎不提供学习信号，且与 `stability_penalty` 存在冲突（鼓励稳定但几乎无法达成）。
- **修改**：将 `stability_penalty` 修改为 `landing_quality_shaping`。不再是一个纯粹的惩罚项，而是根据接近目标的程度，动态调整对速度、姿态和角速度的约束强度。当远离目标时，允许更大的速度和姿态变化以鼓励探索；当接近目标时，惩罚会平滑增加，引导飞行器进入稳定的着陆状态。
- **新增**：新增 `distance_anchor_reward`，一个基于当前距离的、非常小的负奖励（或称为生存奖励）。这为智能体提供了一个持续的信号：远离目标是不好的，即使没有明显的进度变化。这有助于缓解 `progress_delta_reward` 在某些状态下可能为零的问题。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：环境接口中仍然没有提供明确的成功或失败信号（如 `info["success"]`），因此这些组件仍然被禁止。
- **下一轮训练后应该重点观察**：
    1.  `landing_quality_shaping` 的平均值和绝对值，确认它是否在接近目标时提供了有效的引导，而不是像上一轮那样成为一个恒定的惩罚。
    2.  `distance_anchor_reward` 的平均值，确认它是否起到了持续的引导作用。
    3.  外部评估奖励（external eval reward）和平均回合长度是否有所提升，以验证是否缓解了“早期失败/坠毁”的问题。

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
    # 奖励每一步更接近目标，提供密集的导航引导
    progress_delta = current_distance - next_distance
    progress_delta_reward = 10.0 * progress_delta
    
    # ========== 组件2: 持续引导 - distance_anchor_reward ==========
    # 提供一个微弱的持续信号，鼓励保持接近目标的状态
    distance_anchor_reward = -0.1 * next_distance
    
    # ========== 组件3: 着陆质量塑造 - landing_quality_shaping ==========
    # 根据距离目标远近，动态调整对速度、姿态的约束
    # 当远离目标时，约束很弱；当接近目标时，约束逐渐增强
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    speed_penalty = speed
    
    # 使用 sigmoid-like 函数根据距离生成一个 0 到 1 之间的权重
    # 距离越近，权重越高，惩罚越强
    proximity_weight = 1.0 / (1.0 + 10.0 * next_distance)
    
    # 组合惩罚项，并由 proximity_weight 调节
    shaping_penalty = -proximity_weight * (1.0 * angle_penalty + 0.5 * angular_vel_penalty + 0.5 * speed_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + distance_anchor_reward + shaping_penalty
    
    # 构建组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "distance_anchor_reward": distance_anchor_reward,
        "landing_quality_shaping": shaping_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
