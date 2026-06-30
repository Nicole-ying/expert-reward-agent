# Reward Revision for Env_001 - Iteration 6

## Design Changes

**保留：**
- `progress_delta_reward`：主学习信号，奖励每一步更接近目标，系数保持10.0
- `distance_anchor_reward`：持续引导信号，系数保持-0.5
- `smooth_landing_bonus`：接近目标时的平滑着陆奖励

**削弱：**
- 无

**修改：**
- 将 `smooth_landing_bonus` 的激活阈值从 `next_distance < 0.5` 改为 `next_distance < 0.8`，扩大激活范围，让智能体更早开始学习着陆姿态控制
- 将 `smooth_landing_bonus` 的系数从0.5提高到1.0，增强着陆阶段的引导信号强度

**新增：**
- 无

**为什么仍然不使用 terminal_success_reward / terminal_failure_penalty：**
- 仍然没有可靠的显式成功/失败信号可用，info 字段不可靠

**下一轮训练后应该重点观察：**
- `smooth_landing_bonus` 的触发频率和平均值，确认激活范围扩大后是否有效引导着陆
- 外部评估奖励的均值和最大值，看是否有所提升
- 平均回合长度是否变化，反映智能体是否更早完成着陆

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
    # 保留：提供持续的接近目标激励
    distance_anchor_reward = -0.5 * next_distance
    
    # ========== 组件3: 平滑着陆奖励 - smooth_landing_bonus ==========
    # 修改：扩大激活范围（0.5 -> 0.8），提高系数（0.5 -> 1.0）
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    
    # 在更广的范围内激活着陆引导
    if next_distance < 0.8:
        # 奖励低速度、小角度和低角速度
        speed_bonus = max(0.0, 1.0 - speed)
        angle_bonus = max(0.0, 1.0 - angle_penalty)
        angular_vel_bonus = max(0.0, 1.0 - angular_vel_penalty)
        smooth_landing_bonus = 1.0 * (speed_bonus + angle_bonus + angular_vel_bonus)
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