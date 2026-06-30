# Response Record

# Reward Revision for Env_001 - Iteration 10

## Design Changes

- **保留**: `progress_delta_reward` 作为主要导航信号，系数保持 10.0
- **保留**: `distance_anchor_reward` 作为持续接近目标的锚定奖励，系数保持 -0.5
- **削弱**: `smooth_landing_bonus` 的激活距离从 0.3 扩大到 0.5，系数从 0.2 提高到 0.5，但增加速度阈值限制（speed < 2.0），避免过早激活着陆引导干扰导航
- **修改**: 着陆引导使用更平滑的指数形式，降低对精确姿态的敏感度
- **新增**: 无新增组件，保持简洁
- **仍然不使用 terminal_success_reward / terminal_failure_penalty**: 因为 info 中仍然没有可靠的 success/failure 信号
- **下一轮重点观察**: 平均 episode length 是否增加（>70），以及 external reward 是否改善（>-400）

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
    
    # 提取接触标志
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
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
    # 修改：扩大激活距离（0.3 -> 0.5），提高系数（0.2 -> 0.5）
    # 新增：增加速度阈值限制（speed < 2.0），避免过早激活着陆引导
    # 原因：扩大激活范围让着陆引导更早介入，但通过速度阈值确保只在合理速度下激活
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    
    # 在接近目标且速度不太快时激活着陆引导
    if next_distance < 0.5 and speed < 2.0 and (left_contact > 0.5 or right_contact > 0.5):
        # 使用指数衰减形式提供更平滑的引导，降低对精确姿态的敏感度
        speed_bonus = 1.0 / (1.0 + speed * 0.5)
        angle_bonus = 1.0 / (1.0 + angle_penalty * 2.0)
        angular_vel_bonus = 1.0 / (1.0 + angular_vel_penalty * 2.0)
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
