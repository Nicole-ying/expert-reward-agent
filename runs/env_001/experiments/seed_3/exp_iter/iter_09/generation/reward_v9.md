# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_delta_reward**：保留作为主要学习信号，系数保持10.0，提供密集的导航引导。
- **distance_anchor_reward**：保留作为持续接近目标的激励，系数保持-0.5。
- **smooth_landing_bonus**：保留作为接近目标时的着陆引导，但进一步调整激活条件和系数。

### 削弱或删除了什么
- **smooth_landing_bonus 的激活距离**：从0.4缩小到0.3，避免过早激活着陆引导干扰主要导航信号。
- **smooth_landing_bonus 的系数**：从0.3降低到0.2，进一步降低着陆引导对整体奖励的影响。

### 修改了什么
- **smooth_landing_bonus 的激活条件**：在距离<0.3时激活，并增加对接触标志的检查（left_support_contact或right_support_contact为1时激活），确保只在真正接近着陆时提供引导。
- **smooth_landing_bonus 的计算**：从速度、角度、角速度的线性组合改为更平滑的指数衰减形式，提供更连续的引导。

### 新增了什么
- **接触标志检查**：在smooth_landing_bonus中增加对next_obs[6]和next_obs[7]的检查，确保只在有接触时激活着陆引导，避免在悬停时错误激活。

### 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty
- 环境仍然没有提供明确的成功/失败信号（如info['success']或info['failure']），使用这些奖励会导致训练不稳定或误导。

### 下一轮训练后应该重点观察什么
- **mean_eval_reward**：是否从-9.00提升到正值区域。
- **mean_episode_length**：是否从1000.00下降，表明智能体开始更早完成或失败。
- **progress_delta_reward 的分布**：是否保持正均值，表明智能体在持续接近目标。
- **smooth_landing_bonus 的触发率**：是否在接近目标时稳定触发，提供着陆引导。

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
    # 修改：进一步缩小激活范围（0.4 -> 0.3），降低系数（0.3 -> 0.2）
    # 新增：增加接触标志检查，确保只在有接触时激活
    # 原因：避免过早激活着陆引导干扰主要导航信号，同时确保只在真正着陆时提供引导
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    
    # 只在非常接近目标且有接触时激活着陆引导
    if next_distance < 0.3 and (left_contact > 0.5 or right_contact > 0.5):
        # 使用指数衰减形式提供更平滑的引导
        speed_bonus = 1.0 / (1.0 + speed)
        angle_bonus = 1.0 / (1.0 + angle_penalty)
        angular_vel_bonus = 1.0 / (1.0 + angular_vel_penalty)
        smooth_landing_bonus = 0.2 * (speed_bonus + angle_bonus + angular_vel_bonus)
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