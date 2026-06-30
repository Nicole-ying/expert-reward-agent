# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_delta_reward**：保留作为主要学习信号，系数保持10.0不变。这是最有效的密集导航引导。
- **distance_anchor_reward**：保留作为持续接近目标的锚定激励，系数保持-0.5不变。
- **smooth_landing_bonus**：保留着陆引导机制，但进行关键修改。

### 削弱或删除了什么
- **无删除**：所有组件都保留，因为上一轮没有证据表明某个组件有害。

### 修改了什么
- **smooth_landing_bonus**：将激活阈值从0.8降低到0.6，系数从1.0降低到0.5。原因：
  - 外部评估奖励为-38.66，平均episode长度为1000，说明智能体可能过早进入着陆模式但无法稳定着陆。
  - 过大的着陆奖励可能干扰了主要的导航信号，导致智能体在远处就尝试着陆。
  - 缩小激活范围并降低系数，让着陆引导只在真正接近目标时生效，且不主导总奖励。

### 新增了什么
- **无新增**：保持组件数量不变，专注于调整现有组件的参数。

### 为什么仍然不使用 terminal_success_reward / terminal_failure_penalty
- 仍然没有显式的成功/失败信号可用。info字段不可靠，且环境合约明确禁止使用这些信号。

### 下一轮训练后应该重点观察什么
- 外部评估奖励是否提升（从-38.66向更高值移动）。
- 平均episode长度是否变化：如果变短，说明智能体可能提前坠毁；如果保持1000，说明仍在探索。
- smooth_landing_bonus的触发频率和贡献比例，确保它不主导总奖励。

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
    # 修改：缩小激活范围（0.8 -> 0.6），降低系数（1.0 -> 0.5）
    # 原因：避免过早激活着陆引导干扰主要导航信号
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    
    # 只在更接近目标时激活着陆引导
    if next_distance < 0.6:
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