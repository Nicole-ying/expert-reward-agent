# Reward Revision

## 设计说明

### 相比上一轮，保留了什么
- **progress_reward**：核心学习信号，权重保持10.0，证据显示nonzero_rate接近1.0，是唯一稳定的正向信号。
- **distance_reward**：保留-0.1 * next_dist作为距离锚点，防止agent漂移。
- **smooth_landing_shaping**：保留接近引导结构，但调整权重和计算方式。

### 削弱或删除了什么
- **stability_penalty**：从上一轮进一步削弱，将angle_penalty从0.05降至0.02，angular_vel_penalty从0.025降至0.01，speed_penalty从0.025降至0.01。证据显示stability_penalty均值为-0.011，虽然绝对值不大，但外部评估均值为-5.48，说明agent可能过早崩溃或无法有效接近目标，过度惩罚稳定性会抑制探索。

### 修改了什么
- **smooth_landing_shaping**：从1.5提升至3.0，并调整计算逻辑。将near_target_factor的阈值从2.0改为1.5（更早激活接近引导），speed_comfort的阈值从1.0改为0.8（更严格的速度奖励），angle_comfort的阈值从0.5改为0.3（更严格的姿态奖励）。同时增加contact条件：当左右脚任一接触且距离较近时，给予额外着陆奖励。这样在接近目标时，agent会同时追求低速、稳定姿态和接触，形成更明确的着陆引导。

### 新增了什么
- **contact_landing_bonus**：当距离<0.5且任一接触脚为True时，给予额外奖励。这是基于"guarded landing proxy"原则：near target + low speed + stable angle + contact。但为了避免稀疏性，这个bonus是连续的（基于speed_comfort和angle_comfort的乘积），而不是硬性的0/1。

### 为什么仍然不使用terminal_success_reward / terminal_failure_penalty
- 环境仍然没有提供显式的success/failure信号。info字段不可靠，obs中也没有明确的终止标志。使用terminal奖励会导致训练不稳定，因为agent无法区分"成功着陆"和"随机终止"。

### 下一轮训练后应该重点观察什么
- **progress_reward的均值**：是否保持在0.01-0.03之间，nonzero_rate是否仍接近1.0。
- **smooth_landing_shaping的均值**：是否从接近0变为正值，说明agent开始学习接近目标。
- **contact_landing_bonus的触发率**：是否在episode后期出现，说明agent确实在尝试着陆。
- **外部评估reward**：是否从-5.48提升至-3或更高，说明整体策略改善。
- **mean_episode_length**：是否仍为1000，说明agent没有过早崩溃。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留，权重不变)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定/安全约束：stability_penalty (进一步削弱权重)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.02 * abs(next_body_angle)      # 从0.05降至0.02
    angular_vel_penalty = 0.01 * abs(next_ang_vel)   # 从0.025降至0.01
    speed_penalty = 0.01 * speed                     # 从0.025降至0.01
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 平滑接近与着陆引导：smooth_landing_shaping (增强并细化)
    near_target_factor = max(0.0, 1.0 - next_dist / 1.5)  # 距离越近，因子越大，阈值从2.0改为1.5
    speed_comfort = max(0.0, 1.0 - speed / 0.8)           # 速度越低，奖励越高，阈值从1.0改为0.8
    angle_comfort = max(0.0, 1.0 - abs(next_body_angle) / 0.3)  # 姿态越正，奖励越高，阈值从0.5改为0.3
    smooth_landing_shaping = 3.0 * near_target_factor * (0.5 * speed_comfort + 0.5 * angle_comfort)  # 权重从1.5提升至3.0
    
    # 4. 接触着陆奖励：contact_landing_bonus (新增，基于guarded landing proxy)
    contact = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    near_target = next_dist < 0.5
    if contact and near_target:
        contact_bonus = 2.0 * speed_comfort * angle_comfort  # 连续奖励，避免稀疏性
    else:
        contact_bonus = 0.0
    
    # 5. 距离锚点：distance_reward (保留)
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + smooth_landing_shaping + contact_bonus + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "smooth_landing_shaping": smooth_landing_shaping,
        "contact_landing_bonus": contact_bonus,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```