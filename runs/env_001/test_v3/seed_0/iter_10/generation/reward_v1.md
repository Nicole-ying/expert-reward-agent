# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward
    # Reward for moving closer to the target (0,0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = progress_delta * 10.0  # Scale to meaningful magnitude
    
    # 2. Stability penalty: penalize high velocity, large angle, and high angular velocity
    # Use next_obs to penalize resulting state after action
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = abs(next_body_angle) * 0.5
    angular_vel_penalty = abs(next_angular_vel) * 0.3
    speed_penalty = speed * 0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. Soft landing proxy: small bonus when near target, slow, stable, and both contacts
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # 4. Small distance anchor to prevent drifting far away
    distance_anchor = -next_dist * 0.5
    
    # Combine components
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward** (主学习信号，权重10.0)
   - 角色：引导智能体每一步都向目标(0,0)靠近
   - 计算方式：当前距离减去下一步距离，正值表示接近目标
   - 这是最直接的任务引导信号，让智能体学会向目标移动

2. **stability_penalty** (稳定约束，负值)
   - 角色：惩罚高速、大姿态角和大角速度，鼓励稳定飞行
   - 包含三个子项：速度惩罚(0.2)、角度惩罚(0.5)、角速度惩罚(0.3)
   - 权重较小，避免过度约束导致智能体不敢移动

3. **soft_landing_bonus** (任务完成proxy，固定值2.0)
   - 角色：当智能体同时满足近目标、低速、稳定姿态、双接触时给予小奖励
   - 条件组合严格，避免contact hacking
   - 权重小，不会主导学习信号

4. **distance_anchor** (辅助锚点，权重0.5)
   - 角色：提供持续的距离惩罚，防止智能体在远处徘徊
   - 权重很小，仅作为辅助信号

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- 环境明确说明`explicit_success_flag_available=false`和`explicit_failure_flag_available=false`
- info始终为空字典，无法区分成功和失败终止
- 使用这些项会需要虚构success/failure信号，违反设计原则

## 留到后续迭代的组件

- **energy_penalty**：当前动作空间是离散的(4个动作)，过早加入燃料惩罚可能导致智能体不敢使用引擎，无法学习基本移动
- **time_penalty**：没有时间步数信息，且可能导致智能体冒险快速失败
- **gated_reward**：复杂度过高，v1不需要
- **terminal_success_reward**：需要wrapper暴露success信号后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近来回震荡，无法稳定着陆
   - 观察：progress_reward接近0但soft_landing_bonus很少触发
   - 对策：增加stability_penalty权重或调整soft_landing条件

2. **high_reward_without_success**：智能体获得高progress_reward但从未触发soft_landing
   - 观察：progress_reward持续为正但soft_landing_bonus=0
   - 对策：收紧soft_landing条件或增加距离阈值

3. **fast_crash_near_goal**：智能体高速冲向目标然后坠毁
   - 观察：progress_reward很高但stability_penalty也很高
   - 对策：增加速度惩罚权重或加入速度门控

4. **agent_afraid_to_move**：智能体原地不动，获得0奖励
   - 观察：所有组件都接近0
   - 对策：降低stability_penalty权重或增加progress_reward权重