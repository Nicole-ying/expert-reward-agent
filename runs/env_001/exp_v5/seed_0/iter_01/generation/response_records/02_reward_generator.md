# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    # Position relative to target (goal is at origin)
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # Velocity
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # Orientation
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # Contact flags
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== Component 1: Progress Delta Reward (Main Learning Signal) ==========
    # Distance to target (Euclidean distance)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # Progress: positive when moving closer to target
    progress_delta = current_dist - next_dist
    
    # Scale progress reward - encourage consistent progress
    progress_reward = 5.0 * progress_delta
    
    # ========== Component 2: Stability Penalty (Light Constraint) ==========
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = abs(body_angle)
    angular_vel_penalty = abs(angular_vel)
    
    # Small weights to avoid over-constraining
    stability_penalty = -0.1 * speed - 0.5 * angle_penalty - 0.05 * angular_vel_penalty
    
    # ========== Component 3: Soft Landing Proxy (Task Completion Hint) ==========
    # Small bonus when near target, slow, stable, and both supports contact
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(body_angle) < 0.2
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0
    
    # ========== Total Reward ==========
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # ========== Components Dict ==========
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward** (主学习信号): 基于每一步到目标距离的变化量（当前距离 - 下一时刻距离），鼓励智能体持续向目标靠近。权重设为5.0，使其成为主导信号。

2. **stability_penalty** (稳定/安全约束): 轻量惩罚项，包含速度惩罚(-0.1)、姿态角惩罚(-0.5)和角速度惩罚(-0.05)。目的是防止智能体以高速或大姿态角接近目标，为后续稳定着陆做准备。权重较小，避免过度约束导致不敢移动。

3. **soft_landing_bonus** (任务完成proxy): 当智能体同时满足四个条件（靠近目标、低速、姿态稳定、双侧支撑接触）时给予小奖励(2.0)。这是对成功着陆的软性近似，不依赖显式success flag。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info字典为空。没有可靠的信号来判断成功或失败终止，因此不使用终端奖励，避免伪造信号或误判。

## 留到后续迭代的组件

- **energy_penalty**: 当前动作代价未加入，因为v1优先让智能体学会接近目标，后续若发现燃料消耗过大再添加。
- **time_penalty**: 未加入，避免过早鼓励快速完成导致不稳定行为。
- **gated_reward**: 未使用，v1保持简单直接。
- **distance_reward**: 未作为独立组件，因为progress_delta已经包含了距离信息，避免重复。

## 训练后应观察的failure mode

1. **goal_near_oscillation**: 目标附近震荡不完成着陆。表现为progress_reward接近0但soft_landing_bonus始终未触发。解决方案：后续可加入更积极的着陆引导或调整soft_landing条件。

2. **high_reward_without_success**: 智能体在目标附近徘徊获得高progress_reward但从未完成着陆。解决方案：收紧soft_landing条件或加入时间惩罚。

3. **fast_crash_near_goal**: 高速冲向目标导致坠毁。表现为progress_reward高但stability_penalty不足以抑制速度。解决方案：增大速度惩罚权重或加入速度门控。

4. **conservative_behavior**: 智能体过于保守不敢移动。表现为progress_reward长期为负或接近0。解决方案：降低stability_penalty权重或增加progress_reward的激励。
