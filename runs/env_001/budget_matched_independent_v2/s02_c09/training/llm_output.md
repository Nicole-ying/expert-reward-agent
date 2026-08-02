# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from next_obs (post-step state)
    x = next_obs[0]          # horizontal position relative to target
    y = next_obs[1]          # vertical position relative to pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body orientation angle
    ang_vel = next_obs[5]    # angular velocity
    left_contact = next_obs[6]   # left support contact flag (0 or 1)
    right_contact = next_obs[7]  # right support contact flag (0 or 1)
    
    # Also extract from obs for delta calculations
    prev_x = obs[0]
    prev_y = obs[1]
    prev_vx = obs[2]
    prev_vy = obs[3]
    
    # ========== Reward Components ==========
    
    # 1. Distance reward: encourage approaching the target (0,0)
    distance = (x ** 2 + y ** 2) ** 0.5
    distance_reward = -0.1 * distance  # negative penalty, stronger when far
    
    # 2. Velocity penalty: penalize high speed, especially when close to target
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # Scale penalty by distance: closer = more penalty for speed
    proximity_weight = 1.0 / (distance + 0.5)  # higher when close
    velocity_penalty = -0.05 * speed * proximity_weight
    
    # 3. Orientation reward: encourage upright orientation (angle near 0)
    orientation_penalty = -0.02 * (angle ** 2)  # quadratic penalty for deviation
    
    # 4. Angular velocity penalty: discourage spinning
    angular_penalty = -0.01 * (ang_vel ** 2)
    
    # 5. Contact reward: reward stable contact with both supports
    both_contact = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_reward = 0.5 * both_contact
    
    # 6. Progress bonus: reward moving toward target (delta distance reduction)
    prev_distance = (prev_x ** 2 + prev_y ** 2) ** 0.5
    distance_delta = prev_distance - distance  # positive if moving closer
    progress_bonus = 0.2 * max(0, distance_delta)
    
    # 7. Fuel efficiency penalty: penalize engine usage (action 2 = main engine)
    fuel_penalty = -0.05 if action == 2 else 0.0
    
    # 8. Stability bonus: reward being settled (low speed + good orientation + contact)
    is_settled = (speed < 0.1) and (abs(angle) < 0.1) and (abs(ang_vel) < 0.05) and both_contact
    stability_bonus = 1.0 if is_settled else 0.0
    
    # ========== Combine ==========
    total_reward = (distance_reward + velocity_penalty + orientation_penalty + 
                    angular_penalty + contact_reward + progress_bonus + 
                    fuel_penalty + stability_bonus)
    
    components = {
        'distance_reward': distance_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'angular_penalty': angular_penalty,
        'contact_reward': contact_reward,
        'progress_bonus': progress_bonus,
        'fuel_penalty': fuel_penalty,
        'stability_bonus': stability_bonus,
    }
    
    return float(total_reward), components
```

# 设计说明

**任务理解**：这是一个2D飞行器/着陆器任务，目标是在目标垫上快速、稳定地着陆，同时最小化燃料消耗。观测包含位置、速度、姿态角和接触状态。

**信号选择与理由**：
- 使用位置(x,y)计算距离，这是最直接的接近度度量
- 使用速度(vx,vy)惩罚高速，尤其在接近目标时（通过proximity_weight实现）
- 使用角度和角速度惩罚不稳定姿态
- 使用接触标志奖励稳定着陆（双脚接触）
- 使用动作值惩罚引擎使用（action==2对应主引擎）

**数学形式选择**：
- 距离惩罚采用线性形式，避免过大梯度
- 速度惩罚与距离成反比，使agent在远处可以快速移动，近处必须减速
- 角度和角速度采用二次惩罚，对偏差更敏感
- 接触奖励是二值的，但只在双脚都接触时才触发
- 进度奖励鼓励每步都向目标移动，提供密集梯度
- 稳定性奖励是稀疏的，但提供明确的成功信号

**避免奖励漏洞**：
- 速度惩罚与距离耦合，防止agent高速掠过目标
- 进度奖励鼓励持续接近，防止agent原地不动
- 燃料惩罚防止无节制使用引擎
- 稳定性奖励要求同时满足多个条件，防止取巧