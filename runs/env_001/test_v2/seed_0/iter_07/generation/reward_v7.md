```json
{
  "action": "tune",
  "target": "progress_delta_reward, distance_anchor, landing_quality_shaping, stability_penalty",
  "reasoning": "External score 93.79 is far below target 200. Progress_delta_reward mean 0.126 is too small to drive strong progress; distance_anchor negative mean (-0.082) indicates agent stays far from target; landing_quality_shaping nonzero rate 0.32 is too sparse; stability_penalty mean -0.103 is too weak to enforce stability. Only 1 iteration on this skeleton, so tune coefficients: increase progress_delta_reward from 10.0 to 20.0, increase distance_anchor from -0.1 to -0.5, increase landing_quality_shaping from 3.0 to 5.0 and adjust its activation threshold, increase stability_penalty coefficients by 2x.",
  "expected_effect": "Stronger progress signal drives agent toward target faster; larger distance penalty pulls agent closer; more frequent landing shaping guides stable landing; stronger stability penalty prevents excessive speed/angle.",
  "risk_awareness": "Over-tuning may cause instability or oscillation; progress_delta_reward increase may amplify noise; distance_anchor increase may penalize exploration; landing_quality_shaping increase may cause overfitting to near-target behavior."
}
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态变量
    x_pos = obs[0]
    y_pos = obs[1]
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    
    # 速度
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    
    # 姿态
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 接触标志
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 计算距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # 组件1: 主学习信号 - progress_delta_reward (增强系数)
    progress_delta = current_dist - next_dist
    progress_delta_reward = 20.0 * progress_delta  # 从10.0增加到20.0
    
    # 组件2: 条件化稳定约束 - conditional_stability_penalty (增强系数)
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # 根据距离调整惩罚强度：远距离时惩罚更轻
    distance_factor = min(1.0, next_dist / 0.5)  # 0~1, 距离越远因子越大(惩罚越轻)
    # 当距离远时，允许更大的速度和角度
    angle_penalty = 0.6 * abs(body_angle) * (1.0 - 0.5 * distance_factor)  # 从0.3增加到0.6
    angular_vel_penalty = 0.2 * abs(angular_vel) * (1.0 - 0.5 * distance_factor)  # 从0.1增加到0.2
    speed_penalty = 0.4 * speed * (1.0 - 0.5 * distance_factor)  # 从0.2增加到0.4
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 连续着陆质量塑形 - landing_quality_shaping (增强系数并调整激活阈值)
    # 替代稀疏的soft_landing_bonus，提供连续信号
    # 当接近目标时，奖励低速、稳定姿态和双支撑接触
    near_target = max(0.0, 1.0 - next_dist / 0.8)  # 从0.5放宽到0.8，增加激活范围
    speed_quality = max(0.0, 1.0 - speed / 0.5)  # 0~1, 越慢越大
    angle_quality = max(0.0, 1.0 - abs(body_angle) / 0.3)  # 0~1, 越正越大
    contact_quality = 0.5 * (left_contact + right_contact)  # 0~1, 双支撑更好
    
    # 组合成连续信号，仅在接近目标时激活
    landing_quality = near_target * (0.3 * speed_quality + 0.3 * angle_quality + 0.4 * contact_quality)
    landing_quality_shaping = 5.0 * landing_quality  # 从3.0增加到5.0
    
    # 组件4: 增强距离锚点 - distance_anchor (增强系数)
    distance_anchor = -0.5 * next_dist  # 从-0.1增加到-0.5
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + landing_quality_shaping + distance_anchor
    
    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_quality_shaping": landing_quality_shaping,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```