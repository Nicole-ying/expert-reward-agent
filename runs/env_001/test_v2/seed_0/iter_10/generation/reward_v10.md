```json
{
  "action": "tune",
  "target": "progress_delta_reward, landing_quality_shaping, stability_penalty, distance_anchor, low_speed_bonus",
  "reasoning": "External score 19.51 far below target 200. Landing_quality_shaping has high mean (1.57) and strong positive signal but external score is low, indicating it rewards near-goal oscillation without success (goal_near_oscillation). Progress_delta_reward mean is only 0.31, insufficient to drive real progress. Stability_penalty mean is -0.12, too weak to enforce stability. Distance_anchor mean is -0.05, negligible. Low_speed_bonus mean 0.64 with 54% trigger rate is moderate but may reinforce oscillation. Best reward (iter 8) used progress_delta_reward=40.0, landing_quality_shaping=6.0, distance_anchor=-0.2, and no low_speed_bonus, achieving 188.21. Current version increased progress to 60.0 but added low_speed_bonus and reduced landing_quality_shaping to 4.0 and distance_anchor to -0.1, causing score collapse. Revert to best configuration: progress_delta_reward=40.0, landing_quality_shaping=6.0, distance_anchor=-0.2, remove low_speed_bonus, and strengthen stability_penalty coefficients to prevent oscillation.",
  "expected_effect": "Score should recover toward best (188.21) by restoring proven balance between progress and landing quality, while stronger stability penalty prevents near-goal oscillation.",
  "risk_awareness": "Risk of over-constraining agent with stronger stability penalty, but necessary to break oscillation pattern. If score does not recover, consider rebuild with time_penalty skeleton."
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
    
    # 组件1: 主学习信号 - progress_delta_reward (恢复最佳配置40.0)
    progress_delta = current_dist - next_dist
    progress_delta_reward = 40.0 * progress_delta
    
    # 组件2: 条件化稳定约束 - conditional_stability_penalty (增强系数以抑制振荡)
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    # 根据距离调整惩罚强度：远距离时惩罚更轻
    distance_factor = min(1.0, next_dist / 0.5)  # 0~1, 距离越远因子越大(惩罚越轻)
    # 增强惩罚系数，特别是角度和角速度，以抑制近目标振荡
    angle_penalty = 1.0 * abs(body_angle) * (1.0 - 0.5 * distance_factor)  # 从0.6增加到1.0
    angular_vel_penalty = 0.4 * abs(angular_vel) * (1.0 - 0.5 * distance_factor)  # 从0.2增加到0.4
    speed_penalty = 0.6 * speed * (1.0 - 0.5 * distance_factor)  # 从0.4增加到0.6
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 组件3: 连续着陆质量塑形 - landing_quality_shaping (恢复最佳配置6.0)
    # 当接近目标时，奖励低速、稳定姿态和双支撑接触
    near_target = max(0.0, 1.0 - next_dist / 0.8)  # 保持激活范围
    speed_quality = max(0.0, 1.0 - speed / 0.5)  # 0~1, 越慢越大
    angle_quality = max(0.0, 1.0 - abs(body_angle) / 0.3)  # 0~1, 越正越大
    contact_quality = 0.5 * (left_contact + right_contact)  # 0~1, 双支撑更好
    
    # 组合成连续信号，仅在接近目标时激活
    landing_quality = near_target * (0.3 * speed_quality + 0.3 * angle_quality + 0.4 * contact_quality)
    landing_quality_shaping = 6.0 * landing_quality  # 恢复为6.0
    
    # 组件4: 距离锚点 - distance_anchor (恢复最佳配置-0.2)
    distance_anchor = -0.2 * next_dist  # 恢复为-0.2
    
    # 移除low_speed_bonus，因为它可能加剧振荡
    
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