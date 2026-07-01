```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（增强） ==========
    # 势能函数：Phi = -|hull_angle| - 0.5 * |horizontal_velocity - target_velocity|
    # 目标水平速度设为 1.0 m/s（合理行走速度）
    target_velocity = 1.0
    
    # 当前势能
    phi_obs = -abs(obs[0]) - 0.5 * abs(obs[2] - target_velocity)
    # 下一时刻势能
    phi_next = -abs(next_obs[0]) - 0.5 * abs(next_obs[2] - target_velocity)
    
    gamma = 0.99  # 折扣因子
    progress_delta = gamma * phi_next - phi_obs
    
    # 增强主信号系数，从2.0提升至5.0，鼓励移动
    progress_delta_reward = 5.0 * progress_delta
    
    # ========== 稳定约束：stability_penalty（削弱） ==========
    # 大幅降低角度惩罚系数，从-1.0降至-0.3，避免过度保守
    angular_velocity_penalty = -0.5 * abs(next_obs[1])  # 主体角速度
    vertical_velocity_penalty = -0.3 * abs(next_obs[3])  # 垂直速度
    angle_penalty = -0.3 * abs(next_obs[0])  # 主体角度（从-1.0降至-0.3）
    
    stability_penalty = angular_velocity_penalty + vertical_velocity_penalty + angle_penalty
    
    # ========== 动作代价：energy_penalty（保持不变） ==========
    energy_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty
    
    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```