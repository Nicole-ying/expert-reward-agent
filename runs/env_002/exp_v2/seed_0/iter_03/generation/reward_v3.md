```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（大幅增强） ==========
    # 使用水平速度变化作为前进驱动力
    current_horizontal_vel = obs[2]
    next_horizontal_vel = next_obs[2]
    progress_delta = next_horizontal_vel - current_horizontal_vel
    # 系数从5.0提升至15.0，增强驱动力（证据显示均值仅0.012，远不足以驱动学习）
    progress_delta_reward = 15.0 * progress_delta

    # ========== 稳定约束：stability_penalty（适度增强） ==========
    # 惩罚身体倾斜角度过大和角速度过大，防止摔倒
    hull_angle = next_obs[0]  # 身体倾斜角度
    hull_angular_vel = next_obs[1]  # 身体角速度
    # 角度惩罚：仅当角度>0.2时开始惩罚，线性增长（阈值从0.3收紧至0.2）
    angle_penalty = -0.3 * max(0.0, abs(hull_angle) - 0.2)
    # 角速度惩罚：仅当角速度>0.3时开始惩罚，线性增长（阈值从0.5收紧至0.3）
    angular_vel_penalty = -0.2 * max(0.0, abs(hull_angular_vel) - 0.3)
    stability_penalty = angle_penalty + angular_vel_penalty

    # ========== 效率约束：energy_penalty（保持小权重） ==========
    action_magnitude = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.01 * action_magnitude

    # ========== 任务完成proxy：soft_landing_proxy（收紧条件，降低权重） ==========
    leg1_contact = next_obs[12]
    leg2_contact = next_obs[13]
    both_contact = 1.0 if (leg1_contact > 0.5 and leg2_contact > 0.5) else 0.0
    # 收紧直立条件：角度<0.2，角速度<0.5（防止contact_reward_hacking）
    is_upright = 1.0 if (abs(hull_angle) < 0.2 and abs(hull_angular_vel) < 0.5) else 0.0
    # 收紧速度条件：水平速度在0.5到3.0之间（防止低速震荡）
    is_good_speed = 1.0 if (0.5 < next_horizontal_vel < 3.0) else 0.0
    # 降低权重从0.5至0.2，减少稀疏bonus的干扰
    soft_landing_bonus = 0.2 * both_contact * is_upright * is_good_speed

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty + soft_landing_bonus

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```