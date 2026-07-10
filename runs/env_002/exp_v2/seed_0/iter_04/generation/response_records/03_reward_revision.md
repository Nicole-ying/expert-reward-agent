# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（恢复best系数5.0） ==========
    # 使用水平速度变化作为前进驱动力
    current_horizontal_vel = obs[2]
    next_horizontal_vel = next_obs[2]
    progress_delta = next_horizontal_vel - current_horizontal_vel
    # 恢复best的5.0系数，证据显示当前15.0导致大幅负值震荡
    progress_delta_reward = 5.0 * progress_delta

    # ========== 稳定约束：stability_penalty（恢复best系数，保持宽松阈值） ==========
    hull_angle = next_obs[0]  # 身体倾斜角度
    hull_angular_vel = next_obs[1]  # 身体角速度
    # 恢复best的宽松阈值和低系数：角度>0.3，角速度>0.5
    angle_penalty = -0.1 * max(0.0, abs(hull_angle) - 0.3)
    angular_vel_penalty = -0.05 * max(0.0, abs(hull_angular_vel) - 0.5)
    stability_penalty = angle_penalty + angular_vel_penalty

    # ========== 效率约束：energy_penalty（保持小权重） ==========
    action_magnitude = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.01 * action_magnitude

    # ========== 任务完成proxy：soft_landing_proxy（恢复best的宽松条件和权重0.5） ==========
    leg1_contact = next_obs[12]
    leg2_contact = next_obs[13]
    both_contact = 1.0 if (leg1_contact > 0.5 and leg2_contact > 0.5) else 0.0
    # 恢复best的宽松条件：角度<0.4，角速度<1.0，速度0.2~4.0
    is_upright = 1.0 if (abs(hull_angle) < 0.4 and abs(hull_angular_vel) < 1.0) else 0.0
    is_good_speed = 1.0 if (0.2 < next_horizontal_vel < 4.0) else 0.0
    soft_landing_bonus = 0.5 * both_contact * is_upright * is_good_speed

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
