# Response Record

1. **evidence**：当前得分203.9，14/20的episode因高度/姿态失败终止，forward_reward每步幅度远大于其他惩罚；upright_penalty active_rate仅27.4%，说明只在大倾斜时触发，无法提供早期稳定引导。
2. **behavior_diagnosis**：策略偏向激进前倾以获得高速，但无法及时纠正姿态导致频繁摔倒，episode平均长度仅344步且大量失败终止。
3. **signal_completeness**：现有前进、高度、姿态、横向、角速度信号基本完备，但直立惩罚仅在身体严重倾斜（up_z<0.5）时生效，缺少对轻度倾斜的连续反馈，造成梯度稀疏与延迟。
4. **selected_level**：Level 2，因为直立惩罚的数学形态导致其只在倾斜≥60°后起作用，证据表明agent在该阶段已难以恢复，需要连续惩罚来提供更密集的直立梯度。
5. **selected_intervention**：仅修改`upright_penalty`，从`10.0 * max(0.0, 0.5 - body_up_z)`变为`5.0 * ((1.0 - body_up_z) ** 2)`，使任何偏离完全直立的状态都产生连续惩罚，强度随倾角平方增长。
6. **falsifiable_hypothesis**：新增的连续惩罚将强制agent在加速过程中保持更竖直的姿态，减少过度前倾，从而降低摔倒终止率，提升平均episode长度与总体得分。
7. **expected_next_round**：下一轮terminated比例应明显下降，episode平均长度增加，score提高，`upright_penalty`的active_rate上升至接近100%但其episode_sum_mean绝对值可能下降（因不再出现巨大倾斜）。
8. **main_risk**：惩罚可能过强，尤其当body_up_z很低时（如摔倒过程）产生极高惩罚，可能导致agent过度保守、不敢前进，结果前进速度降低，得分不升反降。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # 前进奖励：平方强化高速
    fwd_speed = max(0.0, body_x_vel)
    forward_reward = 15.0 * (fwd_speed ** 2)

    # 软约束：高度惩罚
    height_low = max(0.0, 0.3 - body_z)
    height_high = max(0.0, body_z - 0.9)
    height_penalty = 20.0 * (height_low + height_high)

    # 直立惩罚：连续惩罚，偏离完全直立即产生代价
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_penalty = 5.0 * ((1.0 - body_up_z) ** 2)

    # 横向速度惩罚
    lat_penalty = 2.0 * (body_y_vel ** 2)

    # 角速度惩罚
    ang_penalty = 0.5 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward - height_penalty - upright_penalty - lat_penalty - ang_penalty

    components = {
        'forward_reward': forward_reward,
        'height_penalty': -height_penalty,
        'upright_penalty': -upright_penalty,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }
    return float(total_reward), components
```
