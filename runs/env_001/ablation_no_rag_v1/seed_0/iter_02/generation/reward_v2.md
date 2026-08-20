1. **evidence**: 上一轮训练得分-110.23，episode平均长度仅68.4步，全部被terminated（无truncation），距离惩罚episode_sum_mean占约-66.5且active_rate100%，速度和姿态惩罚幅度较小，表明所有episodes迅速以失败终止。
2. **behavior_diagnosis**: 策略在早期就因倾斜、高速或越界而快速崩溃，未能向目标垫移动，没有学会任何接近行为。
3. **signal_completeness**: 必要职责缺失：没有正向引导奖励告知代理靠近目标是有利的，全部组件为负值可能促使代理尽快终止以减少累积惩罚。
4. **selected_level**: Level 2，证据显示当前-proximity_reward（负距离）无法提供正向学习信号，须改变其数学形态为有界正奖励。
5. **selected_intervention**: 将proximity_reward从 `-distance` 转换为 `1.0/(1.0+distance)`，提供稠密的正向接近激励，同时保持其他惩罚组件不变。
6. **falsifiable_hypothesis**: 引入正的接近奖励应引导代理主动移动向目标垫，从而延长episode长度，减少早期崩溃，并可能实现到达垫面附近。
7. **expected_next_round**: episode长度显著增加，评分绝对值降低或趋近于零，proximity_reward的magnitude_share转为正向且上升，速度/姿态惩罚仍存在但总奖励向正方向偏移。
8. **main_risk**: 若正向接近奖励强度不足以压过全局速度/姿态惩罚，代理仍可能倾向于快速终止；若惩罚过强，可能导致缓慢徘徊而不稳定着陆。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    distance = (x**2 + y**2)**0.5
    proximity_reward = 1.0 / (1.0 + distance)

    tilt_penalty = -0.5 * abs(angle)
    speed = (vx**2 + vy**2)**0.5
    velocity_penalty = -0.1 * speed
    rotation_penalty = -0.1 * abs(angular_vel)

    total_reward = proximity_reward + tilt_penalty + velocity_penalty + rotation_penalty

    components = {
        "proximity_reward": proximity_reward,
        "tilt_penalty": tilt_penalty,
        "velocity_penalty": velocity_penalty,
        "rotation_penalty": rotation_penalty
    }
    return total_reward, components
```