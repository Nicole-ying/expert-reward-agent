**evidence**: score=-123.55, len=206.85, terminated=17/20 (85% body-height failure), truncated=3/20；forward_velocity_reward episode_sum_mean=81.88且active_rate=100%，stability_penalty=-45.38；正向速度信号活跃但无法阻止崩塌，高度相关终止占绝对主导。

**behavior_diagnosis**: agent在早期学习阶段无法维持身体直立，17/20回合因身体高度超出健康范围提前终止；前向速度奖励正常发放但姿态失控导致快速崩塌，稳定性惩罚无法提供足够的直立引导。

**signal_completeness**: 缺少身体姿态/倾斜的稠密反馈——当前仅有高度边界惩罚和角速度惩罚，没有直接衡量“身体是否直立”的信号；quaternion（obs[1:5]）未被使用，是可达且必要的缺失职责。

**selected_level**: Level 2 — 必要信号缺失（身体倾斜反馈不可达），仅调系数无法解决17/20崩塌问题；干预类型为添加连续姿态奖励组件（sparse_to_dense变体：将隐式高度终止风险转化为显式倾斜反馈）。

**selected_intervention**: 新增`orientation_reward`组件，利用body quaternion计算身体z轴在世界坐标系中的投影`body_up_z`，奖励保持直立姿态。

**falsifiable_hypothesis**: 四足机器人崩塌的本质是身体倾斜累积导致高度越界；增加直立姿态奖励应提供即时、稠密的倾斜纠正信号，使agent在学会前向行走之前先学会保持平衡，从而降低early termination比例、提高episode生存长度。

**expected_next_round**: terminated比例应从17/20显著下降（目标<10/20）；episode_length应增长；orientation_reward的active_rate应接近100%且episode_sum_mean为正；score应从-123向0方向改善。

**main_risk**: 新增奖励可能使agent学到“站立不动”策略——只收orientation_reward而不前进；若forward_velocity_reward相对强度不足，可能出现生存但不前进的中等分平台。下一轮需验证forward_velocity_reward是否保持活跃。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the result state (next_obs)
    body_z = next_obs[0]
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # Quaternion-based body uprightness: z-component of body's local up in world frame
    body_up_z = qw * qw - qx * qx - qy * qy + qz * qz
    body_up_z = max(-1.0, min(1.0, body_up_z))

    # Weights
    w_fwd = 1.0
    w_lat = 0.5
    w_ang = 0.001
    w_h = 10.0
    w_orient = 0.5

    # 1. Main learning signal: forward velocity reward
    forward_reward = w_fwd * body_x_vel

    # 2. Stability constraint (penalty for dangerous/inefficient deviations)
    lateral_penalty = w_lat * abs(body_y_vel)
    angular_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)
    height_penalty = w_h * (
        max(0.0, 0.25 - body_z) + max(0.0, body_z - 0.95)
    )
    stability_penalty = -(lateral_penalty + angular_penalty + height_penalty)

    # 3. Orientation reward: reward staying upright to prevent collapse
    orientation_reward = w_orient * body_up_z

    total_reward = forward_reward + stability_penalty + orientation_reward

    components = {
        "forward_velocity_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "orientation_reward": orientation_reward
    }

    return float(total_reward), components
```