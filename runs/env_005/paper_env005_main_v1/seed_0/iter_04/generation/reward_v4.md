`evidence`: 当前score=-137.45，较best(-50.90)大幅倒退；terminated=18/20表明策略频繁跌倒；唯一改动w_h:10→50引入死亡螺旋（score_range下探至-1247）；forward_velocity_reward占magnitude_share 50.4%且在任意姿态下均全额发放，为不稳定前冲提供了错误激励。

`behavior_diagnosis`: 策略学会了前向移动（per-step forward≈0.645），但在姿态倾斜甚至即将跌倒时仍获得全额前进奖励，导致"边倒边冲"的 exploit 行为；一旦进入低高度区，w_h=50产生剧烈惩罚形成死亡螺旋，18/20回合以失败终止。

`signal_completeness`: 现有三信号职责完备——前进引导、姿态稳定、方向约束——但 forward_velocity_reward 未与稳定性条件绑定，使其在机器人失控时仍提供正向梯度，形成 proxy-to-completion 错位。

`selected_level`: Level 2 — `global_to_local_gate`。触发条件：(1) forward_velocity_reward 是唯一主信号但其不顾姿态始终全额发放，证据为18/20失败终止时 forward 仍贡献42.5% signed_share；(2) 纯系数调整(w_h 10→50)已证实无效且恶化；(3) 需要结构变换将前进奖励门控在稳定状态上。

`selected_intervention`: 唯一修改 forward_velocity_reward 组件，将其从 `w_fwd * body_x_vel` 变换为 `w_fwd * body_x_vel * stability_gate`，其中 stability_gate 由身体直立度与高度健康度连续乘积构成；其他组件保持 best 代码不变（w_h=10.0）。

`falsifiable_hypothesis`: 门控后，前进奖励仅在机器人直立且高度正常时发放，消除了"边倒边冲"的激励；策略必须先学会保持稳定姿态才能获得前进奖励，形成自然课程——terminated比例应下降，episode_length应回升，score_range负向极端值应收敛。

`expected_next_round`: terminated 应从 18/20 下降至 ≤15/20；episode_length 应回升至 400+；score_range 下限应收敛（不再出现 <-500 的单回合崩盘）；forward_velocity_reward 的 signed_share 可能下降，但 total score 应改善。

`main_risk`: stability_gate 如果过早过严（尤其在训练早期 body_up_z 普遍偏低时），可能导致前进信号塌缩、策略退化为仅维持站立的局部最优；需通过连续门控（非硬截断）和保留无门的 orientation_reward 来维持早期探索梯度。

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

    # Height health: 1 at center (0.6), 0 at boundaries (0.25, 0.95)
    height_health = 1.0 - abs(body_z - 0.6) / 0.35
    height_health = max(0.0, min(1.0, height_health))

    # Stability gate: only reward forward motion when upright and at healthy height
    stability_gate = max(0.0, body_up_z) * height_health

    # Weights
    w_fwd = 1.0
    w_lat = 0.5
    w_ang = 0.001
    w_h = 10.0
    w_orient = 0.5

    # 1. Main learning signal: forward velocity reward, gated by stability
    forward_reward = w_fwd * body_x_vel * stability_gate

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