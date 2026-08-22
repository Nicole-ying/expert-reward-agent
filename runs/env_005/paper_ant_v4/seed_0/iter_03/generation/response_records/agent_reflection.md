# Response Record

1. **evidence**：当前得分13.82，全部20条episode均因高度越界终止，平均长度仅29.25步；velocity_reward均值18.79，height_penalty均值9.89，其他惩罚项合计约15.86，正负奖励在短命轨迹里勉强持平，机器人快速摔倒。
2. **behavior_diagnosis**：策略为追求前向速度而采取不稳定动作，在高度下降过程中未能及时纠正，导致每次都在30步内跌破0.2m并终止，没有学会持续存活和稳定行走。
3. **signal_completeness**：前向速度、高度约束、直立和角速度惩罚等核心职责均已存在，但高度‑速度之间的权衡点设置过晚，导致速度激励在高度明显下降时仍保持全效，无法有效阻止冒险。
4. **selected_level**：Level 2，因为必要信号可达但现有门控阈值（0.35）未能及时衰减速度奖励，属于“约束在无关阶段妨碍探索”变体——需要在高度下降初期就施加阶段相关的速度抑制，故选择global_to_local_gate思路修改velocity_reward中的门控。
5. **selected_intervention**：仅修改velocity_reward组件内的health_factor低高度阈值，将线性衰减区间从[0.2, 0.35]扩大到[0.2, 0.5]，使身体高度低于0.5时前向速度奖励即开始成比例衰减。
6. **falsifiable_hypothesis**：更早衰减速度奖励会降低机器人因追求速度而持续降低高度的动机，迫使它在较高的高度范围内学会稳定步态，从而延长episode长度并最终提高累计得分。
7. **expected_next_round**：episode平均长度应明显增加，terminated比例可能下降或至少延迟发生，score因存活更久而有望从约14提升，不再全部快速死亡。
8. **main_risk**：门控开启过早可能压制正常步态所需的高度波动（如行走时重心起伏），导致agent趋于站立不动或极慢移动，出现短期内得分停滞甚至下降，但对长期学习稳定行走仍是必要代价。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Core Forward Velocity Reward (mandatory) ==========
    forward_velocity = next_obs[13]
    velocity_reward = 1.0 * forward_velocity

    # ========== Height Safety Penalty (mandatory) ==========
    body_height = next_obs[0]
    height_error = body_height - 0.6
    height_penalty = 10.0 * (height_error ** 2)

    # ========== Upright Pose Penalty (mandatory) ==========
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    upright_error = 1.0 - body_up_z
    upright_penalty = 0.15 * (upright_error ** 2)

    # ========== Lateral Drift Penalty (conditional) ==========
    lateral_velocity = abs(next_obs[14])
    lateral_penalty = 0.1 * (lateral_velocity ** 2)

    # ========== Angular Velocity Penalty (conditional) ==========
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    angular_penalty = 0.05 * (roll_vel ** 2 + pitch_vel ** 2)

    # ========== Soft Health Gate for Extreme Heights ==========
    health_factor = 1.0
    # 修改点：将低高度衰减区间从[0.2, 0.35] 扩大为 [0.2, 0.5]
    if body_height < 0.5:
        health_factor = max(0.0, (body_height - 0.2) / 0.3)
    elif body_height > 0.85:
        health_factor = max(0.0, (1.0 - body_height) / 0.15)

    velocity_reward = velocity_reward * health_factor

    # ========== Total Reward Assembly ==========
    total_reward = (
        velocity_reward
        - height_penalty
        - upright_penalty
        - lateral_penalty
        - angular_penalty
    )

    # ========== Component Dictionary ==========
    components = {
        'velocity_reward': velocity_reward,
        'height_penalty': height_penalty,
        'upright_penalty': upright_penalty,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty
    }

    return float(total_reward), components
```
