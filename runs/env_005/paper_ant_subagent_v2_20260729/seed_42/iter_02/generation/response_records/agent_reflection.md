# Response Record

1. `evidence`：最终得分 -336.84，len 944.4，3/20 terminated，17/20 truncated。forward_progress 累计均值 853.95，正向主导；roll_pitch_hinge -66.45、lateral_drift_hinge -63.78，合计正奖励约 +724，但外部评分仍为负。subagent 指出 gate 允许倾斜下仍给予前进奖励，导致高误报正向信号，而真实环境惩罚使总分负值。
2. `behavior_diagnosis`：策略学会利用宽松的 upright_gate（全开阈值仅 0.5，zero 阈值 0.2）在身体明显倾斜时仍持续获得前进奖励，从而导致以牺牲稳定性换取速度，触发了环境中的隐含大额负惩罚（可能包括控制能耗或未建模的摔倒前兆惩罚）。
3. `signal_completeness`：前进方向、侧移、俯仰翻滚惩罚均已存在，但直立条件的信号不够严格，未能有效阻止倾斜状态下的奖励误发，构成当前最大缺口；未使用关节/能耗维度，暂不新增。
4. `selected_level`：Level 2 结构变换，因为需要改变 upright_gate 的核心数学形态（收紧阈值）来消除误报，而非简单尺度调整。
5. `selected_intervention`：仅修改 upright_gate，将全开阈值从 0.5 提高到 0.8，零门阈值从 0.2 提高到 0.4，使 forward_progress 只在接近直立（body_up_z > 0.4）时才获得奖励，并且在 body_up_z 在 0.4~0.8 之间线性插值。
6. `falsifiable_hypothesis`：收紧直立条线后，agent 必须保持更直立才能获取前进奖励，从而减少倾斜状态下的无效前进，间接消除由此引发的环境隐藏惩罚，最终提高外部评分。如果后续评分未改善，可能表明非直立因素是主因，需另寻方向。
7. `expected_next_round`：forward_progress 的 episode_sum_mean 可能下降，但 score 应显著上升（负值减少），terminated 率进一步降低，upright_gate 的实际激活率上升（因为 body_up_z 需更高才给奖励，但 upright_gate 本身是门控因子，其“active”概念可能表现为 body_up_z 平均值的提高）。
8. `main_risk`：初期可能因奖励稀疏导致探索困难，agent 难以学会直立行走的前进策略；若下一轮 score 未改善且 ep_length 缩短，则需考虑放宽或引入课程机制。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract necessary observations
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_roll_velocity = obs[16]
    body_pitch_velocity = obs[17]

    # Compute body up vector z-component (uprightness)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(-1.0, min(1.0, body_up_z))  # numerical safety

    # Health gate: height
    height_low = 0.2
    height_low_margin = 0.15   # 0.2 + 0.15 = 0.35
    height_high = 0.95
    height_high_margin = 0.05  # 0.95 - 0.05 = 0.90

    # Height gate: linear decay to 0 when approaching boundaries
    if body_z < height_low:
        height_gate = 0.0
    elif body_z < height_low + height_low_margin:
        height_gate = (body_z - height_low) / height_low_margin
    elif body_z > height_high:
        height_gate = 0.0
    elif body_z > height_high - height_high_margin:
        height_gate = (height_high - body_z) / height_high_margin
    else:
        height_gate = 1.0

    # Uprightness gate: tightened thresholds
    upright_threshold = 0.4   # below this -> gate 0
    upright_full = 0.8        # above this -> gate 1
    if body_up_z < upright_threshold:
        upright_gate = 0.0
    elif body_up_z < upright_full:
        upright_gate = (body_up_z - upright_threshold) / (upright_full - upright_threshold)
    else:
        upright_gate = 1.0

    gate = height_gate * upright_gate

    # Forward progress (main signal)
    w_forward = 1.0
    forward_progress = w_forward * max(0.0, body_x_velocity) * gate

    # Lateral drift penalty (hinge)
    w_lat = 0.5
    lateral_drift_penalty = -w_lat * max(0.0, abs(body_y_velocity) - 0.3)

    # Roll/pitch stability hinge
    w_rp = 0.2
    ang_vel_margin = 1.0  # rad/s
    roll_penalty = max(0.0, abs(body_roll_velocity) - ang_vel_margin)
    pitch_penalty = max(0.0, abs(body_pitch_velocity) - ang_vel_margin)
    roll_pitch_penalty = -w_rp * (roll_penalty + pitch_penalty)

    total_reward = forward_progress + lateral_drift_penalty + roll_pitch_penalty

    components = {
        'forward_progress': forward_progress,
        'lateral_drift_hinge': lateral_drift_penalty,
        'roll_pitch_hinge': roll_pitch_penalty
    }

    return float(total_reward), components
```
