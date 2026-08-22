1. evidence：本轮正向奖励（直立 750.9、前进 572.5）远大于得分 265，但 3/20 提前终止致平均 episode 长度 852.7，未达 1000；历史 best（434.5）来自更轻量 forward 系数且无提前终止。
2. behavior_diagnosis：策略在过度激励前进时步态不稳定，少数 episode 身体高度越界提前摔倒，多数 episode 虽存活但效率不高，总得分因此下降。
3. signal_completeness：前进、直立、高度软约束与侧漂惩罚已具备引导走跑所需核心信号，但高度软约束过弱，未能有效阻止安全边界外冲高。
4. selected_level：Level 1——高度惩罚职责合理但系数偏小（episode_sum_mean 仅 -37.7，active_rate 5.5%），仅需加大惩罚幅度即可预期改善提前终止。
5. selected_intervention：只修改 height_penalty 组件，将边界外惩罚系数从 20.0 提升至 60.0。
6. falsifiable_hypothesis：更大高度惩罚将迫使策略更早纠正 body_z 偏离，减少越过 [0.2, 1.0] 导致的提前终止，从而延长 episode 并提升总分。
7. expected_next_round：terminated 比例下降（<3/20），episode_length 均值接近 1000，score 回升，height_penalty episode_sum_mean 负值暂时增大但随后因行为改善而降低。
8. main_risk：系数过强可能过度约束步长，降低前进速度，使 forward_progress 减少；若下次得分无改善则需重新考虑前进与稳定的平衡。 

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract signals ----------
    body_z          = obs[0]
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]
    body_x_vel      = obs[13]
    body_y_vel      = obs[14]
    roll_vel        = obs[16]
    pitch_vel       = obs[17]

    # compute torso uprightness (world z projection)
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)

    # -------- 1. forward progress (bounded, scaled up) --------
    forward_reward = 60.0 * (body_x_vel / (1.0 + abs(body_x_vel)))

    # -------- 2. health height (soft boundary cost) --------
    height_penalty = 0.0
    if body_z < 0.3:
        height_penalty = -60.0 * (0.3 - body_z)   # increased from 20.0
    elif body_z > 0.9:
        height_penalty = -60.0 * (body_z - 0.9)

    # -------- 3. upright stability (posture + angular velocity) --------
    upright_reward = 1.0 * body_up_z
    ang_penalty    = -0.005 * (roll_vel**2 + pitch_vel**2)
    upright_stability = upright_reward + ang_penalty

    # -------- 4. lateral drift penalty --------
    lateral_penalty = -0.1 * (body_y_vel**2)

    # -------- total reward and components --------
    total = forward_reward + height_penalty + upright_stability + lateral_penalty

    components = {
        "forward_progress":       forward_reward,
        "height_penalty":         height_penalty,
        "upright_stability":      upright_stability,
        "lateral_drift_penalty":  lateral_penalty
    }

    return float(total), components
```