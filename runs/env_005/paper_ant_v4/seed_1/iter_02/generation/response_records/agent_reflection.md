# Response Record

1. **evidence**：所有回合均存活满1000步，height_penalty 从未触发，upright_stability 贡献了97.1%的奖励（总和975），forward_progress 总和仅10.5（活跃率57.7%），策略几乎静止。
2. **behavior_diagnosis**：策略通过保持直立站立获取了较高的 upright 奖励，因前进奖励相对过弱而陷入不动的最优解。
3. **signal_completeness**：任务所需的前进引导、姿态稳定、高度安全、侧向控制均已具备，但前进信号的尺度被直立奖励严重淹没。
4. **selected_level**：Level 1——尺度修复。forward_progress 的系数过小导致其被 upright 信号完全主导，只需增大其系数即可改变相对重要性。
5. **selected_intervention**：仅将 forward_progress 的倍率从 2.0 改为 20.0，其他所有组件、结构、约束保持不变。
6. **falsifiable_hypothesis**：增大前进奖励的幅度会打破“站立最优”的均衡，驱使策略贪心地提高 body_x_vel，使 forward_progress 的贡献大幅上升，总分随之提高。
7. **expected_next_round**：forward_progress 的 episode_sum_mean 显著增加，signed_share 大幅提升；upright_stability 的份额下降，但仍保持较高绝对值以保证姿态稳定；存活长度仍接近1000，出现 early termination 的概率很低。
8. **main_risk**：过于激进的加速可能使策略采取危险姿态并偶尔摔倒（触发 height_penalty 或 terminated），但 upright 和高度惩罚的存在会提供必要约束。

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
    forward_reward = 20.0 * (body_x_vel / (1.0 + abs(body_x_vel)))

    # -------- 2. health height (soft boundary cost) --------
    height_penalty = 0.0
    if body_z < 0.3:
        height_penalty = -20.0 * (0.3 - body_z)
    elif body_z > 0.9:
        height_penalty = -20.0 * (body_z - 0.9)

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
