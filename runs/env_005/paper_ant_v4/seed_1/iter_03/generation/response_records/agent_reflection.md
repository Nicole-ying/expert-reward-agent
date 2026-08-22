# Response Record

1. **evidence**：当前episode满1000步且无终止，height_penalty全程未激活，但forward_progress总和675远低于upright_stability总和970，导致策略以极慢速度（约0.034 m/s）稳定移动，外部score跌至‑224.8，远低于上一轮best的434.5。  
2. **behavior_diagnosis**：策略陷入过度追求姿态直立的保守行为，仅以蠕动式前移获取正向奖励，缺乏真正的快速前进，外部任务表现严重退化。  
3. **signal_completeness**：前向速度、直立姿态、高度边界和侧向漂移等必要职责均已覆盖且可达，但前向激励相对直立奖励过弱，未形成有效主导信号。  
4. **selected_level**：Level 1，因forward_progress数学形态合理但相对upright_stability的尺度不足，只需调整其系数，不改变结构。  
5. **selected_intervention**：将forward_progress的缩放系数从20.0提高至60.0，其他所有组件保持不变。  
6. **falsifiable_hypothesis**：增强前向奖励会使总奖励中速度分量的权重超过直立分量，迫使策略探索更快步态以最大化累计奖励，从而提升外部任务评价。  
7. **expected_next_round**：episode仍应存活满1000步，forward_progress的episode_sum_mean将明显上升，upright_stability可能小幅下降但仍保持正值，外部score将回升至400+区域。  
8. **main_risk**：系数过高可能导致策略为追求速度牺牲稳定性，触发height_penalty或频繁终止，反而拉低外部score。  

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
