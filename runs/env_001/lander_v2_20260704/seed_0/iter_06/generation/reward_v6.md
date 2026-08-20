`evidence`：当前评估中 agent 活满 1000 步（truncated=20/20），从未成功着陆（terminated=0/20），landing_quality 单步均值约 0.484、episode_sum_mean 高达 484.42 且占 magnitude_share 98.2%，外部得分 -23.32 比 iter3（-17.84，landing_quality 约 1.0/步但无 contact 软乘子）更差——说明上轮改成 contact 乘子后 proxy 仍在被大量收割，外部指标反而恶化。

`behavior_diagnosis`：agent 已学会靠近目标并维持良好姿态（landing_quality active_rate 79.3%），但停留在"近目标+好姿态"的舒适区持续收割状态奖励，从不执行完成着陆的最后一步——典型的 proxy farming，即持续占有好状态即可获得高额奖励，无需真正完成双腿触地、停止的着陆。

`signal_completeness`：distance_reward 提供接近引导，stability_penalty 约束动作平滑，但 landing_quality 是一个纯状态值奖励——占据好状态即可持续获奖，缺少"从好状态到着陆完成"的进展激励，导致 proxy 与外部任务脱节。

`selected_level`：Level 2，触发条件为 `proxy_to_completion_alignment`（proxy 提高但外部任务不升）叠加 `state_to_improvement`（占据好状态即可持续获奖）。上轮 Level 1 尺度调整（加 contact 乘子）已被 evidence 证伪，不可重复调系数。

`selected_intervention`：将 landing_quality 从状态值奖励转换为势能改善奖励（state_to_improvement）——`landing_quality = scale * (potential(next_obs) - potential(obs))`，其中 potential 使用 proximity、soft-contact、pose 三因子的连续乘积；系数保持 5.0 以匹配新值域。

`falsifiable_hypothesis`：势能差形式使 agent 在稳态悬浮时获得零奖励（potential 不变化），仅在实际改善着陆状态时才获得正奖励；这应迫使 agent 从 proxy 舒适区移动至真正完成着陆，从而提高 terminated 比例并改善外部得分。

`expected_next_round`：landing_quality 的 episode_sum_mean 和 magnitude_share 应大幅下降（不再持续积累），active_rate 应反映实际进展而非稳态停留；terminated 比例应从 0/20 上升；外部 score 应不再恶化甚至改善。

`main_risk`：势能函数中 contact 软底（0.3）可能仍允许近距离无接触悬浮时获得微弱改善奖励，如果学到的策略是反复微调姿态而非着陆，需进一步收紧 contact 门控或在 Level 3 加入完成事件 bonus。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next distance to target
    dist_before = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_after = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Progress signal: reward reduction in distance, penalize increase
    distance_reward = (dist_before - dist_after) * 5.0

    # 2. Stability constraint: discourage high speed, tilt, and rotation
    stab_weight = 0.01
    stability_penalty = -stab_weight * (
        abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[4]) + abs(next_obs[5])
    )

    # 3. Landing quality as potential-based improvement (state_to_improvement)
    # --- Potential for current state (before action) ---
    proximity_before = max(0.0, 1.0 - dist_before / 0.5)
    contact_raw_before = (obs[6] + obs[7]) / 2.0
    contact_factor_before = 0.3 + 0.7 * contact_raw_before
    speed_factor_before = max(0.0, 1.0 - (abs(obs[2]) + abs(obs[3])) / 0.5)
    angle_factor_before = max(0.0, 1.0 - abs(obs[4]) / 0.3)
    angvel_factor_before = max(0.0, 1.0 - abs(obs[5]) / 0.3)
    pose_quality_before = (speed_factor_before + angle_factor_before + angvel_factor_before) / 3.0
    potential_before = proximity_before * contact_factor_before * pose_quality_before

    # --- Potential for next state (after action) ---
    proximity_after = max(0.0, 1.0 - dist_after / 0.5)
    contact_raw_after = (next_obs[6] + next_obs[7]) / 2.0
    contact_factor_after = 0.3 + 0.7 * contact_raw_after
    speed_factor_after = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)
    angle_factor_after = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    angvel_factor_after = max(0.0, 1.0 - abs(next_obs[5]) / 0.3)
    pose_quality_after = (speed_factor_after + angle_factor_after + angvel_factor_after) / 3.0
    potential_after = proximity_after * contact_factor_after * pose_quality_after

    # Landing improvement: reward increase in potential
    landing_quality = (potential_after - potential_before) * 5.0

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```