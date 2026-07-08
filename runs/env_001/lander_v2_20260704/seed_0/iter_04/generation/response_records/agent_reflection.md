# Response Record

## 诊断与决策

**evidence**：当前得分为-17.84，全部20条轨迹均被截断在1000步（terminated=0/20），无任何终止事件；landing_quality的episode_sum_mean高达532.39（signed_share 98.2%），是distance_reward（6.11）的87倍，与外部得分-17.84形成巨大反差。

**behavior_diagnosis**：策略已学会在目标附近悬停以持续获取landing_quality的state-based奖励（每步约0.53，1000步累计532），但从未触发着陆终止——处于典型的proxy farming平台期，占据好状态即可持续获奖，无需完成任务。

**signal_completeness**：distance_reward提供进展引导、stability_penalty提供动作约束，职责基本完备。但landing_quality作为持续状态奖励，其数学形态导致"悬停最优"——agent在接近目标且姿态良好时获得最高奖励，不需要实际触地即可最大化累计回报。缺少"完成着陆"与"保持好状态"的区分。

**selected_level**：Level 2 — `state_to_improvement`。证据模式直接命中"占据好状态即可持续获奖"（532的landing_quality vs 0次着陆），单靠调系数（Level 1）无法消除悬停激励的结构性漏洞。上一轮（iter 3）已通过Level 2将sparse soft_landing_proxy改为连续landing_quality并成功消除坠毁模式，但引入了新的proxy farming问题。

**selected_intervention**：将landing_quality从**持续状态值**改为**状态改善量**（potential-based shaping）：`reward = scale * (quality_potential(next_obs) - quality_potential(obs))`，使用有符号差分以避免max(0,⋅)带来的振荡套利风险，scale设为10.0使单回合总贡献上限约20（quality范围[0,2] × 10），与distance_reward的episode_sum_mean（~6）形成合理梯度而不再碾压。

**falsifiable_hypothesis**：改为改善量后，悬停时landing_quality≈0（状态不再变化），agent无法通过维持好状态获取持续回报；只有实际改善姿态、减速、降低距离或最终触地才能获得正向奖励。这将迫使agent从悬停转向完成着陆，episode应出现terminated事件，landing_quality的episode_sum_mean应从532大幅下降，外部得分应上升。

**expected_next_round**：landing_quality的episode_sum_mean将从532降至20以下（受potential差上限约束），active_rate因悬停段不再触发而下降；distance_reward将成为累计奖励的主要组成部分；期望出现terminated>0的轨迹，score应转正并刷新best（-17.84）。

**main_risk**：有符号差分在agent暂时牺牲质量（如加速下降）时会产生负奖励，可能使agent过于保守、不敢接近目标。若下一轮出现episode缩短（坠毁回潮）或agent完全不敢靠近目标（distance_reward趋零），则需考虑切换回max(0, diff)或降低scale。

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

    # 3. Landing quality as improvement (potential-based shaping)
    def quality_potential(o):
        d = (o[0] ** 2 + o[1] ** 2) ** 0.5
        proximity_gate = max(0.0, 1.0 - d / 0.5)
        speed_quality = max(0.0, 1.0 - (abs(o[2]) + abs(o[3])) / 0.5)
        angle_quality = max(0.0, 1.0 - abs(o[4]) / 0.3)
        angvel_quality = max(0.0, 1.0 - abs(o[5]) / 0.3)
        contact_score = (o[6] + o[7]) / 2.0
        return 2.0 * proximity_gate * (
            speed_quality + angle_quality + angvel_quality + contact_score
        ) / 4.0

    quality_before = quality_potential(obs)
    quality_after = quality_potential(next_obs)

    # Improvement-based: only reward change in landing quality
    landing_quality = (quality_after - quality_before) * 10.0

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```
