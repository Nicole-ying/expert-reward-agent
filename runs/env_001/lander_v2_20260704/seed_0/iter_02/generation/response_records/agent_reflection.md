# Response Record

## 分析输出

**evidence**：上一轮所有20个episode全部提前终止（terminated=20/20），score=-518.69，len=62.35。distance_reward=-62.71占96.2%幅度且恒为负值（-dist_to_target），stability_penalty=-2.50占3.8%，soft_landing_proxy触发率为0%。

**behavior_diagnosis**：agent在~62步内全部坠毁，并未向目标平台靠近。距离奖励始终为负值，agent学到的策略是尽快结束episode以停止累积惩罚，而非向目标移动。

**signal_completeness**：缺少正向进展信号——当前距离是状态值而非改善量，无法区分"接近目标"与"远离/坠毁"。完工信号soft_landing_proxy虽存在但完全不可达（0%触发）。稳定性约束存在但过弱，无法在主导性负奖励下阻止坠毁。

**selected_level**：Level 2 — distance_reward的数学形态（-dist_to_target恒负）对进展引导职责不合理，属于state_to_improvement变换，非单纯尺度问题。

**selected_intervention**：将distance_reward从状态值`-dist_to_target`变换为改善量`(dist_before - dist_after) * 5.0`。接近目标时获得正奖励，远离时受罚。系数5.0补偿单步距离增量量级较小的问题，使其与stability_penalty可比。

**falsifiable_hypothesis**：正奖励信号应让agent学会向目标靠拢而非快速坠毁，episode变长、得分改善。如果agent学会接近但悬停不降落，下一轮需对soft_landing_proxy做连续化。

**expected_next_round**：episode_length应显著增加（>62），score应从-518向零或正值改善，distance_reward应出现正值或混合值而非全负。soft_landing_proxy可能仍为0%。

**main_risk**：agent可能学会接近目标后悬停而不完成着陆（缺乏接触激励），且悬停期间每步delta≈0会使distance_reward趋零，缺乏推进力。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current distance to target (from current obs)
    dist_before = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # Next distance to target (from next_obs)
    dist_after = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    # 1. Progress signal: reward reduction in distance, penalize increase
    #    Coefficient 5.0 scales the typically small per-step delta (~0.01-0.03)
    #    to be comparable with other reward components
    distance_reward = (dist_before - dist_after) * 5.0

    # 2. Stability constraint: discourage high speed, tilt, and rotation
    stab_weight = 0.01
    stability_penalty = -stab_weight * (
        abs(next_obs[2]) + abs(next_obs[3]) + abs(next_obs[4]) + abs(next_obs[5])
    )

    # 3. Soft landing proxy: bonus when near target, low speed, upright, both legs contact
    near_target = dist_after < 0.2
    low_speed = abs(next_obs[2]) < 0.2 and abs(next_obs[3]) < 0.2
    stable_angle = abs(next_obs[4]) < 0.1
    low_ang_vel = abs(next_obs[5]) < 0.1
    both_legs = (next_obs[6] == 1.0) and (next_obs[7] == 1.0)

    if near_target and low_speed and stable_angle and low_ang_vel and both_legs:
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    total_reward = distance_reward + stability_penalty + soft_landing_proxy

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```
