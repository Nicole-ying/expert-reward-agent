`evidence`：iter4在iter3 best基础上将landing_quality从状态值改为势能差（state_to_improvement），导致episode长度从1000骤降到68.5、score从-17.84暴跌到-110.72，所有20条轨迹均在150步内提前终止；landing_quality的active_rate仅16.7%，说明势能差形式在早期探索中几乎不提供引导。iter3 best的问题则是状态型landing_quality让agent在目标附近徘徊1000步（per-step约1.0），但双腿从未稳定着陆，外部score仍为负。

`behavior_diagnosis`：iter3 best的agent学会了在目标附近保持良好姿态以持续获取landing_quality奖励，但没有动力完成双腿着陆——contact_score仅作为四个等权因子之一被平均，零接触也能获得75%的最大值。iter4的势能差修正过度，信号过于稀疏，agent在早期就撞毁。

`signal_completeness`：当前信号职责基本完备——distance_reward提供进展引导（100% active）、stability_penalty提供动作约束（100% active）、landing_quality本应提供着陆质量信号。但landing_quality的数学形态使"靠近+姿态好"与"真正着陆"之间的奖励差异太小（加权和允许零接触补偿），导致proxy与外部任务错位。

`selected_level`：Level 2 — 触发条件为proxy_to_completion_alignment：状态型proxy使agent可绕过任务完成（双腿着陆）而持续获得高奖励，外部score为负而proxy奖励高；上一轮state_to_improvement失败证明不能用纯势能差替代，需要对状态型proxy做结构变换使其更贴近任务完成条件。

`selected_intervention`：仅修改landing_quality组件，从加权和形态变为contact作为软乘数的乘法结构。将contact_score从加性因子改为软地板乘法因子`contact_factor = 0.3 + 0.7 * contact_raw`，pose部分仅保留speed/angle/angvel三因子的均值，landing_quality = 5.0 * proximity_gate * contact_factor * pose_quality。系数从2.0调至5.0以匹配新值域（无接触时理论最大值≈0.6→实际≈0.4-0.5，与best版本per-step约1.0的一半相当，保留引导但不让无接触状态过于舒适）。

`falsifiable_hypothesis`：contact软乘数结构使零接触时的landing_quality上限从原来的1.5降至约0.6（proximity_gate=1, pose=1时），而双腿着陆时可达5.0，形成约8倍的奖励梯度。这应迫使agent不再满足于悬停，而是主动寻求双腿接触来完成着陆；同时0.3的软地板保留了接近阶段的连续引导，避免像势能差版本那样完全失去信号。

`expected_next_round`：episode长度应显著回升（不再大量早期坠毁），landing_quality的magnitude_share可能下降但active_rate应保持较高，外部score应相对于iter3 best的-17.84改善（向0或正数靠近），terminated中应有部分episode达成双腿着陆而非全部撞毁。

`main_risk`：contact因子的引入使"接近但未着陆"的奖励减半，如果distance_reward（best中per-step仅0.016）不足以弥补，agent可能在接近阶段就因总奖励过低而失去动力，重新出现早期坠毁。若下一轮出现此情况，需加强distance_reward或调整软地板值。

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

    # 3. Landing quality with contact as soft-multiplier
    # Proximity gate: activates gradually within 0.5 distance of target
    proximity_gate = max(0.0, 1.0 - dist_after / 0.5)

    # Pose quality factors: speed, angle, angular velocity (each in [0, 1])
    speed_quality = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)
    angle_quality = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    angvel_quality = max(0.0, 1.0 - abs(next_obs[5]) / 0.3)

    # Contact factor: soft floor at 0.3, scales to 1.0 with both legs down
    contact_raw = (next_obs[6] + next_obs[7]) / 2.0
    contact_factor = 0.3 + 0.7 * contact_raw

    # Pose quality without contact (average of three factors)
    pose_quality = (speed_quality + angle_quality + angvel_quality) / 3.0

    # Landing quality: proximity-gated, contact-multiplied pose quality
    landing_quality = 5.0 * proximity_gate * contact_factor * pose_quality

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```