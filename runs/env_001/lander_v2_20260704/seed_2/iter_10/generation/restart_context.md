# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -33.720

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_reward + landing_quality + stability_penalty | 7 | -33.720 | -111.980 | unsolved |
| progress_delta + weighted_stability_penalty | 1 | -109.220 | -109.220 | unsolved |
| distance_reward + soft_landing_bonus + stability_penalty | 1 | -110.970 | -110.970 | unsolved |

## Previous interventions

- iter 2 (score=-112.960, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2** — the 0.5% active_rate binary bonus matches the `sparse_to_dense` evidence pattern exactly; a coefficient tweak cannot fix structural sparsity. | 5. selected_intervention
- iter 4 (score=-109.790, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 1 —— 稳定性惩罚的职责和数学形态合理，但尺度过强阻碍早期探索和必要机动。上一轮修改了landing_quality形态（已生效），本轮不应重复修改同一组件；stability_penalty系数尚未被调整过，且ratio虽低于0.5但处于crash场景下的"过度约束"状态。 | selected_intervention**：仅调整stability_penalty的四个权重系数，将其整体幅度降至约40%（w_vx: 0.15→0.06, w_vy: 0.05→0.02, w_angle: 0.2→0.08, w_angvel: 0.2→0.08）。distance_reward和landing_quality保持不变。
- iter 6 (score=-33.720, structure=distance_reward + landing_quality + stability_penalty): `selected_level`：Level 2——着陆信号缺失属于必要职责不完备，且Level 1单纯降稳定性系数无法弥补缺失的终端引导。 | `selected_intervention`：以best代码为基础，将landing_quality从乘积几何平均(product^0.2)改为和基联合满足(sum_of_factors)，保留distance_reward和light stability_penalty不变。
- iter 7 (score=-108.690, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 1。landing_quality 与 distance_reward 量级抵消，且 landing_quality 过大是形成舒适区的直接原因，应通过系数调整重建相对梯度，不改变数学形态。 | selected_intervention**：仅将 landing_quality 系数从 0.2 降至 0.1，其他组件不变。
- iter 8 (score=-34.070, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2：同一骨架家族已在 iter 3/4/6/7 迭代 4 轮，3/4 次得分为 -108 左右，仅 iter 6 偶然成功，说明当前数学结构存在系统性训练不稳定。需要对 distance_reward 做 `unbounded_to_bounded` 结构变换，将线性无界惩罚改为有界饱和形式，从结构上消除"冲撞是局部最优"的问题。 | 5. selected_intervention
- iter 9 (score=-111.980, structure=distance_reward + landing_quality + stability_penalty): 4. `selected_level`：Level 2——`state_to_improvement`变换。证据直接表明持久状态奖励被利用（长episode、高landing_quality累积、低外部进展），且上轮指数距离修改未消除悬停行为，需要结构变换而非尺度调整。 | 5. `selected_intervention`：将landing_quality从状态值`0.2 * sum_of_factors(next_obs)`变换为势能差`5.0 * (sum_of_factors(next_obs) - sum_of_factors(obs))`，系数5.0匹配改善形式的值域（单episode最大改善约5.0→总贡献约25，与distance/stability可比较）。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
