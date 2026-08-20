# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -17.840

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_reward + landing_quality + stability_penalty | 4 | -17.840 | -73.820 | unsolved |
| distance_reward + soft_landing_proxy + stability_penalty | 2 | -85.750 | -85.750 | unsolved |

## Previous interventions

- iter 2 (score=-85.750, structure=distance_reward + soft_landing_proxy + stability_penalty): selected_level**：Level 2 — distance_reward的数学形态（-dist_to_target恒负）对进展引导职责不合理，属于state_to_improvement变换，非单纯尺度问题。 | selected_intervention**：将distance_reward从状态值`-dist_to_target`变换为改善量`(dist_before - dist_after) * 5.0`。接近目标时获得正奖励，远离时受罚。系数5.0补偿单步距离增量量级较小的问题，使其与stability_penalty可比。
- iter 3 (score=-17.840, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2** — the `soft_landing_proxy` active_rate is near zero and its hard-binary mathematical form collapses feedback, directly matching the `sparse_to_dense` trigger: "completion/landing bonus 的 acti | 5. selected_intervention
- iter 4 (score=-110.720, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 2 — `state_to_improvement`。证据模式直接命中"占据好状态即可持续获奖"（532的landing_quality vs 0次着陆），单靠调系数（Level 1）无法消除悬停激励的结构性漏洞。上一轮（iter 3）已通过Level 2将sparse soft_landing_proxy改为连续landing_quality并成功消除坠毁模式，但引入了新的proxy fa | selected_intervention**：将landing_quality从**持续状态值**改为**状态改善量**（potential-based shaping）：`reward = scale * (quality_potential(next_obs) - quality_potential(obs))`，使用有符号差分以避免max(0,⋅)带来的振荡套利风险，scale设为10.0使单回合总贡献上限约20（quality
- iter 5 (score=-23.320, structure=distance_reward + landing_quality + stability_penalty): `selected_level`：Level 2 — 触发条件为proxy_to_completion_alignment：状态型proxy使agent可绕过任务完成（双腿着陆）而持续获得高奖励，外部score为负而proxy奖励高；上一轮state_to_improvement失败证明不能用纯势能差替代，需要对状态型proxy做结构变换使其更贴近任务完成条件。 | `selected_intervention`：仅修改landing_quality组件，从加权和形态变为contact作为软乘数的乘法结构。将contact_score从加性因子改为软地板乘法因子`contact_factor = 0.3 + 0.7 * contact_raw`，pose部分仅保留speed/angle/angvel三因子的均值，landing_quality = 5.0 * proximity_gate * con
- iter 6 (score=-73.820, structure=distance_reward + landing_quality + stability_penalty): `selected_level`：Level 2，触发条件为 `proxy_to_completion_alignment`（proxy 提高但外部任务不升）叠加 `state_to_improvement`（占据好状态即可持续获奖）。上轮 Level 1 尺度调整（加 contact 乘子）已被 evidence 证伪，不可重复调系数。 | `selected_intervention`：将 landing_quality 从状态值奖励转换为势能改善奖励（state_to_improvement）——`landing_quality = scale * (potential(next_obs) - potential(obs))`，其中 potential 使用 proximity、soft-contact、pose 三因子的连续乘积；系数保持 5.0 以匹配新值域。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
