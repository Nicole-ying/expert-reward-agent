# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -10.240

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_anchor + progress_reward + soft_landing_near + stability_penalty | 1 | -10.240 | -10.240 | unsolved |
| distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty | 4 | -17.330 | -17.330 | unsolved |
| distance_anchor + progress_reward + soft_landing_bonus + stability_penalty | 1 | -116.830 | -116.830 | unsolved |
| distance_anchor + progress_reward + soft_landing_proximity + stability_penalty | 2 | -187.420 | -196.870 | unsolved |
| shaping_reward + stability_penalty | 1 | -591.150 | -591.150 | unsolved |

## Previous interventions

- iter 2 (score=-187.420, structure=distance_anchor + progress_reward + soft_landing_proximity + stability_penalty): 4. **selected_level**：Level 2 — sparse_to_dense。证据直接否定当前二元合取形态（0.4% active_rate），且上一轮首次设计尚未做 Level 2 变换。 | 5. **selected_intervention**：仅修改 soft_landing_bonus 组件，从稀疏二元合取变换为连续有界乘积代理信号 `soft_landing_proximity`，使用 `max(0, 1 - x/threshold)` 对各维度提供梯度，并保持相同权重 0.5。
- iter 3 (score=-110.470, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): 4. **selected_level**：Level 2 — `dense_to_task_event`。证据直接否定当前无门控连续形态：连续代理在未接触时激活，得分反比稀疏二元版本差70分，需用接触门控将奖励对齐到任务完成条件。 | 5. **selected_intervention**：唯一修改soft_landing组件，改为**接触门控的连续着陆奖励**——仅当`left_contact>0.5`或`right_contact>0.5`时才激活连续梯度乘积，同时放宽阈值（dist→1.0, vel→0.5, angle→0.3, angvel→0.3）使着陆条件比Iter 1更可达，w_landing提至1.0补偿门控后的稀疏性。
- iter 4 (score=-110.830, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): 4. selected_level: Level 2：`global_to_local_gate`。stability_penalty的数学形态对接近阶段约束不足，证据直接表明agent在目标附近高速crash。该组件职责正确（安全约束），但需要从全局均匀变为接近平台时显著增强。符合"约束在无关阶段妨碍探索"的证据模式。 | 5. selected_intervention
- iter 6 (score=-196.870, structure=distance_anchor + progress_reward + soft_landing_proximity + stability_penalty): `selected_level`：Level 2 — 触发条件为`sparse_to_dense`模式：完成信号因接触门控几乎不触发（episode_sum_mean=0.008），Agent在接触到平台前就已失败，需要将稀疏事件奖励转化为连续过程证据。 | `selected_intervention`：唯一目标组件为`soft_landing_contact_gated`，将其从接触门控（`if contact_active:`）改为距离门控（`dist_factor`自动在dist>threshold时归零），保持乘积形态不变（dist_factor × vel_factor × angle_factor × angvel_factor × contact_boost），使Agent在
- iter 7 (score=-71.150, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): 4. `selected_level`：Level 2 — `product_to_noncollapsing_joint`。证据直接表明乘积塌缩（active_rate=0.6%），不是单纯尺度问题。 | 5. `selected_intervention`：从best代码（iter3，接触门控版）出发，将soft_landing_contact_gated内部四个因子的**乘积**改为**算术平均**，即`dist_factor * vel_factor * angle_factor * angvel_factor` → `(dist_factor + vel_factor + angle_factor + angvel_factor)
- iter 8 (score=-17.330, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): `selected_level`：Level 1。distance_anchor的职责（持续拉力）、符号（负值惩罚距离）和数学形态（线性）均合理，证据主要表明该组件过强（|anchor|/progress ≈ 1.2 > 0.5），且本骨架家族尚未做过尺度修复。 | `selected_intervention`：将w_dist从0.1降至0.01（10倍衰减），使distance_anchor从主导惩罚退化为轻量约束。其他所有参数保持不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
