# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -120.250

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | 3 | -120.250 | -120.250 | unsolved |
| angle_penalty + contact_quality + goal_proximity + soft_success + thrust_penalty + velocity_penalty | 1 | -1471.590 | -1471.590 | unsolved |

## Previous interventions

- iter 2 (score=-183.320, structure=angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty): 4. `selected_level`：Level 2 – 修改目标组件的数学形式。因为 success_proxy 的 active_rate 仅 0.1%，单纯调整权重（Level 1）无法改变激活频率；必须将其从稀疏的乘积门控转换为更容易触发、提供连续梯度的加性奖励，同时保持权重合理以避免淹没原有惩罚。 | 5. `selected_intervention`：仅修改 `success_proxy` 组件。将其公式从基于几何平均的乘积门控改为对位置、速度、姿态、接触状态的线性衰减得分加和，每一项独立给出0到1之间的正向奖励，之和乘以一个调低的权重（w_proxy 由 20.0 降至 2.0），使得在非完美状态下也能提供逐步增强的正向拉力。
- iter 3 (score=-120.250, structure=angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty): 4. `selected_level`：Level 2 — 修改goal_proximity的数学形态，将其从无界线性惩罚改为有界惩罚，以消除全覆盖惩罚压制探索和诱导虚假补偿的根源。 | 5. `selected_intervention`：唯一目标是goal_proximity，将 `comp_prox = w_prox * ( -abs(x) - abs(y) )` 改为 `comp_prox = w_prox * ( - (x**2 + y**2) / (1.0 + x**2 + y**2) )`，并调整权重 `w_prox = 5.0`（使远处步惩罚约‑5，近处接近0）。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
