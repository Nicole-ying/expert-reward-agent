# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -107.730

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty | 4 | -107.730 | -107.730 | unsolved |
| attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + settling_stability | 1 | -121.510 | -121.510 | unsolved |

## Previous interventions

- iter 2 (score=-110.570, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): selected_level: L3: 得分始终接近负值，远低于目标 200，且主信号框架将距离改进错误地导向跌落，符合“score 始终接近零或负→重写整个 reward，换完全不同的主信号框架”的规则。第一轮即发现根本错误，必须重构。 | selected_intervention: 完全重写奖励函数，采用经典着陆器势能差框架：
- iter 3 (score=-110.090, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): selected_level**：**L2: 理由：仅调系数（L1）无法解决“远距离速度被错误惩罚”的结构性问题，必须改变 `speed_penalty` 的数学形态，使其依赖距离。 | selected_intervention**：
- iter 4 (score=-107.730, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): `selected_level`：**L2: 理由：增减系数（L1）无法解决着陆奖励稀疏的本质问题；必须修改 landing_contact 组件的数学形态，将其从离散终态事件变为连续函数。 | `selected_intervention`：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
