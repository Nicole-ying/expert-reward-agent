# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 149.440

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| approach_reward + soft_landing_proxy + stability_penalty | 4 | 149.440 | 136.730 | unsolved |

## Previous interventions

- iter 2 (score=149.440, structure=approach_reward + soft_landing_proxy + stability_penalty): `selected_level`：Level 1——stability_penalty的数学形态（加权绝对值和）本身合理，证据主要指向其系数过强（|penalty/progress| ≈ 1.09 > 0.5），先做尺度修复。 | `selected_intervention`：仅将stability_penalty三个系数全部降低10倍——w_speed: 0.01→0.001, w_angle: 0.1→0.01, w_angvel: 0.05→0.005，使|penalty/progress|目标值≈0.11，接近轻约束起点0.1。
- iter 3 (score=-10.510, structure=approach_reward + soft_landing_proxy + stability_penalty): `selected_level`：Level 1。approach_reward 的职责、符号和数学形态均合理，证据仅指向其相对 soft_landing_proxy 过弱（累计比 ≈1:67），不需要结构变换。 | `selected_intervention`：将 approach_reward 乘以系数 25.0，其他组件不变。预期总 approach_reward 从 ~1.07 升至 ~26.75，约为 soft_landing_proxy 的 37%，使进度信号在早期学习中具备实际引导力。
- iter 4 (score=136.730, structure=approach_reward + soft_landing_proxy + stability_penalty): `selected_level`：Level 1。soft_landing_proxy 在 best 轮（iter 2）中曾正常激活（sum=1.728），说明其数学形态在合理 approach 尺度下可达；当前 active_rate=0 是 25x approach 的后果，不是 proxy 结构的固有缺陷。 | `selected_intervention`：仅修改 approach_reward 系数，从 25.0 降至 3.0。best 的 1.0 已验证可行，25.0 已验证失败，3.0 是介于两者之间的中等强度，保留"尽可能快到达目标"的任务激励同时避免压倒着陆信号。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
