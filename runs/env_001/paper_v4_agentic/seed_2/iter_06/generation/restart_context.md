# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -110.220

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| orientation_penalty + proximity_delta + velocity_danger | 1 | -110.220 | -110.220 | unsolved |
| landing_bonus + orientation_penalty + proximity_delta + velocity_danger | 2 | -111.880 | -111.880 | unsolved |
| orientation_penalty + proximity_delta + soft_approach_bonus + velocity_danger | 1 | -115.170 | -115.170 | unsolved |
| orientation_penalty + proximity_delta + velocity_penalty | 1 | -116.460 | -116.460 | unsolved |

## Previous interventions

- iter 4 (score=-111.880, structure=landing_bonus + orientation_penalty + proximity_delta + velocity_danger): 修改方案属于 **Level 1 尺度修复**：保持连续乘积形式，但大幅松弛门控阈值（距离衰减系数 0.3 → 1.0，速度/角度截止 0.3 → 0.5，使因子能在常见的着陆前状态中被激活），并将权重从 80.0 降为 20.0，避免单步奖励过度支配总回报（校准：预计激活时 per‑step ≤ 20 × 0.1~0.3 = 2~6，不超过主信号 proximity_delta 的 2~3 倍，可接受）。其他组件保持不变。
- iter 5 (score=-115.170, structure=orientation_penalty + proximity_delta + soft_approach_bonus + velocity_danger): 本轮修改一个组件：将 `landing_bonus`（僵尸组件，active_rate=0%）替换为 `soft_approach_bonus`。原组件依赖腿接触标志与严格速度阈值，从未被激活；agent 的终止模式表明它在高速撞击中 crash，没有机会产生腿接触或满足窄阈值。新组件去除腿接触依赖，使用连续的 y 高度 gate 与速度、角度因子，在接近着陆垫低高度且速度、姿态良好时给予正奖励，从而提供可学习的软着陆梯度。数学形式为三

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
