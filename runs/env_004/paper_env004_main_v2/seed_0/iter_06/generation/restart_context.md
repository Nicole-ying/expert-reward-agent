# Fresh Restart Evidence

- target_score: 3800.000
- best_score_so_far: 2926.190

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| ang_vel_penalty + speed_posture_reward | 3 | 2926.190 | 2509.870 | unsolved |
| ang_vel_penalty + speed_posture_reward + vertical_reward | 1 | 2080.000 | 2080.000 | unsolved |
| ang_vel_penalty + energy_penalty + speed_posture_reward | 1 | 724.240 | 724.240 | unsolved |

## Previous interventions

- iter 2 (score=2606.950, structure=ang_vel_penalty + speed_posture_reward): `selected_level`：Level 1（尺度修复）——posture_factor的指数衰减形态和符号均正确，但k_angle=10.0对需要自然躯干振荡的跳跃步态过于严厉，使有效奖励在中高速时被过度压缩。 | `selected_intervention`：单一组件修改——将posture_factor中的k_angle从10.0降至5.0，其余所有组件和系数不变。
- iter 3 (score=2509.870, structure=ang_vel_penalty + speed_posture_reward): `selected_level`: Level 2 — 同一骨架家族已迭代2轮未刷新best，且第2轮尺度修复（k_angle 10→5）未改善反而恶化，触发从Level 1升级到结构变换；证据模式为乘积耦合可能限制任务所需的行为空间。 | `selected_intervention`: 将`speed_posture_reward`从乘法耦合`forward_vel * exp(-k*angle²)`改为加法解耦`forward_vel - w_posture * angle²`，w_posture=5.0（使0.2rad倾角时惩罚约0.2，约为典型forward_vel的13%），ang_vel_penalty保持不变。
- iter 4 (score=724.240, structure=ang_vel_penalty + energy_penalty + speed_posture_reward): `selected_level`: **Level 2**。触发条件：必要职责缺失（动作效率约束），且Level 1仅调系数已在iter1→iter2→iter3中反复尝试未刷新best。 | `selected_intervention`: 在iter1乘法骨架基础上**新增energy_penalty组件**（对三关节扭矩的平方和施加轻量惩罚），不改变speed_posture_reward和ang_vel_penalty的现有数学形态。
- iter 5 (score=2080.000, structure=ang_vel_penalty + speed_posture_reward + vertical_reward): `selected_level`: Level 2 — 需要加入新的跳跃引导信号来缩小2926→3800的差距，不能简单回退best。 | `selected_intervention`: 以best代码为基础，新增`vertical_reward`组件：用`max(0, next_obs[6])`奖励向上的垂直速度，系数0.3，鼓励更有力的跳跃以提升前进效率。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
