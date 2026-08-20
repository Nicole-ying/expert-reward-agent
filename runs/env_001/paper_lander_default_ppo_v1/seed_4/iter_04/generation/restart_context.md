# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 187.040

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | 1 | 187.040 | 187.040 | unsolved |
| contact_bonus + orientation_penalty + proximity_reward + speed_penalty_gated | 2 | 126.730 | 126.730 | unsolved |

## Previous interventions

- iter 2 (score=126.730, structure=contact_bonus + orientation_penalty + proximity_reward + speed_penalty_gated): `selected_level`：Level 2 — proximity_reward当前为无界状态值（每步-distant），满足"state_to_improvement"证据模式：agent在远处每步承受大额负奖励，靠近后信号强度急剧衰减，而着陆组件因相对尺度过小无法接管；仅调系数（Level 1）无法消除"待在远处就持续受罚"的结构性问题。 | `selected_intervention`：唯一修改proximity_reward，从无界状态值`-distance`变为势能差分`2.0 * (distance - next_distance)`（正=靠近，负=远离）。其他三个组件（orientation_penalty、speed_penalty_gated、contact_bonus）完全不改动。
- iter 3 (score=187.040, structure=orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated): 4. selected_level: Level 2**: `persistent_to_transition_event` / `proxy_to_completion_alignment` — the contact_bonus is a state-value proxy that can be farmed indefinitely without task completion; it must be restructured | 5. selected_intervention

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
