# Expert reward-design reasoning core

Do not start from a named formula. First evaluate the reward as a learning system.

## Mandatory expert questions

1. **Reachability**: Can a random or early policy encounter a non-trivial signal? Identify signals that are effectively absent because their event or gate is too rare.
2. **Directionality**: Does a local task improvement produce correctly signed feedback? Distinguish rewarding a good state from rewarding movement toward that state.
3. **Scale**: Can one term numerically suppress the main task drive or dominate the return? Respect temporal semantics: dense differences, persistent state values, penalties, and rare events are not directly comparable by one fixed ratio.
4. **Credit assignment**: How many actions or timesteps separate useful behavior from feedback? Decide whether denser evidence, a transition event, or a potential difference is needed.
5. **Task alignment**: Does maximizing the proxy imply completing the external task, or can the policy optimize an intermediate state forever?
6. **Exploit resistance**: Can the policy profit from inaction, hovering, oscillation, repeated event triggering, unsafe speed, or rapid termination?

Unknown evidence must remain unknown. Component statistics describe the visited policy distribution; they do not prove causal contribution.

## Functional component roles

Use roles to decide what the reward must accomplish before choosing formulas:

- `task_completion`: distinguish actual completion or a verified completion transition.
- `process_progress`: provide directional evidence toward task completion.
- `state_quality`: value a useful state such as stability, proximity, or posture.
- `safety_constraint`: discourage unsafe states or transitions.
- `action_cost`: discourage energy, magnitude, or abrupt control when task competence exists.
- `time_efficiency`: prefer faster completion without creating a fast-failure incentive.
- `survival_maintenance`: preserve valid operation when continued existence is itself required.
- `exploration_drive`: reward novelty or coverage when extrinsic evidence is insufficient.
- `stage_event`: represent phase transitions in a multi-stage task.
- `multi_objective_preference`: encode an explicit trade-off among competing objectives.

Not every reward needs every role. It needs the smallest set whose responsibilities make the task learnable and aligned. A diagnostic modulator or gate is not an additive reward role.

## Mathematical structures

Select a mathematical structure after selecting a role:

- `state_value`: reward the quality of the current or next state; dense but may reward occupancy.
- `state_improvement`: reward the signed change in quality; directional but may telescope or be noisy.
- `potential_difference`: use gamma*Phi(next)-Phi(current); preserves optimal-policy structure under its theoretical assumptions.
- `binary_event`: precise but sparse and difficult to discover.
- `continuous_bounded_proxy`: dense with controlled extremes, but still vulnerable to proxy misalignment.
- `continuous_unbounded_proxy`: expressive but sensitive to physical scale and outliers.
- `additive_term`: independently learnable evidence, but objectives can compensate for one another.
- `multiplicative_joint`: enforces joint satisfaction, but several small factors can collapse the signal.
- `soft_joint`: soft-minimum, geometric mean, or guarded combination that preserves joint semantics without total collapse.
- `local_gate`: activate a constraint only in its relevant phase; a wrong gate can remove learning evidence.
- `transition_event`: reward entering a state rather than occupying it repeatedly.
- `dynamic_weight`: change priorities with verified training progress or task phase; adds non-stationarity.
- `hierarchical_state`: represent explicit task stages; useful only when stage evidence is available.

Knowledge cards are design primitives and risk reminders, not a closed formula menu. New structures are allowed only when their role, temporal semantics, evidence source, and risks are stated explicitly.
