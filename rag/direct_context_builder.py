import re


def extract_selected_route_id(environment_card_md):
    """Keep the legacy route id for logging/backward compatibility only."""
    m = re.search(r"\*{0,2}selected_route_id\*{0,2}\s*:\s*([a-zA-Z0-9_]+)", environment_card_md or "")
    if m:
        return m.group(1).strip()
    print("WARNING: environment_card.md 中未找到 selected_route_id，回退到 unknown_task_route")
    return "unknown_task_route"


EXPERT_SCHEMA_CONTEXT = r"""# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子和少量动力学类型示例，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive: `w * signal`
  - penalty: `-w * abs(error)` 或 `-w * error**2`
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：无界状态值可能支配总奖励；状态值可能被占据/刷分，而不代表任务完成。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - `x / (1 + abs(x))`
  - `1 / (1 + k * abs(error))`
  - `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或 velocity/proximity 类信号容易被刷。
- 风险：threshold 过小会导致反馈饱和或无梯度。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * (1 / (1 + k * abs(posture_error)))`
- 使用条件：前进/接近奖励导致不健康冲刺、翻倒或失稳。
- 风险：gate 太严格会抑制探索；跳跃类任务尤其要轻。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 风险：乘积容易塌缩；单一接触或单一事件不能当成功。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. Expert Task Template A: planar_monoped_hopping

### 适用任务线索
- 平面单腿或少腿跳跃式前进；
- 观测中有 torso_height、torso_angle、forward_velocity、vertical_velocity、joint angles/speeds；
- 动作为连续关节力矩；
- 终止通常与高度、躯干角度或状态非法有关；
- 任务要求 sustained forward progress，而不是只保持直立。

### 主职责 mandatory reward roles
1. sustained_forward_progress
   - 目的：鼓励持续向前运动，而不是短时间冲刺。
   - 可用信号：forward_velocity。
   - 推荐算子：dense_state_signal、bounded_signal、soft_health_gate。
   - 风险：velocity_burst_then_fall、shuffling_without_real_hop。

2. healthy_posture_or_height_constraint
   - 目的：保持身体高度和躯干姿态在健康范围附近，同时允许必要跳跃动态。
   - 可用信号：torso_height、torso_angle、torso_angular_velocity。
   - 推荐算子：quadratic_penalty、bounded_signal、soft_health_gate。
   - 风险：约束过强会抑制跳跃，使 agent 不敢探索。

### 条件职责 conditional reward roles
1. light_energy_regularization
   - 条件：只有当策略已经能产生前进后再加入。
   - 可用信号：action。
   - 推荐算子：quadratic_penalty。
   - 风险：过早加入会导致 energy_freeze。

2. vertical_dynamics_regularization
   - 条件：如果策略只是滑行、乱跳或原地弹跳，再考虑轻量垂直动态约束。
   - 可用信号：vertical_velocity、torso_height。
   - 推荐算子：bounded_signal、quadratic_penalty。
   - 风险：直接奖励 vertical activity 可能导致原地弹跳。

### 慎用/禁用 avoid roles
- bipedal_alternating_contact_reward：单腿任务不适配；
- contact_reward_without_contact_signal：没有接触信号时不能使用；
- unconditional_alive_bonus：容易站着不动或拖时间；
- strong_vertical_activity_reward：容易学成原地弹跳。

---

## 4. Expert Task Template B: multi_legged_body_locomotion

### 适用任务线索
- 多足身体或高维关节身体；
- 动作为连续 torque；
- 目标是沿某一方向持续前进；
- 观测可能包含 body orientation、body velocity、joint positions/speeds、actions；
- 失败通常表现为翻滚、侧向漂移、腿部乱动、能耗过高或原地抖动。

### 主职责 mandatory reward roles
1. directional_forward_progress
   - 目的：鼓励沿目标方向前进。
   - 可用信号：forward_velocity 或对应方向速度。
   - 推荐算子：dense_state_signal、bounded_signal。
   - 风险：sideways_drift、velocity_farming、thrashing_forward。

2. body_orientation_health
   - 目的：避免躯干翻滚、侧翻或极端姿态。
   - 可用信号：torso_orientation、body_angle、angular_velocity。
   - 推荐算子：quadratic_penalty、bounded_signal、soft_health_gate。
   - 风险：过强会导致 over_conservative_policy。

### 条件职责 conditional reward roles
1. light_action_energy_regularization
   - 条件：策略已有明显前进后再加入。
   - 可用信号：action。
   - 推荐算子：quadratic_penalty。
   - 风险：过强会导致 agent_afraid_to_move。

2. lateral_drift_control
   - 条件：只有当侧向速度或姿态信号明确可用时使用。
   - 可用信号：side_velocity、body_orientation。
   - 推荐算子：quadratic_penalty、bounded_signal。
   - 风险：如果没有明确信号，不得伪造。

### 慎用/禁用 avoid roles
- bipedal_alternating_gait：多足身体不一定需要双足交替步态；
- monoped_vertical_hopping：多足行走不应被引导成跳跃；
- contact_reward_without_contact_signal：没有接触信号时不能使用；
- unconditional_alive_bonus：容易导致站立或原地抖动。

---

## 5. reward_v1 生成要求

- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确职责，不能为了显得完整而堆叠。
- 以 environment_card.md 的 reward_role_decomposition 为主，本文件模板为辅。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类职责、复杂门控和 curriculum_weighting 默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 `return float(total_reward), components`；components 必须是 dict。
"""


def build_expert_reward_context(environment_card_md, chunks_path=None, max_chars=6500):
    """Build a fixed expert-schema context without retrieval.

    The route id is returned only for backward-compatible logging. The content is a
    compact expert schema used by the reward generator; it is not a RAG result.
    """
    route_id = extract_selected_route_id(environment_card_md)
    md = EXPERT_SCHEMA_CONTEXT
    if len(md) > max_chars:
        md = md[:max_chars] + "\n\n<!-- truncated to max_expert_context_chars -->\n"
    return route_id, md
