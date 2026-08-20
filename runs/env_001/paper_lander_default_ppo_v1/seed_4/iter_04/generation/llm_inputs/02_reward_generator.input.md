# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
该任务是一个 2D 车辆/着陆器的轨迹优化问题。智能体从视口上方中央附近以随机初始推力出发，必须驾驶自身到达画面中央的目标平台，并在其上稳定停靠（settle）。核心目标是**快速到达并安全停在目标点**，次要目标是**尽量少用引擎推力**以节省燃料。智能体需要学会平滑减速、保持直立姿态，并用两个支撑脚平稳接触平台。任何碰撞、飞出视口或不稳定的翻滚都应避免。不要将此任务与纯加速度最小化或多目标探索混淆，其核心明确是以“精准停靠”为成功的导航到达任务。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**: 任务描述的核心是“到达并停靠在中心目标平台”，这是典型的导航到达任务。尽管有燃料节俭和姿态稳定等附属目标，但成功与否由是否到达并稳定停靠决定，没有其他与到达目标同等权重的冲突目标，因此归入导航到达类。

## 3. 观察空间 observation_space
- **type**: Box (连续)
- **shape**: (8,)
- **dtype**: float32 (推断)
- **各维度含义**:

| 索引 | 名称 | 含义 | 奖励中可用 |
|------|------|------|------------|
| 0 | x_position | 相对目标平台的水平坐标 | true |
| 1 | y_position | 相对目标平台高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 机体方向角 | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑脚接触标志 (0/1) | true |
| 7 | right_support_contact | 右支撑脚接触标志 (0/1) | true |

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**:

| 动作值 | 名称 | 含义 |
|--------|------|------|
| 0 | no_engine | 无推进，所有引擎关闭 |
| 1 | left_orientation_engine | 点燃左侧姿态控制引擎（产生顺时针力矩） |
| 2 | main_engine | 点燃主引擎（产生向上的推力） |
| 3 | right_orientation_engine | 点燃右侧姿态控制引擎（产生逆时针力矩） |

- 这些动作直接影响水平/垂直速度和角速度，通过多次交互控制轨迹。燃料消耗与是否点火相关，可通过动作值间接测量。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 中的 **settled** 部分 —— 当机体在目标平台上稳定停靠且不再运动时触发，暗示成功。
- **failure-like termination**: `crash_or_body_contact` （非平台接触的碰撞，如撞击地面/边界）以及 `horizontal_position_outside_viewport` （水平飞出视口）。
- **ambiguous termination**: `body_not_awake_or_settled` 同样包括“机体失去活动能力（未唤醒）”的情况，可能是体力耗尽、卡住等不成功状态，需结合其他指标判断。
- **truncation**: 无显式最大步数截断，但环境可能隐式包含。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（step 返回的 info 为空字典）
- **forbidden_or_uncertain_info_fields**: 任何假设的 `info["success"]` 或 `info["failure"]` 均不存在，禁止在奖励函数中使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`  (当前观察的 8 维向量)
- `action` (当前选择的离散动作 0~3)
- `next_obs` (下一观察，可用于差分或接触检测)
- `info` 中只能使用该环境允许的字段，但本环境中 `info` 恒为空 `{}`，**不得依赖任何 `info` 字段**
- `training_progress` 仅当 prompt 明确允许时可用，当前未声明，默认为不可用

禁止使用：
- `original_reward`：已完全遮蔽，禁止访问
- 任何未声明的 `info` 字段
- 任何未在观察空间中声明的 `obs` 切片（如越界索引）
- 任何对环境内部状态（如燃料量、真实坐标）的硬编码假设

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`(x_position), `obs[1]`(y_position) → 可计算到目标 (0,0) 的距离、逼近项
- **速度**：`obs[2]`(x_velocity), `obs[3]`(y_velocity) → 可用于速度惩罚、减速奖励
- **姿态**：`obs[4]`(body_angle), `obs[5]`(angular_velocity) → 可用来奖励直立、角速度抑制
- **接触**：`obs[6]`(left_support_contact), `obs[7]`(right_support_contact) → 可探测着陆触地、双脚稳定接触
- **动作/引擎**：`action` 本身可用来惩罚非零推力（动作0为无推力），间接衡量燃料消耗
- **其他**：`next_obs` 可用于差分（如速度变化、接触状态变化）以增强奖励，但需注意其并非独立信号，而是 `obs` 的下一帧。

## 8. 不确定或不可用的信号
- **显式任务成功标志**：info 中无 `success`
- **显式失败原因**：info 中无 `failure_reason`
- **精确燃料量**：观察中无燃料计；只能通过 `action` 是否为 0 来粗略衡量推力是否被使用
- **目标位置绝对坐标**：观察已相对化，目标总是 (0,0)，无需额外信号
- **平台位置或关键点**：只通过接触箱信号间接反映，无法得到精确的平台边界

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D single rigid body (lander / vehicle)
  actuator_type:
    - main_engine (vertical thrust)
    - left_orientation_engine (clockwise torque)
    - right_orientation_engine (counterclockwise torque)
    - no_engine (passive)
  contact_structure: two support feet (left, right) with binary contact flags
primary_objectives:
  - Reach the target pad (position (0,0) )
  - Settle stably with both feet in contact and near-zero velocity
secondary_objectives:
  - Minimize fuel usage (avoid unnecessary engine firings)
  - Maintain upright orientation (body_angle near 0)
  - Achieve fast arrival (implicit time pressure from sparse reward)
main_failure_risks:
  - Crashing into ground or walls
  - Flying outside horizontal bounds
  - Rotating uncontrollably (tumbling)
  - Hovering indefinitely without landing
  - Wasting thrust and never approaching
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: `goal_proximity`**
  **purpose**: 鼓励智能体向目标 (0,0) 靠近。
  **why_required**: 核心导航需求，没有接近奖励智能体无法学习方向。
  **usable_signals**: `obs[0]`, `obs[1]` → 欧氏/曼哈顿距离；`next_obs[0:2]` 差分。
  **risks**: 距离奖励设计不当可能导致高速冲过目标，需配合减速项。

- **role_id: `safe_landing`**
  **purpose**: 奖励在低速、双支撑接触且姿态良好时完成停靠。
  **why_required**: 真实目标为“settle”，仅位置接近不够，必须平稳接触并停止。
  **usable_signals**: `obs[2]`, `obs[3]` (速度), `obs[4]` (倾角), `obs[6]`, `obs[7]` (接触标志)；可用 `next_obs` 验证接触状态变化。
  **risks**: 该奖励为稀疏事件，需与密集奖励配合，否则不易收敛；阈值设置需谨慎。

- **role_id: `attitude_stability`**
  **purpose**: 惩罚大倾角和角速度，维持直立姿态。
  **why_required**: 过大的倾角可能导致碰撞或无法正确接触支撑脚；直立姿态是成功停靠的重要前提。
  **usable_signals**: `obs[4]`, `obs[5]`.
  **risks**: 与目标接近冲突时（如需要倾斜来减速）可能过度干扰，应使用较小的系数或仅在接近地面时激活。

- **role_id: `fuel_efficiency`**
  **purpose**: 惩罚任何推力动作（动作1,2,3），鼓励使用无推力滑行。
  **why_required**: 任务明确要求“尽可能少用引擎推力”，且离散动作允许直接衡量。
  **usable_signals**: `action` (是否为 0).
  **risks**: 单纯的推力惩罚会让智能体不敢使用引擎，必须与其他奖励平衡，尤其在需要主引擎减速时。

### 10.2 条件职责 conditional_roles
- **role_id: `soft_landing_surge_penalty`**
  **condition_to_use**: 当智能体历史表现出高速触地倾向时启用；或根据 `training_progress` 在后期引入。
  **usable_signals**: `obs[2]`, `obs[3]` (速度), `obs[1]` (高度) → 靠近地面时若垂直速度过大多施加惩罚。
  **risks**: 过早引入会阻碍探索，可通过 `training_progress` 控制强度。

- **role_id: `approach_velocity_adaptation`**
  **condition_to_use**: 当训练出现“悬停不决”或“高速冲撞”两种极端时。
  **usable_signals**: `obs[0]`, `obs[1]` (距离), `obs[2]`, `obs[3]` (速度) → 期望速度随距离减小而降低。
  **risks**: 强加约束可能造成非最小相位动态，需要平滑过渡。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: `time_pressure`**
  **reason**: 任务描述提及“尽可能快”，但时间信息在标准 MDP 中不可直接观测（无步数惩罚），且若通过外部 `training_progress` 调权可能破坏自然探索。当前环境未提供剩余时间信号，不宜作为显式奖励项，改为依赖稀疏成功奖励的 decay 效应。
  **forbidden_or_missing_signals**: 缺少实时时间或剩余步数。

- **role_id: `contact_only_reward`**
  **reason**: 仅依赖 `obs[6]`, `obs[7]` 而不考虑位置和速度容易产生“悬停并用脚轻触”的投机行为，不能保证真正停靠。
  **forbidden_or_missing_signals**: 虽然接触信号可用，但缺少平台受力或稳定指示器，因此不宜单独作为主要成功判据。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | obs[0], obs[1], next_obs[0:2] | - | negative_distance, quadratic_distance, bounded_difference | 常用负欧氏距离或指数衰减，注意避免奖励密度过高 |
| safe_landing



# expert_reward_context.md

# Expert Schema Context（非检索版）

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
