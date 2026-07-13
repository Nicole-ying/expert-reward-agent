# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。智能体从视口顶部中心附近出发，受初始随机力作用，必须尽快飞向并稳定降落在中央目标垫上，同时尽可能少地使用引擎推力。核心目标是精确、快速、安全地到达并停稳在目标点；次要目标是燃料效率。任务要求智能体学会接近目标、减速、保持姿态稳定，并实现双支撑脚的软着陆接触。

---

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`  
- **confidence**: high  
- **reason**: 任务的核心是到达一个明确的空间目标（目标垫），并完成稳定停靠。燃料消耗、时间最优化是附属优化，不是权重相当且冲突的多目标情形，因此不属于 multi_objective_task。  
- **dynamics_subtype**: `goal_approach_and_soft_contact`（趋近目标并实现低速、稳定接触/停靠）

---

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: 隐含为 float32（根据标志位的 1.0/0.0 推断）  

各维度含义与奖励可用性：

| 索引 | 名称          | 含义                                 | reward_usable |
|------|---------------|--------------------------------------|---------------|
| 0    | x_position    | 相对于目标垫中心的水平坐标（单位未知） | true          |
| 1    | y_position    | 相对于垫高度的垂直坐标                 | true          |
| 2    | x_velocity    | 水平线速度                             | true          |
| 3    | y_velocity    | 垂直线速度                             | true          |
| 4    | body_angle    | 机体朝向角（弧度或角度未知）          | true          |
| 5    | angular_velocity | 角速度                              | true          |
| 6    | left_support_contact  | 左支撑脚接触标志（1.0 表示接触） | true          |
| 7    | right_support_contact | 右支撑脚接触标志（1.0 表示接触） | true          |

所有维度均可用于奖励设计，无不可用维度。

---

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  

| action | name                        | meaning                                        |
|--------|-----------------------------|------------------------------------------------|
| 0      | no_engine                   | 不点火，依靠当前动量滑行                      |
| 1      | left_orientation_engine     | 点燃一个姿态引擎，可产生侧向力矩（改变角度）|
| 2      | main_engine                 | 点燃主引擎，提供主要推力（改变速度与位置）    |
| 3      | right_orientation_engine    | 点燃对侧姿态引擎，与动作 1 反向              |

---

## 5. step 与终止条件分析

### 5.1 终止模式
根据 `terminated` 的逻辑组合，判断终止由以下任一条件触发：

- **crash_or_body_contact**  
  包含“撞击”或身体非预期接触（可能包括与垫之外的地形碰撞）。从任务文本“make safe contact”看，与目标垫的正常双支撑接触应不在此范畴，但匿名源代码无法确认。暂认定为“不良接触”导致失败。

- **horizontal_position_outside_viewport**  
  水平出界（远离目标垫，飞出视口）。

- **body_not_awake_or_settled**  
  当身体不再活动（awake）或已足够稳定（settled）时触发。结合“settle at the pad”目标，该条件极可能是**成功降落的标志**：双支撑接触且速度/角速度极低，达到稳定停靠。

因此，推断成功性终止对应于 `body_not_awake_or_settled` 为 true，且在不满足 crash/出界的前提下发生。其余两个为失败性终止。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: **false**  
  `info` 返回空字典，无显式成功标志。
- **explicit_failure_flag_available**: **false**  
  同样无显式失败原因字段。
- **allowed_info_fields**: 无（info 为空，禁止依赖）
- **forbidden_or_uncertain_info_fields**: 所有 info 字段均不可用

智能体必须通过观测本身（位置、速度、接触标志等）间接推断成功或失败，不能依赖环境标志。

---

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观测）
- `action`（当前动作）
- `next_obs`（下一观测）
- `info` 中明确允许的字段（当前无）
- `training_progress` 仅在 prompt 明确指示时使用

禁止使用：
- `original_reward`（被官方屏蔽的原始奖励）
- 任何未声明的 `info` 字段
- 未声明的 `obs` 切片或隐含的全局状态

---

## 7. 可用于奖励函数的信号
- **位置信号**：`x_position`, `y_position`  
- **速度信号**：`x_velocity`, `y_velocity`  
- **姿态信号**：`body_angle`, `angular_velocity`  
- **接触信号**：`left_support_contact`, `right_support_contact`  
- **动作/引擎信号**：`action`（用于衡量推力激活）  
- **其他**：可通过相邻观测差值构造变化量（如速度变化），但需谨慎避免时间步依赖。

---

## 8. 不确定或不可用的信号
- **success / failure 标志**：不存在于 info，不能直接用于奖励。
- **环境真实奖励**：`original_reward` 不可用。
- **步数/剩余时间**：无显式字段，除非从 training_progress 推断，但该参数不可作为主要奖励依据。
- **地面真值接触类型**：无法区分 crash 与软着陆，只能依赖接触标志的组合和速度大小间接判断。

---

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2d_lander_with_two_contact_pads
  actuator_type: discrete_main_engine_and_orientation_engines
  contact_structure: two_side_support_feet
primary_objectives:
  - reach_target_position (x_position ≈ 0, y_position ≈ 0)
  - achieve_stable_settlement (low linear and angular velocity)
  - ensure_safe_dual_contact (both feet touching pad)
secondary_objectives:
  - minimize_engine_usage (fuel efficiency)
  - minimize_time_to_settle (implicitly through episode reward signal)
main_failure_risks:
  - violent_crash_or_high_speed_impact
  - flying_out_of_horizontal_bounds
  - unstable_one_leg_landing
  - excessive_orientation_maneuvers_causing_loss_of_control
  - early_termination_before_dual_contact_settlement
```

---

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: `proximity_to_target`  
  **purpose**: 驱使智能体尽快向目标垫中心（(x_position,y_position)=(0,0)）移动。  
  **why_required**: 任务核心是到达目标，若无此项奖励，智能体无动力接近垫子。  
  **usable_signals**: `x_position`, `y_position`  
  **risks**: 可能造成冲刺过度，忽略减速，需配合速度惩罚。

- **role_id**: `soft_landing_velocity`  
  **purpose**: 在接近目标区域时抑制水平与垂直速度，确保低速软著。  
  **why_required**: 高速撞击会导致 crash 或失败终止，任务要求“settle”而非撞击。  
  **usable_signals**: `x_velocity`, `y_velocity`，并可与位置误差耦合（如乘上门限）。  
  **risks**: 若全局施加速度惩罚，可能使早期探索过慢，需条件性激活（靠近目标时权重加大）。

- **role_id**: `stability`  
  **purpose**: 鼓励保持小姿态角和低角速度，尤其在接触前后。  
  **why_required**: 翻滚或侧侧歪斜会导致 crash 或单脚接触失败。  
  **usable_signals**: `body_angle`, `angular_velocity`  
  **risks**: 过度抑制角速度可能使姿态调整困难，需在任务初期适当放松。

- **role_id**: `dual_contact_incentive`  
  **purpose**: 奖励双支撑脚同时接触的状态。  
  **why_required**: 双足接触是实现稳定停靠的必要条件，也是任务“make safe contact”的体现。  
  **usable_signals**: `left_support_contact`, `right_support_contact`  
  **risks**: 单一接触可能不会被计入，但需防止智能体在悬空时勉强点地但不稳定就终止；配合稳定性角色使用。

- **role_id**: `fuel_efficiency`  
  **purpose**: 惩罚任何非零油门动作（no_engine 除外），以降低总推力使用。  
  **why_required**: 任务明确要求“as little engine thrust as possible”。但该目标次要于到达与稳定，应作为常驻小惩罚，避免主目标达成前过度削弱探索。  
  **usable_signals**: `action != 0`  
  **risks**: 过大的燃料惩罚会导致智能体不敢使用引擎，无法抵达目标。

### 10.2 条件职责 conditional_roles
- **role_id**: `landing_zone_soft_constraint`  
  **condition_to_use**: 当 `|x_position|` 和 `y_position` 均进入较小阈值后，进一步加强速度、角度约束，促使在最后阶段精细减速。  
  **usable_signals**: `x_position`, `y_position`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`  
  **risks**: 阈值选择不当可能导致过早收敛，或在目标附近无法到达正确位置。

- **role_id**: `settlement_bonus`  
  **condition_to_use**: 在两个支撑脚同时接触，且速度/角速度低于门限时，给予一次性或持续奖励，引导智能体主动降低能量直至终止。  
  **usable_signals**: `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity`, `angular_velocity`  
  **risks**: 若门限过严，智能体可能无法获得奖励；若过宽，鼓励假稳定。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `time_penalty`  
  **reason**: 没有时间剩余或步数信号，直接硬编码每步惩罚（如 -0.01）会与任何正负奖励冲突，且无法确保为尽快到达服务，因为负奖励的累积会使得智能体尽可能早地终止（可能是 crash 等坏终止），可能鼓励快速失败。因此禁止任何时间惩罚形态。  
  **forbidden_or_missing_signals**: 缺少步数/剩余时间指标，也不应使用 `training_progress` 作为时间信号，因为 prompt 未授权其视为可靠时间度量。

- **role_id**: `survival_bonus`  
  **reason**: 此环境不适合存活奖励，任务目标是尽快结束（settle），鼓励延长存活会与快速到达冲突。

---

## 11. role_to_signal_m



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
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

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
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

