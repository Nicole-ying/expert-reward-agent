# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器着陆/停靠任务。飞行器起始于视野上方中央区域，受到随机初速度扰动。主要目标是尽快抵达中央的目标垫（target pad）并稳定停靠（settle），以最小的引擎推力消耗完成。飞行器需要学会靠近目标、降低速度、保持姿态平稳，并实现安全的双足（左右支撑）接触，最终停在目标垫上。避免与目标垫以外的任何部位发生碰撞、飞出视口边界或长时间不稳定晃动。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 
- 核心目标是到达中央目标垫并稳定停靠，属于典型的到达目标位置（goal reaching）任务。
- 附属目标（省燃料、快速、姿态平稳、安全接触）服务于主目标的优化，但不构成独立且权重相当的多目标，因此不选 multi_objective_task。
- 任务不涉及持续前进、生存平衡、稀疏探索或驾驶安全约束。

动力学子类型 dynamics_subtype: goal_approach_and_soft_contact  
解释：飞行器需要接近目标，降低速度并实现稳定低速接触，与“接近目标并低速、稳定接触/停靠”的特征完全匹配。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，因为 Box 默认 float32）
- obs[0]: x_position（相对于目标垫的水平坐标），含义：飞行器在 x 方向上偏离目标垫中心的距离，可用于奖励接近目标；reward_usable: true
- obs[1]: y_position（相对于目标垫高度的垂直坐标），含义：飞行器高度与目标垫高度之差，可用于奖励下降/靠近垫面；reward_usable: true
- obs[2]: x_velocity（水平线速度），含义：横向移动速度，可用于惩罚过大侧向速度或奖励静止；reward_usable: true
- obs[3]: y_velocity（垂直线速度），含义：竖直方向速度，可用于惩罚硬着陆（大负值）或奖励稳定；reward_usable: true
- obs[4]: body_angle（机体角），含义：飞行器倾斜角度，可用于奖励保持平正姿态；reward_usable: true
- obs[5]: angular_velocity（角速度），含义：机体旋转速率，可用于奖赏姿态稳定；reward_usable: true
- obs[6]: left_support_contact（左支撑腿接触标志，1.0 或 0.0），含义：左腿是否与目标垫良好接触，可用于奖励双足着陆；reward_usable: true
- obs[7]: right_support_contact（右支撑腿接触标志，1.0 或 0.0），含义：右腿是否与目标垫良好接触，可用于奖励双足着陆；reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（不操作）—— 不做任何推力/姿态调整，依赖当前动量。
- action 1: left_orientation_engine（左姿态引擎）—— 点火一个姿态引擎，产生逆时针/顺时针力矩以调整机体角度。
- action 2: main_engine（主引擎）—— 点火主引擎，产生主要推力（可能方向固定，对 body 坐标系）。
- action 3: right_orientation_engine（右姿态引擎）—— 点火相反的另一个姿态引擎。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 机体不再活跃或已稳定停靠。这可能意味着飞行器已处于静止状态且（通常）两个支撑腿接触目标垫。高风险：如果发生于 crash 后也可能导致不再活跃，因此该终止条件不能单方面被认定为成功。需要结合接触标志、位置、速度等信号综合判断。
- failure-like termination: 
  - crash_or_body_contact —— 身体某部位（非支撑腿）发生碰撞，可能是撞击地面或目标垫以外的区域。
  - horizontal_position_outside_viewport —— 水平方向飞出视口边界。
- ambiguous termination: 无其他。
- truncation: 无显式时间截断（step 源码未显示 max_steps，默认无截断，由 gym wrapper 决定，但环境无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （empty dict，无可用字段）
- forbidden_or_uncertain_info_fields: 所有未在源中列出的信息字段（none）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 当前观测 (8,)
- action: 当前动作 (int 0-3)
- next_obs: 下一时刻观测 (8,)
- info: 空字典（无有效奖励信号）
- training_progress: 仅当 prompt 明确允许时可使用（当前环境无特殊说明，建议不使用）

禁止使用：
- original_reward（已遮蔽，不得参考）
- 任何未声明的 info 字段
- 任何未声明的观测切片（例如私有内部状态）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1]) 以及 next_obs 对应值。
- velocity: x_velocity (obs[2]), y_velocity (obs[3])。
- orientation: body_angle (obs[4]), angular_velocity (obs[5])。
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])。
- action/engine: 当前动作可以用于奖励/惩罚引擎使用（鼓励使用 no_engine）。
- other: 差分信号，如位置变化、速度变化、角度变化，均可从 obs 和 next_obs 构建。

## 8. 不确定或不可用的信号
- 成功标志：无 info['success']、info['failure']、info['termination_reason']。
- 目标垫具体位置：只能通过相对 x,y 推断目标在 (0,0)（或 (0, pad_height)），但 pad_height 未知，不能直接使用绝对高度目标。
- 燃油剩余 / 推力大小：无法获取。
- 外部风力：未知。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_2d
  actuator_type: discrete_thruster_and_orientation_engines
  contact_structure: two_leg_support (left & right contacts)
primary_objectives:
  - minimize_dist_to_target (x,y)
  - achieve_stable_settlement (low velocity, low angular velocity, both contacts = 1)
  - land_safely (avoid crash contact, stay inside viewport)
secondary_objectives:
  - minimize_fuel_usage (favor no_engine action)
  - minimize_landing_time (indirect, episode length)
  - keep_body_upright (minimize |angle|)
main_failure_risks:
  - crashing body part other than legs
  - drifting out of horizontal bounds
  - oscillation or never settling
  - high speed impact leading to bounce or crash
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity_shaping
  purpose: 驱动飞行器靠近目标垫中心（x=0, y→pad_height）。通常用距离或负距离开方作为稠密信号。
  why_required: 没有稠密引导，离散动作探索极难到达目标。
  usable_signals: [obs[0], obs[1], next_obs[0], next_obs[1]] 
  risks: 过度惩罚高度可能让 agent 害怕下降；需要用正值鼓励接近而非纯负值。

- role_id: soft_landing_condition
  purpose: 在接近目标垫且双足接触、速度极低、角度接近零时给予显著正奖励，形成成功着陆信号。
  why_required: 作为稀疏成功事件的替代，因为无显式 success flag，必须根据状态判断。
  usable_signals: [obs[6], obs[7], next_obs[2], next_obs[3], next_obs[5], next_obs[4]]
  risks: 条件阈值设置不当会导致过早奖励未稳定着陆，或永远得不到奖励。

- role_id: energy_efficiency
  purpose: 惩罚使用主引擎或姿态引擎，鼓励 idle (action 0)。
  why_required: 任务明确要求“using as little engine thrust as possible”。
  usable_signals: [action]
  risks: 权重不能过高，否则 agent 宁可悬停或飞越也不开引擎，无法完成着陆。

### 10.2 条件职责 conditional_roles
- role_id: upright_orientation_bonus
  purpose: 当接近地面或低速时，奖励保持 body_angle 接近 0，促进平稳着陆。
  condition_to_use: 仅在靠近目标垫且速度较低时开启，避免早期过度约束姿态。
  usable_signals: [obs[4], obs[5], next_obs[4], next_obs[5]]
  risks: 全局施加可能抑制必要的姿态调整动作。

- role_id: terminal_velocity_penalty
  purpose: 在终局阶段（高度很低且横向偏移小时）惩罚过大的垂直速度，防止硬着陆。
  condition_to_use: y_position 接近零（或低高度）且 |x_position| 较小时，对 y_velocity 负大值施加惩罚。
  usable_signals: [obs[1], obs[3], next_obs[1], next_obs[3]]
  risks: 可能让 agent 不敢下降，需配合 goal_proximity_shaping。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_reward（基于 info 的二元奖励）
  reason: 无可用 info 字段，禁止使用。
  forbidden_or_missing_signals: info['success'] / info['failure']

- role_id: final_goal_sparse_reward（稀疏成功奖励）
  reason: 无显式终止成功标志，只能通过状态推断。强行用 terminated & condition 判断可能引入噪声（比如 body_not_awake_or_settled 也可能是 crash 后的静止状态），风险极高，不推荐。

- role_id: progress_penalty（按时间步惩罚存活）
  reason: 环境无时间截断，使用步数惩罚可能迫使 agent 过急导致 crash；但可作为次要，不建议作为主职责，因为可能导致不安全行为。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_shaping | x_position, y_position (obs & next) | 无 | dense_state_signal, negative_l2_norm, bounded_signal | 可使用指数衰减或倒置高斯型奖励，离目标越近奖励越高 |
| soft_landing_condition | left_support_contact, right_support_contact, x_velocity, y_velocity, angular_velocity, body_angle | 无 | indicator_product, threshold_bonus, bounded_signal | 多个条件同时满足时给予较大正奖励 |
| energy_efficiency | action | 无 | discrete_action_penalty, zero_action_bonus | 非零动作施加小负数惩罚，或零动作给正数奖励 |
| upright_orientation_bonus | body_angle, angular_velocity | 无 | bounded_signal, quadratic_penalty | |cos(angle) -1| 或角速度平方惩罚 |
| terminal_velocity_penalty | y_position, y_velocity | 无 | threshold_quadratic_penalty, bounded_signal | 当 y_position 小于阈值且 y_velocity 过负时激活 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 飞行器学会悬停在目标上方但不下降 | y_position 稳定大于某值，接触永不触发，episode 未终止 | 增加目标高度上的奖励梯度，或给接近地面时额外奖励 |
| 快速坠落 crash 或硬着陆 | y_velocity 极大负值，左/右接触全为 0，或 crash 终止多 | 强化终端速度惩罚，降低主引擎惩罚，或提供速度阻尼引导 |
| 左右摇晃无法稳定 | body_angle 摆动幅度大，角速度频繁变化，从未同时获得两个接触 | 增加姿态稳定奖励（仅在低速时），或引入连续 no_engine 奖励链 |
| 省油导致过早关闭引擎，飘出视口 | x_position 绝对值增大，终止多因 outside viewport | 增加 x 方向接近目标奖励权重，边缘软边界惩罚 |
| 只会使用一个姿态引擎，无法调整双向 | 动作分布严重偏向 1 或 3，另一方向很少使用 | 检查姿态奖励是否对称，适当增加未使用动作的探索噪声 |



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

