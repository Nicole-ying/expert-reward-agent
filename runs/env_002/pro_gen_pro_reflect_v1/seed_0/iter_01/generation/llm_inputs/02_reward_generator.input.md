# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个平面双足行走控制问题。双足身体必须在不平坦地形上尽可能远、尽可能快地向前移动，同时希望尽量减少关节能耗。核心目标是生成稳定前进的步态并安全走完全程；摔倒会立即终止回合，视为失败。到达地形尽头也终止回合，可理解为成功完成。能耗最小化是附属目标，不应干扰前进和平衡主任务。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是让双足身体持续通过地形前进，没有指定的固定目标点，只有方向性的前进进度，附属有速度和能耗优化。这完全符合 locomotion_continuous_control 的定义——让身体沿期望方向行进，并通过地形完成尽可能多的位移。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 （推断）
- obs[0]: hull_angle，主躯干相对直立方向的倾角，越过大可能导致摔倒，可用于奖励保持直立。
  reward_usable: true
- obs[1]: hull_angular_velocity，躯干角速度，可用于抑制剧烈摇晃。
  reward_usable: true
- obs[2]: horizontal_velocity，前向/后向线速度（前进为正），核心前进进度信号。
  reward_usable: true
- obs[3]: vertical_velocity，垂直速度，可能反映弹跳或不平整接触，可用来惩罚不必要弹跳。
  reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度。
  reward_usable: true（可结合周期性步态约束，但非必需）
- obs[5]: hip1_speed，腿1髋关节角速度。
  reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度。
  reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度。
  reward_usable: true
- obs[8]: leg1_contact，腿1是否触地（1.0=接触，0.0=未接触），可用于步态模式分析与奖励。
  reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度。
  reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度。
  reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度。
  reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度。
  reward_usable: true
- obs[13]: leg2_contact，腿2触地标志。
  reward_usable: true
- obs[14]~obs[23]: lidar_0..lidar_9，10个激光雷达距离测量值，描述前方地形高度剖面。
  reward_usable: true（可用于提前准备地形适应，但对前进奖励不易直接使用）

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- dtype: float32
- 范围: 每个动作分量均在 [-1.0, 1.0] 之间
- action[0]: hip_torque_leg1，施加于腿1髋关节的扭矩
- action[1]: knee_torque_leg1，施加于腿1膝关节的扭矩
- action[2]: hip_torque_leg2，施加于腿2髋关节的扭矩
- action[3]: knee_torque_leg2，施加于腿2膝关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain，走到地形末端，视为成功完成全程。
- failure-like termination: body_fallen_over，身体摔倒，任务失败。
- ambiguous termination: 无。
- truncation: 未提及超时，但实际环境中可能发生，环境规范中为 False 截断，prompt 未给出具体限制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典，没有显式成功标记）
- explicit_failure_flag_available: false（info 为空，没有显式失败标记；但可以从终止条件推断，终止标志本身并未传入奖励函数）
- allowed_info_fields: 空字典 {}，无可用的额外信息。
- forbidden_or_uncertain_info_fields: 任何假设的 info 字段如 "success", "failure", "termination_reason" 均未出现，禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（前一步的 24 维观测）
- action（4 维动作扭矩）
- next_obs（执行 action 后的 24 维观测）
- info（目前为空字典，无可使用字段）
- training_progress（仅当 prompt 明确允许时可用，此处未要求，不建议使用）

禁止使用：
- original_reward（官方被遮蔽的奖励，不得参考或重建）
- 任何未在 info 中声明的字段（如 “success”, “failure” 等）
- 任何未在 obs/next_obs 中声明的维度或隐式推导的状态
- 禁止依赖环境内部的地面真值位置、速度等隐藏变量

## 7. 可用于奖励函数的信号
- position: 无绝对位置测量，只有速度。
- velocity: horizontal_velocity (obs[2])、vertical_velocity (obs[3])、hull_angular_velocity (obs[1]) 以及各个关节角速度（obs[5,7,10,12]）。
- orientation: hull_angle (obs[0]) 可作为倾角惩罚。
- contact: leg1_contact (obs[8])、leg2_contact (obs[13])，可用于步态切换或支撑相惩罚。
- action/engine: 动作扭矩 action（4 维向量），可用于能量或动作平滑惩罚。
- other: LIDAR 数组 (obs[14:23]) 提供地形距离，可间接用于不确定性惩罚或鼓励适应。

## 8. 不确定或不可用的信号
- 绝对位置/前进距离：没有 x 坐标，不能直接奖励位移累计。
- 终点成功标志：未知，不能分配终点大奖励。
- 摔倒前的预警信号：可间接通过 hull_angle 和 vertical_velocity 设计惩罚，但没有单独的“即将摔倒”标志。
- 能耗真值：没有功耗测量，只能用 action 扭矩大小（平方和）作为近似。
- 地形难度度量：LIDAR 可反映前方起伏，但无法量化难度，谨慎使用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal_rigid_body_with_two_legs
  actuator_type: torque_controlled_hip_and_knee_motors
  contact_structure: binary_foot_ground_contacts
primary_objectives:
  - maximize_forward_progress (walk as far and fast as possible)
  - avoid_falling (maintain upright posture and stability)
secondary_objectives:
  - minimize_energy_consumption (torque usage)
main_failure_risks:
  - falling_over (triggers early termination)
  - unstable_oscillation_due_to_poor_joint_coordination
  - excessive_energy_cost_with_little_forward_speed
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress_reward
  purpose: 鼓励 agent 产生正向水平速度，以达成“尽可能远和快”的主目标。
  why_required: 任务核心是前进，没有位置反馈，只能用速度作为代理。这是唯一明确的进度信号。
  usable_signals: [horizontal_velocity (obs[2])]
  risks: 可能导致原地高频抖动骗取速度奖励，需配合其他职责（如平衡惩罚、能耗惩罚）抑制。

- role_id: upright_stability_penalty
  purpose: 惩罚躯体倾角远离直立，预防摔倒，增强步态稳定性。
  why_required: 摔倒立即终止，会切断后续所有机会，必须在每一步维持稳定。
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过重的姿态惩罚会抑制向前的运动幅度，需与前进奖励平衡。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency_penalty
  purpose: 抑制过大的关节扭矩，降低能耗，平滑动作轨迹。
  condition_to_use: 在训练中后期或速度已经达到一定阈值后权重逐渐上升，但早期应保持轻量，以免阻碍探索。
  usable_signals: [action (all 4 dimensions)]
  risks: 过强惩罚会冻结探索，导致步态过小无法前进。

- role_id: contact_alternation_incentive
  purpose: 鼓励周期性支撑相切换，推动形成自然步态。
  condition_to_use: 当 agent 出现明显双腿同时拖地或始终单腿支撑的异常模式时可增加，需谨慎调节。
  usable_signals: [leg1_contact (obs[8]), leg2_contact (obs[13])]
  risks: 机械的接触交替奖励可能与动力学冲突，造成不稳定；可能仅用于诊断性引导，不宜作为强奖励。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_success_bonus
  reason: 缺乏成功标志和位置信息，无法可靠判断“到达终点”，且不允许使用被遮蔽的 original_reward。因此不能给予终点大奖励。
  forbidden_or_missing_signals: [reached_end_of_terrain flag not exposed]

- role_id: terrain_aware_preplanning_reward
  reason: LIDAR 信号可使用，但如何将其映射为奖励十分间接，易破坏训练稳定性，且不在核心目标中。不建议作为奖励项，最多作为探索性引导以观察效果。
  forbidden_or_missing_signals: 无明确地形通过难度或期望路径。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress_reward | horizontal_velocity (obs[2]) | - | dense_state_signal, bounded_signal (clip negative or zero) | reward signal: forward speed; clip negative part to zero to avoid punishing backward motion |
| upright_stability_penalty | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | - | quadratic_penalty (on angle), absolute_penalty (on ang vel) | penalty = w1 * angle² + w2 * |ang vel| |
| energy_efficiency_penalty | action[0],action[1],action[2],action[3] | - | quadratic_penalty (sum of squares) | cost = torque_sum_sq; weight should be small early |
| contact_alternation_incentive | leg1_contact (obs[8]), leg2_contact (obs[13]) | - | difference_from_constant or alternating index (e.g., (c1-c2)²) | could encourage symmetry; use with low weight |
| terminal_success_bonus | - | explicit “reached end” flag, absolute position | N/A | 禁用 |
| terrain_aware_preplanning_reward | lidar (obs[14:23]) | terrain difficulty map | N/A | 不建议直接奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 无法产生有效前进（原地踏步或抖动） | horizontal_velocity 在正负交替，平均值接近 0；能耗偏高 | 增加 forward_progress_reward 权重，或引入速度平滑项（惩罚速度剧烈变化） |
| 过早频繁摔倒 | 回合长度短，hull_angle 多次突破阈值 | 加大 upright_stability_penalty，或使用 hull_angle 阈值敏感二次惩罚；也可在信息中检查垂直速度异常 |
| 步态凝固或单腿拖动 | 某一条腿 contact 长时间为 1 或 0 | 引入 contact alternation incentive，或检查关节角度范围是否过小 |
| 能耗极大但速度低 | 动作平方和很大，而 horizontal_velocity 很小 |



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

