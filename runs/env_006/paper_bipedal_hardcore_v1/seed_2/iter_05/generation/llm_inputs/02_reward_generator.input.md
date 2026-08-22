# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个双足机器人依靠独立髋、膝关节力矩在不规则地形（包含梯子、树桩、坑洞等）上前进，尽可能走得更远且高效。机器人配备 10 个前方地面 LiDAR 距离传感器，可预判障碍物。主要目标是稳定行走并抵达地形终点，避免摔倒；次要目标是尽量减少不必要的关节力矩消耗。奖励设计需要鼓励向前移动、维持身体平衡、控制能耗，但不能直接使用任何官方奖励，且无额外显式的成功或失败信息字段。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心任务是让双足机器人持续前进穿越崎岖地形，主目标是“前进”而非到达指定目标点，没有抓取或操控物体，也没有强安全约束或生存导向的存活平衡。附属的节能、平稳属于典型连续控制任务的次要目标，不构成多目标权重冲突。因此属于 locomotion_continuous_control 族。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32
- obs[0]: hull_angle, 身体倾角，reward_usable: true（可用于摔倒检测）
- obs[1]: hull_angular_velocity, 身体角速度，reward_usable: true（辅助平衡惩罚）
- obs[2]: horizontal_speed, 质心水平速度，reward_usable: true（前进奖励核心信号）
- obs[3]: vertical_speed, 质心垂直速度，reward_usable: true（用于检测坠落或异常）
- obs[4]: joint_0_angle (hip_1), reward_usable: true（关节角度，可辅助步态）
- obs[5]: joint_0_speed (hip_1 角速度), reward_usable: true
- obs[6]: joint_1_angle (knee_1), reward_usable: true
- obs[7]: joint_1_speed (knee_1 角速度), reward_usable: true
- obs[8]: joint_2_angle (hip_2), reward_usable: true
- obs[9]: joint_2_speed (hip_2 角速度), reward_usable: true
- obs[10]: joint_3_angle (knee_2), reward_usable: true
- obs[11]: joint_3_speed (knee_2 角速度), reward_usable: true
- obs[12]: leg_1_ground_contact (二值 0/1)，reward_usable: true（步态分析，摔倒检测）
- obs[13]: leg_2_ground_contact (二值 0/1)，reward_usable: true
- obs[14]~obs[23]: lidar_1~lidar_10, LiDAR 距离读数，reward_usable: true（可用于地形预判，但非直接奖励信号，主要用于策略内部）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- action[0]: hip_1_torque, 第一髋关节力矩，连续值
- action[1]: knee_1_torque, 第一膝关节力矩，连续值
- action[2]: hip_2_torque, 第二髋关节力矩，连续值
- action[3]: knee_2_torque, 第二膝关节力矩，连续值

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形终点，通常视为成功，但无显式标志）
- failure-like termination: body_fallen_over（身体倾倒，通常视为失败，也无显式标志）
- ambiguous termination: 无
- truncation: 未说明 episod 有最大步数截断，从描述看假设仅由上述条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典）
- forbidden_or_uncertain_info_fields: 所有自定义字段均不可用
- 推断方法：终止时，可通过下一状态或终止时刻的观测信号推断类型：
  - body_fallen_over 可间接从 hull_angle 的绝对值突然超过某个阈值（例如 > 1.0 rad）、vertical_speed 的剧烈负值或 leg_ground_contact 变为零持续多步推导得到，标记为 derived_possible。
  - reached_end_of_terrain 可间接从 agent 持续前进了较远距离后 episode 终止，但 hull_angle、vertical_speed、leg_contact 等均保持正常（无跌倒迹象）推导得到，同样标记为 derived_possible。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：所有 24 维观测
- action：4 维动作（力矩）
- next_obs：下一时刻的 24 维观测
- info：空字典（无可用字段）
- training_progress：当前 prompt 没有明确允许，不使用

禁止使用：
- original_reward（已被掩盖）
- 任何未在 info 中声明的字段
- 直接读取终止原因（因为无 info 字段提供）

## 7. 可用于奖励函数的信号
- position/distance: 无绝对 x 坐标，只能通过水平速度积分相对前进距离，可使用 horizontal_speed 作为主信号。
- velocity: horizontal_speed (obs[2]), vertical_speed (obs[3]), hull_angular_velocity (obs[1])
- orientation: hull_angle (obs[0])，用于检测倾倒及姿态惩罚
- contact: leg_1_ground_contact (obs[12]), leg_2_ground_contact (obs[13])，可用于步态、摔倒推断
- action/engine: 动作扭矩 action[0:4]，用于能耗惩罚
- other: LiDAR 读数 (obs[14:24])，环境提供，但不建议直接用于奖励，因为它是感知信息而非任务指标；不过可用于奖励机制如“异常地形导致调整过大”的辅助惩罚，必要性低。
- derived_possible: 通过信号组合推断的 success（到达终点）和 failure（摔倒）事件，用于构建 sparse 成功/失败奖励，但无法直接读取标志，需在 compute_reward 中使用基于阈值的条件判断（例如在 terminated 状态下检查 next_obs 的 hull_angle 是否过大，或 vertical_speed 是否异常等）。此类推断有误判风险，但可在 reward shaping 中利用。

## 8. 不确定或不可用的信号
- explicit success/failure flags: 无
- 绝对位置、终点距离：无
- 地形类型、障碍物标签：无
- 能耗真实测量：仅能通过动作平方和近似
- 稳定站立判定：leaf 复杂接触不存在

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: bipedal_rough_terrain_locomotion
control_type: continuous
morphology:
  body_type: bipedal with rigid hull
  actuator_type: torque-controlled hip and knee joints (2 legs, 2 DOF each)
  contact_structure: two feet, binary ground contact sensors
primary_objectives:
  - Maximize forward distance traveled (survive and progress)
  - Avoid falling over (maintain balance)
secondary_objectives:
  - Minimize joint torque usage (energy efficiency)
  - Smooth gait (avoid excessive joint accelerations)
main_failure_risks:
  - Falling due to rough terrain or improper gait adaptation
  - Overreaction to LiDAR inputs causing instability
  - Stuck in local optima of inefficient but stable walking
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励机器人持续向前移动，最大化行进距离
  why_required: 任务是前进型 locomotion，没有前进奖励策略不会学习行走
  usable_signals: [horizontal_speed (obs[2])]
  risks: [可能鼓励高速但不稳定的步态，需配合平衡惩罚]

- role_id: fall_prevention
  purpose: 检测并惩罚身体倾倒，终止时给予较大负奖励
  why_required: 摔倒终止是主要失败模式，必须被惩罚
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1]), vertical_speed (obs[3]), leg_ground_contacts (obs[12,13])；终止时 next_obs 中可推断摔倒事件]
  risks: [基于阈值的伪死判定可能误判暂时的倾斜为摔倒；需要合理的阈值和延迟确认]

### 10.2 条件职责 conditional_roles
- role_id: energy_penalty
  purpose: 惩罚过大的关节力矩，促进高效步态
  condition_to_use: 当任务要求“最小化不必要的关节扭矩”时启用；作为次要目标，权重应低于前进奖励
  usable_signals: [action[0:4] 或连续力矩的平方]
  risks: [过高的惩罚权重可能导致机器人不敢用力，无法行走]

- role_id: smooth_gait_penalty
  purpose: 惩罚关节速度/加速度突变，鼓励流畅步态
  condition_to_use: 若环境需要适应粗糙地形，光滑性可能有利，但非强制
  usable_signals: [joint speeds (obs[5,7,9,11])，或相邻动作差]
  risks: [可能抑制必要的快速调整，削弱对地形的响应能力]

- role_id: survival_bonus
  purpose: 每存活一步给予小额正奖励，缓解稀疏性
  condition_to_use: 当仅使用 terminated 信号而不够密集时
  usable_signals: [不存在显式存活信号，但可通过“未摔倒”判断，即每步只要未终止则视为存活]
  risks: [在终止不可提前判断时难以实现，但此环境 terminates 仅由仿真状态引起，compute_reward 在 call 时已知 terminated 参数，因此可通过外部调用传入 terminated 标志（实际 compute 函数包含 terminated 参数吗？接口原型只给了 obs, action, next_obs, original_reward, info, training_progress，未包含 terminated）。所以 survival_bonus 需要 terminated 参数，若接口不允许则不可直接使用。但通常 reward 函数会在 step 内被调用并可访问 terminated 变量，此处约定未明确禁止，但谨慎处理。可以考虑如果无法获取 terminated，则使用连续正向速度奖励替代存活奖励。]
  risks: [若无 terminated，则只能通过正向速度实现，可视为同一职责]

### 10.3 慎用/禁用职责 avoid_roles
- role_id: lidar_usage_reward
  reason: LiDAR 是感知输入，不应直接作为奖励信号，否则策略可能学会“欺骗”奖励而非真正前进；仅在特殊场景中作为辅助特征使用，但无可靠度量标准。
  forbidden_or_missing_signals: [缺少 LiDAR 读数与地形相关的奖励度量（如通过障碍得分）]

- role_id: target_matching
  reason: 任务无显式目标位置，仅有终点；终点没有观测距离或方向，不适合做引导奖励。若使用 inferred success 给予稀疏奖励，但需要可靠检测。
  forbidden_or_missing_signals: [无终点距离信息]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_speed (obs[2]) | absolute x position | dense_state_signal (linear scaling) | 直接使用水平速度作为即时奖励 |
| fall_prevention | hull_angle (obs[0]), hull_angular_velocity, vertical_speed, leg_contacts; derived termination flag (body_fallen_over inferred) | explicit fall flag | threshold_penalty (角度 > 阈值 大负奖励), termination_moment_penalty | 在 terminated 时基于 next_obs 中的倾角等判定是否摔倒，给予负奖励 |
| energy_penalty | action[0:4] | true torque cost | quadratic_penalty (sum of squares) | 惩罚力矩平方和 |
| smooth_gait_penalty | joint speeds (obs[5,7,9,11]), actions difference | none | bounded_signal (penalize high speed), action_diff_penalty | 可选，当步态过于抖动时启用 |
| survival_bonus (若 terminated 可用) | terminated flag (from env.step) | none | constant_bonus per step | 每步存活给小额奖励，需在 reward 函数中传入 terminated 参数，如不可行则使用 forward_progress 替代 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 机器人原地摇摆，不前进 | horizontal_speed 接近零，动作震荡 | 增大 forward_progress 权重，或加入步态对称性奖励 |
| 快速摔倒（早期终止） | hull_angle 迅速超过阈值，vertical_speed 突变 | 加强 fall_prevention 惩罚权重，引入平衡姿态奖励（如 hull_angle 的绝对惩罚） |
| 缓慢但高能耗行走 | energy_penalty 权重不足，动作方差大 | 逐渐增大 energy_penalty 系数，或者添加动作平滑度惩罚 |
| 到达终点但奖励总和不达预期 | reached_end 推断成功但 forward_progress 与 survival 总和较低 | 为成功终止添加较大的稀疏奖励（当可靠检测成功时） |
| LiDAR 使用不当导致策略不稳定 | 动作随 LiDAR 读数剧烈变化 | 不宜在奖励中直接使用 LiDAR，检查策略利用 LiDAR 的方式，也许引入 smoothness penalty |



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

---

## 1. Expert Schema 使用规则

- environment_card.md 中的任务画像和可用信号优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role（任务需要什么类型的奖励信号），再选 signal（哪个观测维度承载这个 role），再选 formula operator（用什么数学形式表达），最后写代码。
- 如果某个 role 需要的信号在观测空间中不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- reward_v1 以主学习信号和必要的稳定/安全约束为重点。效率、能耗、复杂门控和动态权重可以在后续迭代中按需加入，但不应因"模板没列"而排除合理的设计。

---

## 2. 信号完备性自查清单

在完成初始设计后，逐一检查以下信号类型是否被覆盖——不是每个任务都需要全部，但每一项的缺失应是有意选择：

- **主进展信号**：agent 朝任务目标前进时是否获得正向反馈？该信号是否每步都有梯度？
- **灾难性失败信号**：是否存在明确的终止惩罚（如摔倒、飞出边界）？如果观测中可推断失败状态，是否给予了足够强的负向信号？
- **效率/代价信号**：连续动作空间中是否有能量消耗或控制代价约束？离散动作空间中是否有不必要的动作惩罚？
- **任务完成信号**：终止条件中是否包含 success-like 条件？相应的观测是否可被用来构造任务完成的软近似信号？
- **健康/稳定约束**：agent 是否因缺少姿态/速度/位置约束而产生不安全行为？

---

## 3. Formula Operator Library

每个算子包含：数学形式、使用条件、适用证据。

### 3.1 dense_state_signal
数学形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2`
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
使用条件：该状态信号每步可观测，且与某项任务职责直接相关。
适用证据：
  - 凸化 → episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小（agent 满足于低水平稳态）。
  - hinge → 约束组件的 active_rate≈100%（全时惩罚）但 terminated 率仍高，说明 agent 在安全范围内也被持续惩罚，需要只在越界时生效的 hinge。
风险：线性正奖励在信号平台期无梯度；凸化权重过大可能诱导极端行为；hinge 的 threshold 需根据环境卡片的观测范围设定。

### 3.2 improvement_delta
数学形式：`old_measure - new_measure`（期望减少时）或 `next_value - current_value`（期望增加时）
使用条件：obs 和 next_obs 中存在可比较的标量度量，该度量沿最优路径应单调变化。
适用证据：有明确的进展度量（位置、距离、高度、角度等），且该度量的变化比瞬时速率更能反映真实进展。
与 dense_state_signal 的选择：如果要鼓励"处于某种好状态"，用 `w * signal`。如果要鼓励"朝好方向改变"，用 delta。delta 的优势是 agent 无法在好状态上停滞不前，必须持续改善。适合：agent 当前的绝对状态值不能完全反映进展（如位置——站在原点不动 vs. 走到终点但位置绝对值可能相同）。
注意：对观测中直接给出的速度信号（如 `horizontal_velocity`）不要做 delta——速度本身已经是变化率。对观测中的位置/角度/距离类信号优先考虑 delta。

### 3.3 potential_based_shaping
数学形式：`potential(next_obs) - potential(obs)`
使用条件：(1) 任务有一个可量化的进展度量（如位置、距离、高度）；(2) 该度量沿最优路径应单调变化；(3) 能从观测中构造一个标量的 potential function。
如何构造 potential：从观测中选择一个在任务完成时达到极值、且沿最优路径单调变化的信号（或信号组合）。potential 的计算只能依赖观测，不能依赖环境内部状态。
与 improvement_delta 的关系：两者数学上等价。potential_based_shaping 的优势在于允许将多个信号编码到一个 potential 中（如同时考虑位置和姿态），而 improvement_delta 通常用于单个度量。
风险：potential 若与任务目标不一致会系统性地误导策略。reward_v1 中如果存在天然的进展度量，优先使用 improvement_delta 的简单形式；当需要组合多个信号构造进展度量时，使用 potential_based_shaping。

### 3.4 quadratic_penalty
数学形式：`-w * error**2` 或 `-w * sum(action_i**2)`
使用条件：约束信号连续可观测，惩罚不应压制主学习信号。用于轻量抑制——需要约束但不至于触发终止的行为。
适用证据：某维度出现高频大幅波动或极端值但未触发终止。
与 hinge 的选择：如果约束有明确的安全边界（如身体倾角超过 X 度必摔），用 hinge（3.1）。如果只是希望"越小越好"没有硬边界（如控制代价、小幅抖动），用 quadratic。
风险：权重过大导致 agent 不敢行动。

### 3.5 soft_health_gate
数学形式：`main_reward * gate_factor`，gate_factor ∈ [0, 1] 在身体状态恶化时平滑衰减。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (safe_bound - current) / margin))`
使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
适用证据：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍 >0——agent 在"先冲后死"，需要在健康恶化时切断主奖励而非额外加罚。
风险：gate 太严格抑制探索；衰减区间应设在"接近危险但尚未终止"的范围内。

### 3.6 terminal_event
数学形式：`if failure_condition: reward = -PENALTY`（硬覆盖 per-step 奖励），或 `if success_condition: reward = +BONUS`
使用条件：(1) 存在可从观测推断的灾难性失败状态（如身体倾角超过阈值 + 接触地面）或任务完成状态；(2) 环境 info 为空因此无法直接读取终止原因。
如何构造：不要依赖 info 字段判断终止原因。可从观测推断：摔倒 → hull_angle 突然偏转 + 身体位置急剧下降；到达终点 → 持续前进中 episode 突然终止（truncated）；出界 → 位置坐标超出有效范围。
适用证据：agent 频繁触发某种终止模式，但当前奖励没有针对该模式提供差异化信号——比如所有终止回合 reward 都一样，agent 无法区分成功和失败。
与 hinge/gate 的区别：hinge 在越界前提供连续梯度，gate 在恶化时衰减主信号。terminal_event 在事件发生的那一刻提供硬信号——没有梯度，但语义明确（"这就是你应该避免/追求的结果"）。

### 3.7 action_efficiency
数学形式：`-w * sum(|action_i|)` 或 `-w * sum(action_i**2)`
使用条件：动作空间 ≥ 2 维连续控制，且任务包含隐含的效率需求（如 locomotion、manipulation）。
适用证据：agent 学会完成任务但动作幅度异常大、能耗高——说明缺效率约束。通常系数较小（主信号 per-step 的 1-5%），避免压制探索。
注意：离散动作空间通常不需要此算子，因为离散动作的选择隐含了代价。首次迭代可不加入，后续迭代若观察到无效动作频繁出现再考虑。

### 3.8 joint_condition_proxy
数学形式：`factor_1 * factor_2 * ...`（每个 factor 为连续 bounded 形式）或 `(f1 + f2 + ...) / n` 或 `(f1 * f2 * ...) ** (1/n)`
使用条件：没有显式 success flag，但有连续信号可构造任务完成的软近似。
适用证据：agent 能在各子条件分别取得进展但无法同时满足。
风险：乘积塌缩（一个 factor→0 则整体→0）；用几何平均或算术平均可缓解。

### 3.9 bounded_signal
数学形式：`x / (1 + abs(x))` 或 `1 / (1 + k * abs(error))` 或 `max(0, 1 - abs(error) / threshold)`
使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。用于压缩极端值而非施加约束。
与 hinge 的区别：bounded 是从两端压缩信号范围，hinge 是只在超出阈值时施加惩罚。如果目标是"值不应超过 X"，用 hinge；如果目标是"值不应该爆炸但无所谓具体范围"，用 bounded。

### 3.10 preview_conditioned_reward
数学形式：`main_reward * preview_factor`，preview_factor 基于观测中能反映**未来状态**的信号（如距离传感器、高度采样、前方地形探测），在不利前景下从 1 平滑衰减到下限。
使用条件：(1) 观测中存在提供前方/未来信息的维度；(2) 该维度可以映射到"前景好/坏"的连续度量；(3) agent 的失败模式与"无法提前调整行为以应对即将到来的状态变化"相关。
如何构造：从提供未来信息的观测中选择一个标量信号，设计一个在安全前景下接近 1、危险前景下接近下限（如 0.3-0.5）的衰减函数。下限不为零以避免完全抑制探索。
适用证据：agent 在相似的瞬时状态下表现差异大（同样的速度/姿态，有时成功有时失败），说明当前状态本身不足以区分好坏——缺少关于"接下来会发生什么"的信息。
与 soft_health_gate 的区别：gate 用当前的**身体状态**乘主奖励（"我已经歪了，别冲了"——被动响应）。preview 用**未来信息**乘主奖励（"前面是坑，别冲了"——主动预判）。两者可以共存：`main_reward * health_gate * preview_factor`。
风险：preview 信号若有噪声会导致主奖励波动；衰减下限设太低会抑制必要探索。

---

## 4. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"定位到合适的算子变换。
以数学语义和训练表现证据为准，不要求组件名完全匹配。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |





# Fresh Restart Evidence

- target_score: 300.000
- best_score_so_far: -59.970

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| balance_penalty + forward_progress | 1 | -59.970 | -59.970 | unsolved |
| balance_penalty + forward_reward + terrain_gate + terrain_roughness | 1 | -74.850 | -74.850 | unsolved |
| air_stability_penalty + balance_penalty + forward_progress | 1 | -86.300 | -86.300 | unsolved |
| air_stability_penalty + balance_penalty + forward_reward + terrain_gate + terrain_roughness | 1 | -95.840 | -95.840 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
