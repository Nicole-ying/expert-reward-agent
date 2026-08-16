# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 车辆/着陆器轨迹优化问题。主体从顶部中心附近开始，带有初始随机力。主要目标是**到达并稳定停靠在中央目标垫上**（即位置归零、速度接近零、姿态稳定且双腿安全接触）。次要目标是**尽可能快**地完成该过程，同时**尽量少用引擎推力**（节约燃料）。  
不要将该任务与纯粹的燃料最小化问题混淆，因为到达目标仍是首要约束。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 任务核心是让主体到达固定目标区域并稳定下来，终态条件基于位置、速度、接触状态，属于典型的导航到达任务；快速和低燃耗只是性能品味，不构成单独冲突目标。

动力学子类型 dynamics_subtype: **goal_approach_and_soft_contact**  
理由：主体必须在接近目标时主动减速，调整姿态，并用两条支撑腿安全接触目标垫，符合“接近 → 减速 → 稳定接触”的动力学模式。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32 (默认)  
- **obs[0]**: `x_position` — 相对于目标垫的水平坐标，reward_usable: true  
- **obs[1]**: `y_position` — 相对于目标垫高度的高度坐标，reward_usable: true  
- **obs[2]**: `x_velocity` — 水平线速度，reward_usable: true  
- **obs[3]**: `y_velocity` — 垂直线速度，reward_usable: true  
- **obs[4]**: `body_angle` — 机体倾斜角，reward_usable: true  
- **obs[5]**: `angular_velocity` — 角速度，reward_usable: true  
- **obs[6]**: `left_support_contact` — 左支撑腿接触标志（0/1），reward_usable: true  
- **obs[7]**: `right_support_contact` — 右支撑腿接触标志（0/1），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- **action 0**: `no_engine` — 不点火，零推力  
- **action 1**: `left_orientation_engine` — 点燃左侧姿态引擎  
- **action 2**: `main_engine` — 点燃主引擎（通常向上推力）  
- **action 3**: `right_orientation_engine` — 点燃右侧姿态引擎  

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 当主体稳定在目标垫上（位置≈0，速度≈0，双腿接触）时触发，极大概率为成功。
- **failure-like termination**:  
  - `crash_or_body_contact` — 任何意外的身体碰撞（如翻倒、腿以外的部分触地）导致失败。  
  - `horizontal_position_outside_viewport` — 水平位置超出视口，丢失目标区域。
- **ambiguous termination**: `body_not_awake_or_settled` 也可能出现在失败坠落并静止于非法区域（如地面远处），但其纹理通常包含速度近零、躺倒状态；但按环境定义，该条件大概率用作成功稳定着陆。
- **truncation**: 代码中无显式步数截断（`terminated` 是组合布尔，无 `Truncated` 返回），因此无自动截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: 无（info 为空字典 `{}`）  
- forbidden_or_uncertain_info_fields: 任何 info 字段（包括 `success`、`failure`、`termination_reason` 等）均不存在，**禁止使用**。

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs` — 当前步观察（8维数组）
- `action` — 执行的动作（0~3）
- `next_obs` — 下一步观察
- `info` — 只允许读取已声明的字段，当前为空，因此**禁止依赖 info**
- `training_progress` — 仅在 prompt 明确允许时使用，**当前不可用**

禁止使用：
- `original_reward` — 官方奖励被掩盖，禁止直接或间接使用
- `official_reward` 变体
- 任何未声明的 obs 切片（如非公开的全局状态）
- 任何未经声明的 info 字段

## 7. 可用于奖励函数的信号
- position:  
  - `x_position`（obs[0]）, `y_position`（obs[1]），均为相对于目标垫的坐标，可用于计算距离、接近程度、高度惩罚等。  
- velocity:  
  - `x_velocity`（obs[2]）, `y_velocity`（obs[3]），可用于速度阻尼、冲击惩罚。  
- orientation:  
  - `body_angle`（obs[4]）, `angular_velocity`（obs[5]），可用于姿态稳定惩罚。  
- contact:  
  - `left_support_contact`（obs[6]）, `right_support_contact`（obs[7]），0/1值，可用于鼓励或惩罚接触状态。  
- action/engine:  
  - `action` 本身，可用于燃料消耗惩罚（非零动作视为引擎使用）。  
- other: 无。

## 8. 不确定或不可用的信号
- 绝对位置或速度（仅有相对目标值）  
- 时间步数、剩余步数（没有在 obs 或 info 中暴露）  
- 燃料剩余量或具体推力值  
- 任何成功/失败标签  
- 外部干扰（如风）的显式测量

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 着陆器（刚体）
  actuator_type: 三个离散推力器：主引擎、左姿态引擎、右姿态引擎
  contact_structure: 两条支撑腿，通过左/右接触标志感知接触
primary_objectives:
  - 将位置 (x,y) 移动到目标垫原点附近（距离趋于0）
  - 在目标垫上以极低速度稳定着陆（vx≈0, vy≈0）
  - 完成着陆时双腿均接触目标垫，姿态接近竖直（angle≈0）且角速度≈0
secondary_objectives:
  - 最小化引擎动作次数（燃料效率）
  - 尽快终止（隐性目标，但无可直接奖励的时间信号）
main_failure_risks:
  - 机体翻倒或非腿部接触（crash）
  - 水平偏离目标垫太远，飞出视口
  - 高速撞击目标垫（穿模/弹起未稳）
  - 在目标垫附近无限期悬停（浪费燃料但未终止）
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  purpose: 将主体拉向目标位置（x→0, y→0）。  
  why_required: 这是到达目标的核心驱动力，否则主体无动力向目标移动。  
  usable_signals: [x_position, y_position]  
  risks: 若权重过高可能驱使高速撞击；需与速度阻尼配合。

- **role_id**: `velocity_damping`  
  purpose: 在接近目标时抑制过高的线速度，防止撞击或弹起。  
  why_required: 仅目标接近不足以诱导稳定着陆，需要减速。  
  usable_signals: [x_velocity, y_velocity]  
  risks: 过强阻尼会使主体停滞在高空，阻碍探索；需与位置奖励平衡或仅在距离较近时激活。

- **role_id**: `orientation_stabilization`  
  purpose: 惩罚大倾角和高角速度，鼓励保持竖直姿态。  
  why_required: 最终着陆需要双腿平稳接触，姿态偏差会导致翻滚或单腿着地失败。  
  usable_signals: [body_angle, angular_velocity]  
  risks: 早期阶段过度惩罚角运动可能抑制必要的姿态调整。

- **role_id**: `soft_contact_encouragement`  
  purpose: 鼓励双腿同时接触目标垫，且接触时速度足够小以避免撞击惩罚。  
  why_required: 最终终止条件要求稳定的双腿接触，仅靠位置、速度奖励可能双腿虚接或单腿着陆。  
  usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity]  
  risks: 直接奖励接触可能导致主体匆忙“摔”向垫子；需结合低速条件。

### 10.2 条件职责 conditional_roles
- **role_id**: `fuel_efficiency_penalty`  
  condition_to_use: 当主体已较为接近目标垫（例如距离 < 阈值）时激活，以避免在远程探索时过多惩罚必要的推力。  
  usable_signals: [action]  
  risks: 若全程惩罚非零动作，会严重阻碍探索，使主体干脆不使用引擎而飘离。

- **role_id**: `final_settlement_bonus`  
  condition_to_use: 仅在满足全部稳定条件（极低速度、双腿接触、姿态小、位置小）时赋予一次性奖励，否则不给予。  
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]  
  risks: 单独使用不会产生平滑梯度，仅作为终端增强；需配合稠密奖励使用。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `time_efficiency_reward`  
  reason: 缺少步数计数或剩余时间信号，无法直接奖励快速完成。  
  forbidden_or_missing_signals: [current_step, remaining_steps]

- **role_id**: `external_collision_penalty`  
  reason: 碰撞事件由环境直接终止轨迹，无法在终止帧计算奖励（终止时不会调用奖励函数或需特殊处理），且 info 无碰撞标记。  
  forbidden_or_missing_signals: [collision_flag]

- **role_id**: `official_reward_imitation`  
  reason: 官房奖励被屏蔽，不得重建或模仿。

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | — | negative_distance, quadratic_penalty, bounded_potential | 推荐用距离的负值或反比，避免因二次项导致远处梯度过大 |
| velocity_damping | x_velocity, y_velocity | — | velocity_magnitude_penalty, squared_speed, soft_clip | 可以与距离门控（如仅当距离<阈值时应用）结合 |
| orientation_stabilization | body_angle, angular_velocity | — | absolute_angle_penalty, squared_penalty | 角度和角速度可分离加权 |
| soft_contact_encouragement | left_support_contact, right_support_contact, x_velocity, y_velocity | — | contact_product * speed_penalty_inverse, conditional_bonus | 鼓励双腿接触且速度小；可设计为近距离下的激励 |
| fuel_efficiency_penalty | action | engine_force_info (不可用) | action_nonzero_indicator, step_penalty | 仅在近距离激活，避免阻碍初期探索 |
| final_settlement_bonus | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | — | all_condition_satisfied_bonus, sparse_terminal_reward | 一次性奖励，置放于所有指标小于阈值时 |

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主体从不接近目标，飘出视口 | 奖励曲线不上升，位置偏差持续很大 | 检查目标 proximity 奖励权重是否过小，或速度阻尼是否过早限制推力 |
| 高速撞击目标垫并弹出/翻倒 | 接触前速度很大，轨迹终止于 crash | 增强临近垫时的速度惩罚，或引入与距离成反比的“下降速度限制”奖励 |
| 长时间在高空悬停，不下降 | y 位置保持较大正值，速度接近零 | 增加高度惩罚或渐进式高度减小奖励，或减少动作惩罚以鼓励向下推力 |
| 单腿着陆无法终止 | 只有一条腿接触，主体微小摇摆但无法稳定 | 加强 soft_contact 奖励对双腿同时接触的激励，同时微调姿态稳定权重 |
| 过度使用姿态引擎导致颠簸 | 动作频繁为 1 或 3，角速度震荡 | 在靠近垫子时激活 fuel_efficiency_penalty，减轻姿态震荡 |



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





# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -107.730

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty | 4 | -107.730 | -107.730 | unsolved |
| attitude + fuel + landing + progress | 1 | -111.910 | -111.910 | unsolved |
| attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + settling_stability | 1 | -121.510 | -121.510 | unsolved |
| attitude + distance + fuel + landing + velocity | 2 | -471.600 | -653.720 | unsolved |

## Previous interventions

- iter 2 (score=-110.570, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): selected_level: L3: 得分始终接近负值，远低于目标 200，且主信号框架将距离改进错误地导向跌落，符合“score 始终接近零或负→重写整个 reward，换完全不同的主信号框架”的规则。第一轮即发现根本错误，必须重构。 | selected_intervention: 完全重写奖励函数，采用经典着陆器势能差框架：
- iter 3 (score=-110.090, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): selected_level**：**L2: 理由：仅调系数（L1）无法解决“远距离速度被错误惩罚”的结构性问题，必须改变 `speed_penalty` 的数学形态，使其依赖距离。 | selected_intervention**：
- iter 4 (score=-107.730, structure=attitude_constraint + fuel_efficiency + goal_proximity + landing_contact + speed_penalty): `selected_level`：**L2: 理由：增减系数（L1）无法解决着陆奖励稀疏的本质问题；必须修改 landing_contact 组件的数学形态，将其从离散终态事件变为连续函数。 | `selected_intervention`：
- iter 7 (score=-471.600, structure=attitude + distance + fuel + landing + velocity): selected_level**：L3（主信号框架重写） | selected_intervention: 将 progress 替换为 **距离奖励**（针对当前状态的函数，而非增量），例如 `w_dist / (1 + dist_next)`，保证越近越好但不会奖励快速坠落。
- iter 8 (score=-653.720, structure=attitude + distance + fuel + landing + velocity): selected_level**：**L2: 理由：单纯调系数（L1）无法改变 landing 死信号和速度惩罚的全局压制；需要修改数学形态以引入“连续接近奖励”和“阶段门控”，且增加势能差形式避免驻留奖励。 | selected_intervention: 1. **距离奖励**：由状态值改为**势能差**（`potential(next)-potential(now)`），迫近给正、远离给负，消除驻留奖励。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
