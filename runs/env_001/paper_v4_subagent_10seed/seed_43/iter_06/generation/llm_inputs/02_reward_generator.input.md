# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主体是一个二维飞行器，从视口顶部中心附近受随机初始力出发，核心目标是**尽快、平稳地降落在中心目标垫上**。着陆要求在两个支撑腿均接触目标垫的同时，保持低速度、姿态稳定。附属优化目标是**尽量少用引擎推力**，即整个过程中减少不必要的姿态调整和主引擎点火次数，但必须以安全着陆为前提。任务不包括巡航、越障或抓取等无关动作。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求“reach and settle at a central target pad”，即到达指定目标位置并停靠，着陆点是唯一的、静态的。附属目标（快、省燃料）是对轨迹质量的约束，不构成与“到达”同等权重的多目标冲突。控制类型为离散动作，但底层是连续物理，符合导航目标到达 → 接近阶段与软着陆的条件组合。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: **x_position** – 机体相对于目标垫水平坐标（中心为0），reward_usable: true
- obs[1]: **y_position** – 机体相对于垫面高度，reward_usable: true
- obs[2]: **x_velocity** – 水平线速度，reward_usable: true
- obs[3]: **y_velocity** – 垂直线速度，reward_usable: true
- obs[4]: **body_angle** – 机体倾角（弧度），reward_usable: true
- obs[5]: **angular_velocity** – 角速度，reward_usable: true
- obs[6]: **left_support_contact** – 左支撑腿是否接触（0/1），reward_usable: true
- obs[7]: **right_support_contact** – 右支撑腿是否接触（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作0: no_engine – 无引擎输出（惯性漂行）
- 动作1: left_orientation_engine – 点燃左侧姿态引擎（产生角/线加速度，调节姿态）
- 动作2: main_engine – 点燃主引擎（产生主体坐标系推力，通常向上或前方）
- 动作3: right_orientation_engine – 点燃右侧姿态引擎（与左引擎相反方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**: 机体与地面或垫面发生非期望碰撞（可能是高速撞击、侧翻等），视为**失败**。
- **horizontal_position_outside_viewport**: 水平位置超出视口范围，视为**失败**。
- **body_not_awake_or_settled**: 机体进入休眠状态或已稳定停止（包括成功着陆后速度归零），属于 **ambiguous termination**；需结合左右接触标志和相对位置判断是否为成功着陆。
- 本环境不设置显式最大步数截断（truncation），若 episode 自然终止前无上述触发，则由环境内部最大 step 截断，但不通过 info 提供。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 在 step 源码中返回空字典，禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有字段均不存在

**重要**: 成功或失败只能通过 next_obs 间接推断。当 episode 终止时，可以根据 next_obs 的接触标志、位置、速度组合判定是否为成功。例如：左右 support 均接触且|x_position|极小、|y_position|接近0且绝对速度很低，则可认为是成功着陆。此类信号属于 derived_possible，可在奖励中使用但需谨慎组合。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- obs（当前状态，shape(8,)）
- next_obs（下一时刻状态，shape(8,)）
- action（0~3）
- info 中明确允许的字段（当前为无）
- training_progress 仅当 prompt 明确允许时才用

禁止使用：
- original_reward
- official_reward
- 任何未声明的 info 字段（包括但不限于 success, failure, termination_reason 等）
- 未声明的 obs 切片含义

## 7. 可用于奖励函数的信号
- **位置**:
  - `x_position`, `y_position` → 计算与目标（0,0）的欧氏距离 `dist = sqrt(x^2 + y^2)`
  - 距离降低表示向目标靠近。
- **速度**:
  - `x_velocity`, `y_velocity` → 总体速度大小 `speed = sqrt(vx^2 + vy^2)`，可用于鼓励减速。
- **姿态**:
  - `body_angle` → 倾角，理想着陆姿态应为接近0（水平），可用角度绝对值作为 penalty。
  - `angular_velocity` → 角速度，小为宜。
- **接触**:
  - `left_support_contact`, `right_support_contact` → 可推断着陆状态。两个腿同时接触且位置在目标附近、速度低，可视为成功软着陆。
- **动作/引擎**:
  - 动作 `action` (0~3)，可区分是否使用引擎、哪个引擎，用于计算推力惩罚。
- **衍生信号 (derived_possible)**:
  - 成功着陆事件：由 `next_obs` 满足 (left_contact & right_contact) 且 `dist < 阈值` 且 `speed < 阈值` 且 episode 终止（可由环境自动截断或 stable 终止推断，但无法显式获得 terminated flag。在实践中可以通过 `original_reward` 为0或特定值不成依赖，我们只能设计奖励函数在正常步中给予奖励，而非在终止步专门奖励。我们可以利用终止时 next_obs 最后一个画面给予一次性高分，但这要求知道是否终止。由于无法获取 terminated 标志，最好**不在每一步使用“成功事件”奖励**，而是通过密集的接近、减速、姿态奖励来引导，并依靠环境终止条件自然结束，这样更安全。若实在需要，可由训练框架检测 episode 结束时最后一帧 next_obs 并给予额外奖励，但这不在标准 compute_reward 内。）

## 8. 不确定或不可用的信号
- 显式的 success/failure/termination_reason 标志：不可用，info空。
- 是否发生 crash_or_body_contact 的具体类型：不可直接获得。但可通过突然大幅位置变化或异常接触标志推断，但不可靠。
- 视口外判断：只能通过位置绝对值超过某范围（如 x > 1.5 等，需环境边界值）推断，边界可能需从初始状态范围估计。此信号可能衍生但环境未提供确切阈值，作为不确定信号。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (lander-like), single body with two landing supports
  actuator_type: discrete thrusters (main engine + two orientation engines)
  contact_structure: left and right support contacts with ground/pad
primary_objectives:
  - Reach target pad (minimize distance to (0,0))
  - Achieve safe soft contact (both supports touching, low velocity, upright angle)
secondary_objectives:
  - Minimize engine usage (fuel efficiency)
  - Approach quickly (time efficiency)
main_failure_risks:
  - Crashing due to high impact velocity or wrong angle
  - Falling off viewport laterally
  - Oscillating uncontrollably due to overuse of orientation engines
  - Hovering near target without touching down (truncated)
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id: delta_distance_to_target**
  purpose: 鼓励连续步间向目标垫靠近，形成正梯度
  why_required: 核心目标就是到达，距离变化量是最直接的学习信号，且能避免原地悬停得分
  usable_signals: [x_position, y_position] （当前和下一步计算距离之差）
  risks: 如果同时有其他奖励（如速度奖励）可能相互干扰；若接近过快导致撞击，可能需配合安全门控

- **role_id: approach_speed_bonus_with_safety_gate**
  purpose: 在安全距离和低危险速度范围内奖励快速接近，同时通过门控防止撞击
  why_required: 任务要求“as fast as possible”，但不鼓励超速撞击；需要在接近目标时减速，故需要区分阶段
  usable_signals: [x_position, y_position, x_velocity, y_velocity]
  risks: 门控函数选择不当可能导致奖励稀疏或鼓励危险行为，应使用 hinge 型惩罚拦截高速

### 10.2 条件职责 conditional_roles
- **role_id: soft_landing_stability_penalty**
  condition_to_use: 当 agent 处于目标垫附近（dist < 阈值）或已有一腿触地时启用
  usable_signals: [y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 如果在不接近目标时惩罚角速度和垂直速度，可能阻碍正常机动。需根据到目标的距离动态加权。

- **role_id: engine_efficiency_penalty**
  condition_to_use: 贯穿全程，但权重远低于主距离奖励；任务明确要求“as little engine thrust as possible”
  usable_signals: [action]
  risks: 若设定权重过高，可能导致 agent 不使用引擎而无法到达目标；应当使引擎使用惩罚相对主奖励较小，如 -0.01 每步点燃引擎。

- **role_id: terminal_touchdown_bonus (derived)**
  condition_to_use: 若训练框架允许在 episode 结束时根据最终 `next_obs` 判定成功并给予一次性奖励
  usable_signals: [next_obs] 推导 (contact, dist, speed)
  risks: 无法在标准 `compute_reward` 中获取 terminated 信号，因此这个职责强烈依赖外部包装器，不推荐在纯奖励函数内实现；可考虑在 early stop 后手动加分，但本分析认为该职责当前环境中为 conditional 且实现复杂，不建议作为核心。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: dense_speed_penalty_global**
  reason: 持续惩罚速度会抑制向目标的机动，与“尽快到达”冲突
  forbidden_or_missing_signals: 无适应特例

- **role_id: sparse_exploration_bonus**
  reason: 本任务状态空间小，目标明确，无需额外探索奖励
  forbidden_or_missing_signals: 不需要

- **role_id: survival_time_reward**
  reason: 不适用；生存并非目标，长时间存活而不到达毫无意义
  forbidden_or_missing_signals: 无

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| delta_distance_to_target | obs[0:2], next_obs[0:2] (x,y) | None | delta = dist_prev - dist_next, linear improvement | 核心信号，需配合速降防撞 |
| approach_speed_bonus_with_safety_gate | obs[2:4], obs[0:2] | None | speed_reward = (dot(velocity, to_target_direction) * gate(dist)) / max_speed, gate can be hinge at dist_threshold | 鼓励朝目标移动，但近垫时关闭 |
| soft_landing_stability_penalty | obs[3] (vy), obs[4] (angle), obs[6:8] (contacts) | None | penalty = hinge_abs(vy, soft_limit) + hinge_abs(angle, soft_limit), scaled by proximity to target | 近垫时启用，防高速撞击 |
| engine_efficiency_penalty | action (0-3) | None | penalty = -c if action != 0 else 0 | 小常量惩罚使用引擎 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停在目标上方不下降 | y_position 保持小正值，vy≈0，接触为0，episode 被截断 | 增加接近阶段的下降奖励或调整 delta_distance 的竖直分量权重 |
| 降落后弹跳或翻倒 | 接触标志交替闪烁，body_angle 或 angular_velocity 突然大幅变化 | 加强软着陆惩罚（大垂直速度和角速度），降低下落速度限制 |
| 过度使用姿态引擎导致能量浪费并失控 | 角速度持续高，动作频繁选择1和3，而距离减少缓慢 | 提高 engine_efficiency_penalty 权重，或限制姿态引擎使用频率 |
| 直接侧向飞出视口 | x_position 迅速偏离0并超出边界，无减速 | 增加离垫时的横向速度惩罚，或在边界附近大幅惩罚 |
| 高速撞击目标垫后终止 | 触地时 y_velocity 很大，接触标志为1，episode 终止但未得高分 | 严格限制近垫时最大允许速度，应用 soft_landing_stability_penalty |



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

- target_score: 200.000
- best_score_so_far: -18.800

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| landing_approach_reward + progress + soft_landing_penalty | 1 | -18.800 | -18.800 | unsolved |
| contact_success_reward + landing_approach_reward + progress | 1 | -55.780 | -55.780 | unsolved |
| contact_success_reward + progress + soft_landing_penalty | 1 | -112.840 | -112.840 | unsolved |
| landing_bonus + progress + soft_landing_penalty | 1 | -115.300 | -115.300 | unsolved |
| contact_success_reward + landing_gate + progress | 1 | -115.490 | -115.490 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
