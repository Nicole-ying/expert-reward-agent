# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个二维飞行器/着陆器轨迹优化任务。agent 从视窗上方中央附近以随机初始力开始，需要尽快、省油地**到达视窗中央的目标着陆平台，并以安全姿态稳定接触**（即实现软着陆）。  
核心是导航到目标并实现 safe and stable contact，附属优化是节省发动机推力（能量效率）和缩短耗时，但不改变核心目标。  
**不可混淆**：任务不是持续前行（没有前进方向），也不是纯粹的存活（没有存活计时器），而是**定点到达 + 停稳**。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 任务的核心问题是“到达并停稳在目标点”，到达目标位置是主目标，节省燃料和快速是附属优化。不属于 locomotion（无持续前进轴）、不属于 survival（目标不是一直活着）、不属于 sparse exploration（有明显目标距离信号），也不存在多个权重相等且冲突的核心目标（如既要快速又要非常省油但快和省油都是可量化的副目标，到达目标是严格必要条件），因此不划为 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（推测）  

维度说明（索引从 0 开始，均为可用信号，reward_usable 均为 true）：

- **obs[0]**: `x_position` — 飞行器质心相对目标着陆平台的水平坐标（单位未知，相对值）。reward_usable: true  
- **obs[1]**: `y_position` — 飞行器质心相对平台高度的垂直坐标（下正？待确认方向；通常上正，但可通过初始位置和下降过程推断方向）。rewars_usable: true  
- **obs[2]**: `x_velocity` — 水平线速度。reward_usable: true  
- **obs[3]**: `y_velocity` — 垂直线速度。reward_usable: true  
- **obs[4]**: `body_angle` — 机体方向角（弧度）。reward_usable: true  
- **obs[5]**: `angular_velocity` — 角速度。reward_usable: true  
- **obs[6]**: `left_support_contact` — 左支撑腿是否接触平台（布尔化 float: 1.0/0.0）。reward_usable: true  
- **obs[7]**: `right_support_contact` — 右支撑腿是否接触平台。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  

动作含义：
- **action 0**: `no_engine` — 不启动任何发动机（滑行）。
- **action 1**: `left_orientation_engine` — 点燃左定向发动机，产生侧向/旋转力矩，可改变机体角度并小幅移动。
- **action 2**: `main_engine` — 点燃主发动机，产生主要推力（推测在机体坐标系向上或向下，结合角度影响水平和垂直速度）。
- **action 3**: `right_orientation_engine` — 点燃右定向发动机，与左对称，改变旋转和侧向移动。

## 5. step 与终止条件分析
### 5.1 终止模式
环境给出三个终止条件，经抽象后为：
- `crash_or_body_contact` — 飞行器坠毁或身体其他部分（非支撑腿）接触地面/平台，属于 likely failure。
- `horizontal_position_outside_viewport` — 水平坐标超出视窗范围，显然为 failure。
- `body_not_awake_or_settled` — 飞行器“休眠”或已经稳定停靠，这**很可能对应成功软着陆**（双腿接触且速度、角度足够小后触发）。由于任务目标是到达并 settle，该条件可作为 success-like termination。

当前 info 字典为空，无任何 explicit success/failure flag。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: 无（info 为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段（不可用）

**成功推断路径**（derived_possible）：  
当 episode 终止（terminated=True）且最后一次观测满足：  
 `left_support_contact == 1.0 && right_support_contact == 1.0`  
 并将 `abs(body_angle)`、`|x_velocity|`、`|y_velocity|` 控制在很小阈值内，且 `x_position` 和 `y_position` 接近零，则可认为发生了成功软着陆。  
**失败推断路径**：  
若终止时 `abs(x_position)` 很大（出界），或存在坠毁迹象（极端 body_angle 突变、两腿未接触），可判断为失败。由于缺少身体接触传感器，无法直接获得碰撞信号，角度过陡、速度冲击可作为间接证据。

## 6. reward 函数接口契约
函数签名（由调用方约定）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` (8维np.ndarray)  
- `action` (int)  
- `next_obs` (8维np.ndarray)  
- `info` **中明确允许的字段** → 当前无允许字段（info为空），故**禁止使用任何 info 字段**  
- `training_progress` **仅在任务描述或 prompt 明确允许时可用** → 当前未明确允许，故**禁止使用**  

禁止使用：
- `original_reward`  
- 任何官方奖励内部变量  
- 任何未在上述允许清单中出现的数据（包括未声明的 obs 切片、未经允许的环境内部状态）

## 7. 可用于奖励函数的信号
由于 info 不可用，reward 只能依赖 `obs`、`action` 和 `next_obs`。

- **位置**：`obs[0], obs[1]` 和 `next_obs[0], next_obs[1]`  
- **速度**：`obs[2], obs[3]` 和 `next_obs[2], next_obs[3]`  
- **姿态**：`obs[4]` (body_angle) 和 `next_obs[4]`  
- **角速度**：`obs[5]` 和 `next_obs[5]`  
- **接触**：`obs[6], obs[7]` 和 `next_obs[6], next_obs[7]` (双腿接触标志)  
- **动作**：`action` 值（离散 0-3），可用于动作效率惩罚/奖励

**可从观测间接推断的衍生信号**（derived_possible）：  
- 成功率线索：两腿接触 + 小速度 + 小倾角 + 接近零位置 → 可推断成功着陆  
- 坠毁线索：倾角突然超过安全阈值（如 abs(angle)>某一临界值）或速度骤变 → 可推断碰撞

## 8. 不确定或不可用的信号
- 任何显式的 `success`、`failure`、`termination_reason` 标志（info 为空）  
- 身体其他部分接触传感器（只有支撑腿接触）  
- 燃料消耗量/剩余燃料（未提供）  
- 时间步计数或已消耗时间（未在观测或 info 中给出）  
- 平台的实际坐标（相对位置已给出，但绝对坐标可能未知）  
- 发动机推力大小（动作是离散的，推力效果隐藏在动力学中，无法直接获知）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid-body lander
  actuator_type: two side orientation thrusters + one main thruster (discrete on/off)
  contact_structure: two landing legs with contact flags (left/right)
primary_objectives:
  - 到达目标位置：将 (x_position, y_position) 基本降至 (0, 0) 附近
  - 软着陆：接触时速度近乎零 (x_velocity, y_velocity 接近 0)
  - 安全姿态：着陆时 body_angle 接近 0（竖直向上或与平台垂直）
  - 双支撑稳定：着陆时 left_support_contact==1 且 right_support_contact==1
secondary_objectives:
  - 快速完成（隐含时间压力，但无时间指标可用，只能通过奖励塑造中的速度导向间接影响）
  - 节能：尽量减少推力使用（特别是主发动机），动作越少、越零越好
main_failure_risks:
  - 高速撞击平台或地面（速度过大着陆）
  - 机体倾翻（angle 过大导致侧翻或头部触地）
  - 水平飞出视窗（x 超出范围）
  - 悬停过度



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

