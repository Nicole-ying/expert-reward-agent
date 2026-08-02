# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
任务主目标是控制一台具有左右支撑腿和主/侧向引擎的 2D 着陆器，从视野顶部中央附近以随机初始推力开始，尽快到达场地中央的着陆垫，并稳定、安全地停靠在该垫上。次要目标是尽量减少引擎使用（燃料消耗），同时保持车身姿态稳定，实现软着陆。不应将单纯的快速到达或单纯省燃料作为独立核心目标，代理必须在安全着陆的前提下兼顾速度与能效。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定目标垫并停留，附属有速度、姿态、能耗约束，属于典型的导航目标达到任务。不存在其他同等权重且冲突的核心目标。

动力学子类型进一步判断：  
dynamics_subtype: goal_approach_and_soft_contact  
理由：需要朝着固定目标接近，并在低速、低角速度下通过支撑腿产生安全接触完成着陆。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（根据环境惯例，具体从环境读取，但推测为 float）  
- obs[0]: x_position，相对目标垫的水平坐标，reward_usable: true  
- obs[1]: y_position，相对着陆垫高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，车身俯仰/横滚角度，reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）  
- obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine，不点火，不做任何事情  
- action 1: left_orientation_engine，点燃左姿态引擎（产生方向性的力）  
- action 2: main_engine，点燃主引擎（通常向上推力）  
- action 3: right_orientation_engine，点燃右姿态引擎（与左引擎反向）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当 `body_not_awake_or_settled` 为真，且未发生 crash/出界时，可推断为成功着陆并稳定。
- failure-like termination: `crash_or_body_contact`（猛烈碰撞或非法身体接触）和 `horizontal_position_outside_viewport`（水平飞出视野）均视为失败。
- ambiguous termination: 单独的 `body_not_awake_or_settled` 可能发生在成功着陆后不久，也可能是因摔落后静止，需结合位置、速度判断。
- truncation: 环境中未提供 truncation 信号（step 返回 `terminated, False`），因此不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（step 返回 info={}）  
- forbidden_or_uncertain_info_fields: 任何 info 字典中的字段均不存在，不可使用

间接推断路径：
- 成功着陆可从 termination 时接近目标垫中心、垂直速度极低、角度接近 0 且至少一腿接触的条件组合推断（derived_possible）。
- crash 可从 termination 时 y_position 骤降至地面以下或水平位置远超边界等条件间接检测（derived_possible）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前观测，8维）
- action（当前动作，整数 0~3）
- next_obs（下一步观测，8维）
- info 中明确允许的字段（当前环境均为空，无可用字段）
- training_progress（本环境 prompt 未明确允许，默认不应使用）

禁止使用：
- original_reward（原始奖励被屏蔽）
- official_reward
- 未在观测空间中定义的 obs 切片
- info 中任何未声明的字段
- 假设终止原因字符串或 success 标志存在

## 7. 可用于奖励函数的信号
- position：
  - 相对垫的水平距离 `|x_position|`
  - 相对垫高度的垂直距离 `|y_position|`（当 y_position 为正代表高于垫，到达垫面时理想 y≈0）
- velocity：
  - 水平速度 `x_velocity`（软着陆要求接近0）
  - 垂直速度 `y_velocity`（负值代表下落，着陆瞬间需要小）
- orientation：
  - body_angle（理想接近0，可取其绝对值或二次惩罚）
  - angular_velocity（软着陆应接近0）
- contact：
  - left_support_contact / right_support_contact（至少一腿接触可能表明着陆成功，但需结合速度和位置，否则可能鼓励猛烈砸地）
- action/engine：
  - action 本身（可计算引擎使用惩罚，no_engine 时无推力）
- other：
  - 通过 next_obs 与 obs 的差值得出速度变化，可用于检测剧烈推力
  - 衍生信号：是否接近目标垫且速度降低（derived_possible）

## 8. 不确定或不可用的信号
- info["success"] 或 info["failure"]：不存在
- 真实燃料消耗量、引擎推力大小：环境抽象为离散动作，无直接推力值
- 外部风扰等不可观测因素：无相关观测
- 任务耗时或剩余时间：无此信息

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D lander with two support legs
  actuator_type: one main engine (upward thrust) and two lateral orientation engines
  contact_structure: two legs ground contact (binary flags)
primary_objectives:
  - reach target pad (x≈0, y≈0 relative to pad)
  - achieve soft landing (low vertical velocity, near-zero horizontal velocity, near-zero angle)
secondary_objectives:
  - minimize engine ignition count (fuel/thrust penalty)
  - fast arrival (through truncation pressure or sparse bonus)
main_failure_risks:
  - crash due to high vertical speed or flipped angle
  - flying out of bounds horizontally
  - hovering but never landing (inefficient)
  - slamming onto pad by relying on contact reward too early
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 引导代理向目标垫中心移动
  why_required: 没有此职责代理无法得知目标方位
  usable_signals: [x_position, y_position, 派生距离]
  risks: 可能导致高速撞击目标垫，需与软着陆结合

- role_id: soft_landing_velocity
  purpose: 在接近目标时抑制水平和垂直速度，使着陆平稳
  why_required: 没有减速约束会导致硬着陆或撞毁
  usable_signals: [x_velocity, y_velocity, 接近目标时的位置]
  risks: 过早施加减速奖励可能阻碍接近阶段，需与 proximity 联合调节

- role_id: orientation_stability
  purpose: 保持车身水平（角度和角速度小）
  why_required: 大角度会导致支撑腿不能安全接触或增加 crash 风险
  usable_signals: [body_angle, angular_velocity]
  risks: 可能在飞行阶段抑制必要的姿态微调

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 惩罚非必要的引擎使用，鼓励节省燃料
  condition_to_use: 在整个 episode 中启用，但权重应较低，并且当代理已接近 pad 且即将着陆时可适当降低惩罚以避免不点火而漂移
  usable_signals: [action]（encourage action 0）
  risks: 过强的燃料惩罚会导致代理不愿点火而无法控制姿态或减速

- role_id: soft_contact_bonus
  purpose: 当代理以低速、小角度并位于垫上方时腿接触给予小量正向奖励
  condition_to_use: 仅在 (|x_position| < 阈值) 且 (|y_velocity| 小) 且 (|x_velocity| 小) 且 (|body_angle| 小) 时启用
  usable_signals: [left_support_contact, right_support_contact, x_position, y_velocity, x_velocity, body_angle]
  risks: 不加条件直接奖励接触会鼓励摔落式砸地

### 10.3 慎用/禁用职责 avoid_roles
- role_id: success_termination_bonus
  reason: 环境未提供显式成功标志，无法在 episode 结束时可靠发放离散奖励；且用间接推断易出错
  forbidden_or_missing_signals: [info.success, explicit_success_flag]

- role_id: time_penalty
  reason: 无时间截断信号，无法按时间步惩罚；但可通过期望快速完成的压力用 future return 体现

- role_id: dense_orientation_penalty_in_early_phase
  reason: 初期随机推力可能引起大幅摆动，若过早严苛惩罚角度会抑制探索；应逐渐加强

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position, derived_distance | — | dense_state_signal, l2_norm, exponential_decay | 距离越近奖励越高 |
| soft_landing_velocity | x_velocity, y_velocity, x_position (as condition) | — | bounded_signal, quadratic_penalty, conditional_gating | 在接近目标时激活速度惩罚 |
| orientation_stability | body_angle, angular_velocity | — | absolute_penalty, quadratic_penalty | 全 episode 使用，可随目标接进度动态调整权重 |
| fuel_efficiency | action | engine_thrust_metric | action_penalty (binary for non-zero) | 全 episode 启用 |
| soft_contact_bonus | left_support_contact, right_support_contact, x_position, y_velocity, x_velocity, body_angle | — | conditional_reward, logical_and | 仅当多条件满足时给予 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 代理悬停不动，不敢降落 | episode 长度长，位置保持在远处，动作多为 no_engine | 降低燃料惩罚权重，增加距离奖励的渐进压力 |
| 高速撞击目标垫 crash | 支撑腿接触时垂直速度过大，或 body_angle 大 | 增强 soft_landing_velocity 和 orientation_stability 权重，或在接触条件中要求速度<阈值 |
| 飞出水平边界 | x_position 超过某个极限导致终止 | 增加距离中心越远越重的惩罚，并可小量水平位移惩罚 |
| 过早依赖引擎侧推导致抖振 | 姿态频繁切换，angular_velocity 大 | 追加角度变化率的惩罚，或在早期使用较小 action_penalty |
| 左右腿接触奖励被错误奖励摔落 | 接触次数多但 episode 短且 crash 率高 | 收紧 soft_contact_bonus 的条件门，尤其严格限制速度、角度 |



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

