# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
任务主目标是控制 2D 飞行器从视口顶部中央附近开始，尽快到达并稳定降落在中央目标平台上。降落必须安全：接近目标点、速度极低、姿态稳定、身体与平台温和接触。次要目标包括尽量减少发动机使用（燃料经济），隐含了速度‑燃料之间的权衡。不应混淆为纯粹的存活平衡问题或连续地形行走问题，其核心是导航到达与软着陆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 主要目标是到达指定目标位置（着陆 pad），并进行安全接触与稳定停留。快速到达与节省燃料为附属优化项，并非权重相当的多目标。动力学强调接近目标、减速、姿态调整、软着陆，完全符合导航到达类型。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（基于典型实现，但未显式声明，可认为是浮点）  
- 各维度含义及奖励可用性：  
  - obs[0]: x_position（水平相对目标距离），reward_usable: true  
  - obs[1]: y_position（垂直相对目标高度），reward_usable: true  
  - obs[2]: x_velocity（水平线速度），reward_usable: true  
  - obs[3]: y_velocity（垂直线速度），reward_usable: true  
  - obs[4]: body_angle（机体角度），reward_usable: true  
  - obs[5]: angular_velocity（角速度），reward_usable: true  
  - obs[6]: left_support_contact（左支腿接触标志，1/0），reward_usable: true  
  - obs[7]: right_support_contact（右支腿接触标志，1/0），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：  
  - action 0: no_engine（无发动机推力）  
  - action 1: left_orientation_engine（左姿态发动机点火）  
  - action 2: main_engine（主发动机点火，提供向上推力为主）  
  - action 3: right_orientation_engine（右姿态发动机点火）  

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: 身体处于非活跃或已稳定（body_not_awake_or_settled），同时未触发 crash 或出界。此条件通常意味着着陆成功并静止。  
- failure‑like termination:  
  - crash_or_body_contact（极可能是坠毁性接触，如机身撞地、速度过高时的接触）。  
  - horizontal_position_outside_viewport（水平飞出视口）。  
- ambiguous termination: body_not_awake_or_settled 单独无法区分成败，需结合其他条件，但代码中终止判断为“crash_or_body_contact or out of viewport or not awake/settled”，一旦满足任一即终止，因此成功着陆必然伴随 not_awake/settled 且无 crash/出界。  
- truncation: 未提及时间限制，但可能存在最大步长截断（无显式声明）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（step 返回 info = {}）  
- forbidden_or_uncertain_info_fields: 所有 info 字段不可用，包括任何可能存在的 “success”“failure”“termination_reason”  

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用的信号：
- obs 全部 8 维  
- action（0~3 离散值）  
- next_obs（同 obs 结构）  
- training_progress（本次 prompt 未明确允许使用，建议忽略）  

严格禁止使用的信号：
- original_reward（被屏蔽，必须忽略）  
- info（任何字段，包括可能存在的 success、failure 标志）  
- 任何未声明的 obs 切片或全局变量  

## 7. 可用于奖励函数的信号
- 位置偏差：obs[0]（x 距离），obs[1]（y 距离）  
- 速度：obs[2]（vx），obs[3]（vy）  
- 姿态：obs[4]（角度），obs[5]（角速度）  
- 接触：obs[6]（左支腿接触），obs[7]（右支腿接触）  
- 动作：action ∈ {0,1,2,3}（可用于燃料惩罚或鼓励特定行为）  
- next_obs：可提供下一时刻状态，但应避免依赖未来的接触标志等，可做保守使用（如用于差分速度奖励）

## 8. 不确定或不可用的信号
- 明确的“坠毁”标签或大惩罚信号：无直接标志，只能通过位置越界、速度过高、角度过大等间接推断  
- 成功着陆标志：无  
- 任务完成度（progress）：不可直接获得  
- 剩余时间或步数：不可用  
- 原环境奖励：官方奖励被隐藏，不可在自定义奖励中引用  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 刚体飞行器
  actuator_type: |-
    主发动机（垂直推力为主）+ 两个姿态发动机（提供旋转力矩）
  contact_structure: 左右两个支腿，各自有独立的接触检测
primary_objectives:
  - 安全降落到目标平台（位置靠近 x≈0,y≈0，速度极低，角度趋于0）
  - 避免坠毁、飞出视口或无法稳定
secondary_objectives:
  - 最小化发动机使用（节省燃料）
  - 快速完成降落（隐含在燃料与速度的权衡中）
main_failure_risks:
  - 下降速度过大导致硬着陆（crash）
  - 姿态失控，机身撞击地面或平台边缘
  - 水平漂移出视口
  - 长时间振荡无法稳定着陆
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: position_approach  
  purpose: 引导飞行器朝目标 pad 移动，减小 x 和 y 的偏移。  
  why_required: 这是到达任务的基本驱动，没有它 agent 不会向目标靠近。  
  usable_signals: [obs[0], obs[1]]（也可用 next_obs 的同位置）  
  risks: 如果权重过高可能忽视姿态和速度，导致快速但猛撞。  

- role_id: velocity_damping  
  purpose: 要求接触前速度极低，尤其在垂直方向。  
  why_required: 软着陆的核心条件，速度过高将被视为失败。  
  usable_signals: [obs[2], obs[3]]，以及它们的平方或绝对值。  
  risks: 太强的阻尼会使 agent 行动过于保守，下降太慢。  

- role_id: orientation_stabilization  
  purpose: 保持机体接近竖直（角度趋向0），抑制角速度。  
  why_required: 过大倾斜会导致不安全接触或支腿无法同时触地。  
  usable_signals: [obs[4], obs[5]]  
  risks: 过度惩罚角速度可能抑制必要的姿态调整。  

- role_id: fuel_penalty  
  purpose: 对使用发动机的动作施加负奖励，鼓励低推力策略。  
  why_required: 实现“尽可能少用引擎”的次要目标，同时避免无意义的推力浪费。  
  usable_signals: [action]  
  risks: 过重可能导致 agent 不敢点火，无法减速或纠正姿态。  

- role_id: crash_avoidance  
  purpose: 惩罚飞行器进入危险状态（极速、极端姿态、远离 pad）。  
  why_required: 缺乏明确坠毁信号时，必须通过状态约束避免灾难性行为。  
  usable_signals: [obs[0], obs[1], obs[2], obs[3], obs[4]] 的阈值组合。  
  risks: 阈值设定不当可能产生错误惩罚或过于松驰。  

### 10.2 条件职责 conditional_roles
- role_id: soft_contact_encouragement  
  condition_to_use: 当飞行器已非常接近目标（|x|、|y| 小于阈值）且速度较低时，鼓励左右支腿同时接触。  
  usable_signals: [obs[6], obs[7]]  
  risks: 过早鼓励接触可能导致 agent 在远离目标时做无用触地尝试；与 position_approach 可能冲突。  

### 10.3 慎用/禁用职责 avoid_roles
- role_id: success_bonus  
  reason: 无显式成功标志，无法判断任务完成时刻，强行使用会导致稀疏且无法定义。  
  forbidden_or_missing_signals: [success_flag (info 中不存在)]  

- role_id: time_efficiency_penalty  
  reason: 无法获取 episode 时间或剩余步数，训练进度信号 training_progress 未被明确允许使用。  
  forbidden_or_missing_signals: [time_step, episode_length]  

- role_id: progress_speed_bonus  
  reason: 没有明确的进度度量（如已移动距离占全程的比例）；直接鼓励高速将损害安全要求。  
  forbidden_or_missing_signals: [normalized_progress]  

- role_id: energy_efficiency（单独再列）  
  reason: 已由 fuel_penalty 涵盖，且缺少精确能耗计算公式。  
  forbidden_or_missing_signals: [thrust_magnitude, fuel_consumption]  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| position_approach | obs[0], obs[1] | – | negative\_l2\_norm, negative\_manhattan | 可分段设置不同权重，远距离可更宽松 |
| velocity_damping | obs[2], obs[3] | – | negative\_squared, negative\_abs | y 速度通常需要更大权重 |
| orientation_stabilization | obs[4], obs[5] | – | negative\_abs_angle, quadratic\_penalty | 角速度也应衰减 |
| fuel_penalty | action | – | step\_constant\_penalty, per\_action\_cost | 不同动作可有不同代价（主引擎更贵） |
| crash_avoidance | obs[0], obs[1], obs[2], obs[3], obs[4] | crash\_flag | threshold\_penalty, large\_constant\_penalty | 需要设定安全边界并给予重罚 |
| soft_contact_encouragement | obs[6], obs[7] (以及 obs[0], obs[1], obs[2], obs[3] 作为条件) | – | binary\_bonus, conditional\_reward | 建议仅在近目标低速时激活 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过度振荡，无法稳定着陆 | episode 长度过长；位置/速度/角度持续波动 | 增强 velocity_damping 和 angle 惩罚，调整 gain |
| 高速直冲 pad，导致 crash | 最终几步速度极大，奖励曲线骤降 | 加大 crash_avoidance 阈值惩罚，提前施加速度阻尼 |
| 不使用主引擎，仅靠姿态引擎漂移 | action 统计中主引擎使用率极低，下降过慢 | 减轻 fuel_penalty 对主引擎的惩罚，或引入时间衰减 |
| 忽略姿态，机身横躺接触 | 最终 body_angle 很大，仅单腿接触 | 强化 orientation_stabilization 和 soft_contact 中双侧接触要求 |
| 长时间悬停不敢下降 | y 速度长期为正或接近零，episode 超时 | 减轻 fuel_penalty 或 velocity_damping 对下降的限制，可加入轻柔的高度下降奖励 |
| 飞出视口 | x 坐标超出安全范围 | 增加对 x 偏移的急剧惩罚，甚至大常数负奖 |



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
- best_score_so_far: -46.510

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_cost + angle_penalty + progress_shaping + success_landing_bonus | 5 | -46.510 | -98.930 | unsolved |
| action_cost + angle_penalty + landing_velocity_penalty + progress_shaping | 1 | -155.980 | -155.980 | unsolved |
| action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus | 1 | -203.450 | -203.450 | unsolved |

## Previous interventions

- iter 2 (score=-119.510, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2——着陆速度惩罚组件形式错误（稀疏惩罚且门控过窄，形同虚设），需替换为成功着陆奖励，即从惩罚形态变为正向bonus。 | 5. `selected_intervention`：将`landing_velocity_penalty`组件整体替换为`success_landing_bonus`，基于next_obs的接触、位置、速度、角度给出稀疏大正奖励。
- iter 3 (score=-203.450, structure=action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus): 4. selected_level：Level 1（系数/惩罚调整）。progress_shaping 形态本身合理，但需要补充 crash 惩罚以阻断 crash 路径，引导 agent 尝试更安全的着陆。 | 5. selected_intervention：新增 `crash_penalty` 组件，在接触地面但未满足软着陆条件、或 x 位置超出视区边界时给予 -100，与 success_landing_bonus 互斥，其余组件不变。
- iter 4 (score=-114.380, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2，修改 success_landing_bonus 的触发条件，放宽阈值，同时移除 crash_penalty 以恢复 v2 有效结构。 | 5. `selected_intervention`：删除 crash_penalty，将 success_landing_bonus 的判定距离阈值放宽到 0.2，速度阈值放宽到 0.5，角度阈值放宽到 0.2。
- iter 5 (score=-46.510, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2，基于v4成功经验进一步放宽success_landing_bonus的触发阈值以提升active_rate，不做框架级重建。 | 5. `selected_intervention`：将success_landing_bonus的判定条件中的距离阈值由0.2提至0.4，水平/垂直速度阈值由0.5提至1.0，角度阈值由0.2提至0.5，保持接触条件不变。
- iter 6 (score=-111.180, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2，success_landing_bonus的数学形态传达了错误激励（状态值可连续触发），需改为事件触发并收紧阈值。 | 5. `selected_intervention`：将success_landing_bonus修改为仅在从无接触转为有接触时触发，且收紧软着陆条件（距离<0.3，速度绝对值<0.3，角度绝对值<0.3），奖金提高至200。
- iter 7 (score=-98.930, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2——修改 success_landing_bonus 的数学形态，从每步状态值奖励改为基于接触转变的一次性事件奖励，使用 v5 的宽松阈值以避免死组件，同时解决滞留。 | 5. `selected_intervention`：将 success_landing_bonus 改为在支撑脚从无接触到有接触的瞬间，若满足 v5 的软着陆条件（dist<0.4, vel_abs<1.0, angle_abs<0.5）则给予 +100，其余时刻为 0。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
