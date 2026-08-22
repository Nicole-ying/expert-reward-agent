# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
任务主体是一个2D飞行器/着陆器，初始从视野上方中央区域出发并带有随机初速。  
**主目标**：尽快飞到中央目标平台并稳定停靠（settle），要求位置精准、速度近乎为零、姿态稳定，并且两个支撑脚均安全接触地面。  
**次目标**：在此过程中尽可能减少引擎使用（节省燃料），并隐含要求快速完成（时间效率）。  
**不应混淆的目标**：燃料效率与快速到达是一对次要权衡，但不应喧宾夺主；安全平稳着陆优先于速度与能耗。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达并稳定于一个固定的空间位置（目标平台），姿态和速度要求都是为实现“到达并安全停靠”服务，属于典型的导航目标到达类，只是附带了软着陆的动力学约束。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（推断）  
- 各维度定义与可用性：
  - obs[0]: x_position，水平相对目标平台的坐标，reward_usable: true  
  - obs[1]: y_position，垂直相对平台高度的坐标，reward_usable: true  
  - obs[2]: x_velocity，水平线速度，reward_usable: true  
  - obs[3]: y_velocity，垂直线速度，reward_usable: true  
  - obs[4]: body_angle，机体倾角，reward_usable: true  
  - obs[5]: angular_velocity，角速度，reward_usable: true  
  - obs[6]: left_support_contact，左支撑脚接地标志（1.0/0.0），reward_usable: true  
  - obs[7]: right_support_contact，右支撑脚接地标志（1.0/0.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - action 0: no_engine (不做任何推进)  
  - action 1: left_orientation_engine (启动左侧姿态调整引擎)  
  - action 2: main_engine (启动主引擎，提供主要推力)  
  - action 3: right_orientation_engine (启动右侧姿态调整引擎)

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：出现碰撞或非预期的机体接触（典型失败）。  
- **horizontal_position_outside_viewport**：水平位置出界（失败）。  
- **body_not_awake_or_settled**：机体不再活跃/已稳定停靠。此条件**可能代表成功**（如果在目标位置附近、速度低、双足接地），但也可能因为其他原因“静默”失败（如卡在视口外、姿态错误而静止）。需结合状态判断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 返回 {}，不提供任何额外字段）  
- forbidden_or_uncertain_info_fields: info 内所有假设字段（如 “success”, “failure”, “termination_reason” 等）均不存在且不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- obs（当前观测）
- action（本次动作）
- next_obs（下一步观测）
- info（当前已知为空，不可使用任何字段）
- training_progress（仅在任务明确允许时使用）

**禁止使用：**
- original_reward 或任何形式的官方奖励
- 未声明的 info 字段
- 未在观测空间中声明的维度或假设的隐藏状态

## 7. 可用于奖励函数的信号
全部来自观测：
- **位置**：x_position, y_position（目标参考零点）  
- **速度**：x_velocity, y_velocity  
- **姿态与角速度**：body_angle, angular_velocity  
- **接触信号**：left_support_contact, right_support_contact  
- **动作**：action（离散值 0-3）

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：无  
- **时间步数 / elapsed steps**：未提供，不可直接奖励“快速”  
- **燃料消耗量**：未提供，不能直接奖励低能耗，只能通过动作选择间接引导  
- **接触类型细节**：仅有支撑脚接触标志，无法区分安全着陆与碰撞  
- **目标距离的精确度量**：可通过位置计算，但官方未给出额外信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: planar_lander  # 2D着陆器外形
  actuator_type: discrete_thrusters  # 一个主推力和两个姿态调整推力
  contact_structure: two_leg_support_with_contact_flags
primary_objectives:
  - 到达目标位置 (x=0, y=0)
  - 稳定停靠（速度≈0，姿态稳定）
  - 双支撑脚同时接地
secondary_objectives:
  - 节省引擎使用（燃料效率）
  - 尽快速达（时间效率）
main_failure_risks:
  - 撞击目标区域（高速/错误姿态导致 crash）
  - 水平漂移出界
  - 到达目标但姿态倾斜导致翻倒或无法稳定
  - 提前“假静默”而未正确着陆
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: target_approach**  
  purpose: 鼓励减小与目标位置的欧氏距离  
  why_required: 核心到达任务  
  usable_signals: [x_position, y_position]  
  risks: 单纯距离奖励会导致高速撞击，需与速度控制耦合。

- **role_id: soft_landing_speed**  
  purpose: 在接近目标时抑制速度，尤其是垂直接触速度  
  why_required: 实现“settle”的低速接触要求  
  usable_signals: [x_velocity, y_velocity, x_position, y_position]（联合使用，按距离加权）  
  risks: 早期过度惩罚正常机动速度，应渐进激活。

- **role_id: stable_orientation**  
  purpose: 维持机体接近竖直，抑制角速度  
  why_required: 安全接地与姿态稳定  
  usable_signals: [body_angle, angular_velocity]  
  risks: 可能限制必要的机动，需在着陆末段加强。

- **role_id: contact_completion**  
  purpose: 奖励双足同时接地  
  why_required: 标志着陆成功、稳定停靠  
  usable_signals: [left_support_contact, right_support_contact]  
  risks: 过早奖励地面接触可能导致在平台外着陆，需结合位置条件使用。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**  
  condition_to_use: 当 agent 已经接近目标且速度较低时，鼓励选择 no_engine 动作  
  usable_signals: [action, x_position, y_position, x_velocity, y_velocity]  
  risks: 若条件过严，可能阻碍必要的机动或主引擎减速。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: time_minimization**  
  reason: 缺少步数或计时信号，无法准确奖励快速到达；强行奖励可能导致危险加速  
  forbidden_or_missing_signals: [elapsed_steps, episode_length]（环境未提供）

- **role_id: explicit_collision_penalty**  
  reason: 无碰撞细节信号，终止环境时已触发失败，不适合在奖励函数中重复惩罚  
  forbidden_or_missing_signals: [collision_intensity, impact_force]

## 11. role_to_signal_mapping
| role_id              | usable signals                                           | missing signals | candidate formula operators        | notes                                                                 |
|----------------------|----------------------------------------------------------|-----------------|------------------------------------|-----------------------------------------------------------------------|
| target_approach      | x_position, y_position                                   | -               | distance_to_target, negative_distance | 可采用平滑有界惩罚，如 -sqrt(x²+y²) 或 -tanh(distance)              |
| soft_landing_speed   | x_velocity, y_velocity, x_position, y_position           | -               | speed_magnitude, distance_gated_penalty | 速度惩罚可用距离门控：距离小权重增大，或使用 -speed*exp(-dist²) 等形式 |
| stable_orientation   | body_angle, angular_velocity                             | -               | orientation_error, angular_speed_penalty | 可使用角度偏差二次惩罚和角速度二次惩罚                               |
| contact_completion   | left_support_contact, right_support_contact, x_position, y_position | -               | logical_and, dense_contact_reward       | 仅当接近目标且速度低时奖励双接触，避免过早奖励                         |
| fuel_efficiency      | action, x_position, y_position, x_velocity, y_velocity   | -               | action_cost, conditional_idle_bonus    | 动作0可获得小额奖励，但必须在安全接近阶段启用                         |

## 12. 初始训练后应观察的 failure modes
| failure_mode                      | evidence_to_check                                                                     | possible_intervention                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 高速撞击目标区域导致 crash        | 最后几步观测中 x_position、y_position 接近0但 speed 很高，且 episode 终止于 crash     | 加强 soft_landing_speed 职责，尤其是在接近目标时的速度惩罚权重或渐进因子              |
| 漂移出界                          | 水平坐标绝对值持续增大，未收敛到0                                                     | 强化 target_approach 奖励，或加入边界警告（但仍只能用观测）                         |
| 着陆姿态倾斜导致非稳定接地        | body_angle 较大，或双足接触不平衡，但 episode 因 body_not_awake 终止而位置不佳        | 增强 stable_orientation 与 contact_completion 的协同，提高姿态精度的敏感度           |
| 引擎过度使用                      | 大量主引擎或姿态引擎动作，燃料效率低                                                 | 在安全接近阶段适当引入 fuel_efficiency 条件奖励，设置合理的距离阈值                 |
| “假着陆” – 在错误位置静默         | episode 因 body_not_awake 终止，但 x_position 或 y_position 远离零点，且接触标志可能为真 | 必须将 contact_completion 与位置/速度条件强绑定，拒绝在远离目标时的 ground contact 奖励 |



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

