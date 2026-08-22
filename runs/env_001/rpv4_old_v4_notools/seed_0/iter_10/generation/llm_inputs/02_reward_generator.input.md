# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
任务主体是一个 2D 刚体飞行器/着陆器，从视口上方中央附近出发，初始受随机力。
**主目标**：尽快到达并稳定停靠在场景中央的目标垫上，实现双腿同时接触、姿态平稳、速度极低，且全程安全不碰撞、不越界。
**次目标**：尽量少用引擎推力（燃料节省），隐式要求动作决策高效。
**不应混淆的目标**：单纯追求极致的速度而不计安全，或只追求零耗能而停滞不前。

## 2. 任务类型选择
- selected_route_id: navigation_goal_reaching
- confidence: high  
- reason: 核心是到达并停稳在特定目标位置（目标垫），附属的速度、能耗、姿态要求都是围绕这一主要目标展开，不存在多个权重相当、彼此互斥且无法区分主次的目标，因此属于导航目标到达族。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position，相对于目标垫的水平坐标，reward_usable: true
- obs[1]: y_position，相对于目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体朝向角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无推力
- action 1: left_orientation_engine — 启动左侧姿态调节引擎
- action 2: main_engine — 启动主引擎（提供主要推力）
- action 3: right_orientation_engine — 启动右侧姿态调节引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  body_not_awake_or_settled 可能表示机体稳定静止（速度、角速度极小），配合位置接近目标和双支撑腿接触，大概率是成功着陆。  
  但该条件本身不检查位置，可能提前在错误位置稳定，因此属于模糊终止。
- failure-like termination:  
  crash_or_body_contact（机体或部件异常碰撞，如头部触地），  
  horizontal_position_outside_viewport（水平越界）。
- ambiguous termination:  
  body_not_awake_or_settled（如上述）
- truncation: 无

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: {} （空字典，无可用字段）  
- forbidden_or_uncertain_info_fields:  
  - info 整体不可用，不应依赖任何 info 键值  
  - 无法直接知晓官方奖励值（被遮蔽）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs, action, next_obs（全部合法位置通道）
- info 仅限空字典，无可用字段
- training_progress 仅在明确需要时可用，本环境未强制要求使用

禁止使用：
- original_reward（已遮蔽）
- 任何未声明的 info 字段
- 任何对 obs 的越界或未说明切片

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) — 直接度量到目标的距离
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) — 可控制着陆冲击与稳定性
- orientation: obs[4] (body_angle) — 姿态控制，obs[5] (angular_velocity) — 旋转稳定性
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) — 双腿着地指示
- action/engine: action (离散 0~3) — 可构建燃料惩罚或推力惩罚
- other: 无

## 8. 不确定或不可用的信号
- 官方成功/失败标志：不存在  
- 环境给出的稀疏终局奖励：被遮蔽  
- 时间步数/剩余时间：不可见  
- 目标垫的确切接触力/冲击力：不可见  
- 风场等外部扰动：未知  
- 任何 info 内容：不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body (2D vehicle/lander)
  actuator_type: main_thruster, left_orientation_thruster, right_orientation_thruster
  contact_structure: two_support_legs (left/right) with binary contact sensors
primary_objectives:
  - reach_target_pad (x ≈ 0, y ≈ 0)
  - establish_stable_biped_contact (both legs on ground, low velocity, small angle)
secondary_objectives:
  - minimize_engine_usage (fuel efficiency)
  - minimize_time_to_settle (fast settling)
main_failure_risks:
  - crashing or touching ground with non-leg parts
  - drifting outside horizontal boundaries
  - settling prematurely away from the pad
  - flipping or large orientation deviation
  - excessive fuel consumption preventing arrival
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 驱使智能体向目标垫中心移动。  
  why_required: 主目标为到达，无此奖励无法形成到达行为。  
  usable_signals: [obs[0], obs[1]]  
  risks: 可能导致高速直接撞击而非软接触，需配合速度控制。

- role_id: velocity_control  
  purpose: 靠近目标时减小速度，避免剧烈碰撞或反弹。  
  why_required: 着陆安全与稳定需要低速，否则可能跳起或损坏。  
  usable_signals: [obs[2], obs[3], obs[0], obs[1]（用于距离门控）]  
  risks: 过早减速可能延长到达时间。

- role_id: stable_contact  
  purpose: 鼓励双腿同时触地并维持接触，防止单腿着陆或翻倒。  
  why_required: 任务要求“make safe contact”且稳定停靠，双支撑是安全着陆的关键特征。  
  usable_signals: [obs[6], obs[7]]  
  risks: 可能诱导故意摔下以获得接触，需要结合位置与速度约束。

- role_id: attitude_stabilization  
  purpose: 保持机体朝向接近竖直，防止过大倾角导致侧翻。  
  why_required: 着陆稳定和引擎安全要求姿态可控。  
  usable_signals: [obs[4], obs[5]]  
  risks: 过度严厉的姿态奖励可能妨碍姿态调节机动。

- role_id: fuel_efficiency  
  purpose: 惩罚不必要的引擎使用，节省能耗。  
  why_required: 次目标明确要求“as little engine thrust as possible”，是显著优化方向。  
  usable_signals: [action]  
  risks: 权重过高会阻碍跨越距离，或导致长时间滑翔不点火。需与 proximity 奖励平衡。

### 10.2 条件职责 conditional_roles
- role_id: fast_settling_incentive  
  condition_to_use: 当智能体已能较稳定到达目标垫后加入，或训练中后期使用，加速收敛时间。  
  usable_signals: 随时间步隐含（通过奖励标量衰减），或结合距离门控的小幅每步惩罚。  
  risks: 过早加入会使初期探索因持续惩罚而失败；需要配合目标到达奖励，避免鼓励草率着陆。

### 10.3 慎用/禁用职责 avoid_roles
- 无特别禁用的角色，但应避免使用任何依赖 info 或官方奖励的角色，因为相关信号不可用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | obs[0], obs[1] | 无 | distance_penalty, potential-based shaping | 通常用负欧几里得距离或其平方 |
| velocity_control | obs[2], obs[3] (门控用 obs[0], obs[1]) | 无 | squared penalty, bounded_signal | 靠近目标时加重对速度的惩罚 |
| stable_contact | obs[6], obs[7] | 无 | product (both_contact=left*right), binary bonus | 可配合速度接近零时给予正奖励 |
| attitude_stabilization | obs[4], obs[5] | 无 | angle_penalty, quadratic_penalty | 以角度的平方或绝对值惩罚 |
| fuel_efficiency | action (0~3) | 无 | action_cost_vector (0 cost for no_engine, positive for others) | 主引擎和姿态引擎可设置不同代价 |
| fast_settling_incentive | 隐含 (step counter) | 显式时间信号 | small_constant_penalty | 用极小负数每步施加，需谨慎调权 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 无法接近目标，远距离徘徊 | 平均 episode 末端 |x|,|y| 仍然很大，轨迹发散 | 加强 goal_proximity 权重，加入电位塑形 |
| 高速撞击目标垫 | 着陆瞬间 |vy| 大，接触后跳起或翻倒 | 提升 velocity_control 权重，并在接近时启用强力减速惩罚 |
| 单腿着地后翻倒 | left*right 很少同时为1，结束前 body_angle 剧烈变化 | 增加双接触奖励，并严格惩罚大 angular velocity |
| 无明显理由的越界或碰撞 | 轨迹在视口边缘频繁 crash | 检查 initial force 随机扰动过大，或 guidance 奖励不足导致偏航；可引入轻微居中引导 |
| 过量使用引擎，燃料耗尽仍无法到达 | 动作统计中 0 号动作使用极少，总步数短 | 提升 fuel_efficiency 惩罚，同时确保 goal_proximity 能引导高效路径 |
| 过早沉降（稳定在远离目标处） | episode 因 settled 终止，但 |x| 或 |y| 很大 | 将 settled 终止视为无效，需配合位置条件；强化到达奖励 |



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
- best_score_so_far: -103.790

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_penalty + landing_reward + progress_reward | 2 | -103.790 | -103.790 | unsolved |
| action_penalty + contact_reward + descent_reward + progress_reward | 1 | -120.100 | -120.100 | unsolved |
| contact_reward + orientation_penalty + progress_reward + vel_damping | 4 | -124.550 | -159.200 | unsolved |
| contact_quality + landing_gate_penalty + progress | 2 | -477.400 | -477.400 | unsolved |

## Previous interventions

- iter 2 (score=-168.020, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：L2: 这里不是单纯的权重问题，vel_damping 的数学条件需要修改才能从“死组件”变为有效信号。因此选择 L2 修改其公式形态。 | selected_intervention: 目标组件：`vel_damping`（将其重定义为“接近地面时的速度惩罚”）
- iter 3 (score=-124.550, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：**L2**（改动数学形态） | selected_intervention**：
- iter 6 (score=-103.790, structure=action_penalty + landing_reward + progress_reward): 历史记录中（通过 `read_past_reflection` 和 `read_reward_code` 检索）**未出现**“将 landing 奖励改为连续的高度‑速度‑姿态质量函数”的尝试，因此本轮修改不是已有语义的重复。 | selected_level: L2** — 改变 landing_reward 的数学形态，从离散双腿接触奖励变为连续的高度/速度/姿态质量函数。历史中未做过此变换，且当前分数离目标（200）差距远超 10%，L1 微调不足以解决稀疏性问题。 | selected_intervention: 将 `landing_reward` 改写为：
- iter 7 (score=-120.100, structure=action_penalty + contact_reward + descent_reward + progress_reward): selected_level**：**L2: 将 `landing_reward` 从“连续状态质量奖励”改为“下降过程奖励 + 接触事件奖励”，从根本上消除静态悬停可利用点。同步删除 `progress_reward` 的健康门控使其重新生效。 | selected_intervention**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
