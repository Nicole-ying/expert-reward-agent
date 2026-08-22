# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
一个 2D 飞行器从画面上方中心附近出发，受初始随机力作用。核心目标是**到达并稳定在中心目标平台上**（即精准着陆并停稳）。任务同时要求**尽可能快**地完成着陆，并且**尽可能少使用发动机推力**。在过程中，智能体需要学会接近目标、降低速度、保持姿态稳定并实现安全接触（避免硬撞击）。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 核心目标是到达并稳定在目标平台上，加快速度与节省推力属于附属优化，不存在多目标之间不可调和的主次冲突。动力学子类型符合 **goal_approach_and_soft_contact**，因为需要减速、稳定姿态并安全触地。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: float32（推测）  
- **obs[0]**: `x_position`，水平坐标相对于目标 pad，可用于衡量横向距离，reward_usable: true  
- **obs[1]**: `y_position`，垂直坐标相对于 pad 高度，可用于衡量高度，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，用于速度控制，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，用于软着陆控制，reward_usable: true  
- **obs[4]**: `body_angle`，机体角度，用于姿态稳定性，reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，用于姿态稳定，reward_usable: true  
- **obs[6]**: `left_support_contact`，左侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true  
- **obs[7]**: `right_support_contact`，右侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true  

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **action 0**: `no_engine`，无发动机动作（什么都不做）  
- **action 1**: `left_orientation_engine`，左姿态发动机，提供侧向推力或扭矩  
- **action 2**: `main_engine`，主发动机，提供向上推力  
- **action 3**: `right_orientation_engine`，右姿态发动机，提供另一侧推力或扭矩  

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 可能表示机体不再活跃或已稳定（例如成功着陆后静止），但也可能包含坠落静止等失败情况。**无显式成功标志**。  
- **failure-like termination**:  
  - `crash_or_body_contact` – 机体与地面或障碍物碰撞导致的失败。  
  - `horizontal_position_outside_viewport` – 水平位置超出画面边界，视为失败。  
- **ambiguous termination**: 上述 settled 条件含义模糊，其设计意图是探测着陆稳定，但不能直接作为可靠的成功指标。  
- **truncation**: 未提供任何时间步或步数限制的截断信号。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
- `explicit_failure_flag_available`: **false**  
- `allowed_info_fields`: 环境返回空字典，无可用的额外信号。  
- `forbidden_or_uncertain_info_fields`: 所有未声明的 info 字段均不可用，且不应假设存在 `success`、`failure` 或 `termination_reason`。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs` / `next_obs`（当前和下一步观察）
- `action`（当前动作）
- `info` 中明确允许的字段（当前为空，因此不可用）
- `training_progress` 仅在 prompt 明确允许时使用，当前未允许，故禁止使用

**禁止使用**：
- `original_reward`（官方奖励已被屏蔽）
- 任何未声明的 `info` 字段
- 任何未声明的观测切片（如假设观测包含其他维度）

## 7. 可用于奖励函数的信号
- **position**: `obs[0]`（x_position）、`obs[1]`（y_position），可计算到目标 pad 的距离；当两者趋近于 0 时表示到达。  
- **velocity**: `obs[2]`（x_velocity）、`obs[3]`（y_velocity），可用于惩罚或塑造速度，尤其在接近目标时。  
- **orientation**: `obs[4]`（body_angle）、`obs[5]`（angular_velocity），用于保持竖直与姿态稳定。  
- **contact**: `obs[6]`（left_support_contact）、`obs[7]`（right_support_contact），指示是否已接触环境，用于判断着陆状态与安全性。  
- **action/engine**: `action` 取值 1/2/3 代表发动机点火，可计算燃料消耗代价。  
- **other**: 无其他明确可用的信号。

## 8. 不确定或不可用的信号
- 显式的成功/失败标志：无。  
- 目标 pad 的绝对位置或 viewport 边界：未提供，只能通过相对位置推断。  
- 燃料剩余量：没有直接观测值，只能用动作序列间接估计。  
- 碰撞类型或 crash 细节：终止时无额外信息。  
- 安全垂直速度阈值：未给出，需凭经验设定。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid-body vehicle with landing legs
  actuator_type: two orientation thrusters and one main thruster
  contact_structure: binary leg contacts (left/right)
primary_objectives:
  - reach the central target pad and settle stably
secondary_objectives:
  - minimize time to reach (implied by “as fast as possible”)
  - minimize engine thrust usage
  - maintain stable orientation during descent
main_failure_risks:
  - crashing into ground or obstacles at high velocity
  - exceeding horizontal boundaries
  - failing to stabilize orientation, leading to tilting and instability
  - excessive fuel consumption leading to inability to land softly
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  **purpose**: 鼓励向目标 pad 移动，最终到达目标位置（x→0, y→0）。  
  **why_required**: 任务核心是到达，距离减小是主要驱动信号。  
  **usable_signals**: [obs[0], obs[1]]  
  **risks**: 仅追求近距离可能鼓励高速俯冲导致 crash，需配合速度与接触约束。

- **role_id**: `velocity_damping`  
  **purpose**: 在接近目标时抑制速度，实现软着陆。  
  **why_required**: 到达目标时必须几乎零速度才能安全稳定，防止硬碰撞。  
  **usable_signals**: [obs[2], obs[3], obs[0], obs[1]]（可利用距离调制惩罚强度）  
  **risks**: 过强速度惩罚阻碍必要机动，推荐随距离衰减。

- **role_id**: `fuel_efficiency`  
  **purpose**: 惩罚使用任何发动机推力，鼓励节省燃料。  
  **why_required**: 任务明确要求“using as little engine thrust as possible”。  
  **usable_signals**: [action == 1, action == 2, action == 3]  
  **risks**: 过度节省可能导致无法减速或姿态失控，需平衡权重。

- **role_id**: `safe_contact`  
  **purpose**: 在触地时确保垂直速度很小，并且双足同时触地更稳定。  
  **why_required**: 任务要求“make safe contact”，避免硬着陆 crash。  
  **usable_signals**: [obs[6], obs[7], obs[3], obs[1]]（接触标志 + y_velocity，可结合高度）  
  **risks**: 若在非目标区域过早激活，可能误导智能体提前接触错误表面，应仅在接近目标高度时启用。

- **role_id**: `orientation_stability`  
  **purpose**: 保持机体竖直，减小旋转。  
  **why_required**: 描述中要求“keep a stable orientation”，且姿态稳定有助于安全接触。  
  **usable_signals**: [obs[4], obs[5]]  
  **risks**: 与机动性有轻微冲突，但权重较低可接受。

### 10.2 条件职责 conditional_roles
- **role_id**: `fast_progress_encouragement`  
  **condition_to_use**: 早期训练可能进展缓慢，可在智能体向目标移动时给予额外正奖励，避免摸鱼。  
  **usable_signals**: [obs[0], obs[1]] 的变化量（通过 next_obs 差分）  
  **risks**: 可能加剧 crash 风险，需与其他安全职责一起使用。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `boundary_penalty`  
  **reason**: 环境已通过终止条件处理越界，边界绝对坐标未知，额外惩罚可能干扰探索。  
  **forbidden_or_missing_signals**: 边界坐标。

- **role_id**: `explicit_success_reward`  
  **reason**: 无明确的 success flag 可用。

- **role_id**: `time_penalty_per_step`  
  **reason**: 每步直接惩罚时间可能与 fuel_efficiency 高度耦合，且过于粗暴。更推荐通过 progress shaping 实现快速接近。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `goal_proximity` | x_position, y_position | target distance metric | quadratic_penalty, l2_norm | 可用 -sqrt(x²+y²) 或 -(x²+y²) |
| `velocity_damping` | x_velocity, y_velocity, (distance proxy) | desired zero velocity | weighted_squared_penalty, proximity_gated | 速度惩罚乘以随距离衰减的系数，靠近目标时加重 |
| `fuel_efficiency` | action index (1,2,3) | fuel level | action_cost (constant penalty per activation) | 每次点火给予固定负奖励 |
| `safe_contact` | left_contact, right_contact, y_velocity, (y_position) | safe velocity threshold | conditional_penalty, indicator * penalty | 当接触标志为真且高度接近 0 时，惩罚 y_velocity² |
| `orientation_stability` | body_angle, angular_velocity | desired angle=0 | quadratic_penalty | angle² + w·angular_velocity² |
| `fast_progress` (conditional) | (x_position, y_position) diff | – | positive_shaping_on_negative_delta | dist_prev – dist_curr > 0 时给予正奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞地导致早期 crash | 奖励不收敛，episode 极短，频繁 `crash_or_body_contact` | 增加 `velocity_damping` 和 `safe_contact` 权重，强化姿态稳定 |
| 飘出水平边界 | 因越界终止，累计奖励低 | 谨慎增加 soft boundary 惩罚或加强姿态/速度引导 |
| 过度悬停或长时间不下降 | episode 长但距离不减小，燃料惩罚高 | 引入 `fast_progress` 奖励，适当放宽燃料惩罚以允许探索 |
| 单脚着陆导致倾斜失败 | 最终仅一侧接触，body_angle 大 | 在 `safe_contact` 中奖励双接触，对单脚接触施加小负奖励 |
| 姿态剧烈振荡 | angular_velocity 长期大，body_angle 波动 | 提高 `orientation_stability` 权重，或惩罚连续来回点火 |
| 早期耗尽燃料，后期无法减速 | 前期大量使用 main_engine，后期 crash | 加强燃料惩罚，或调整探索噪声鼓励少用主发动机 |
| 无法稳定静止，反复弹跳 | 目标附近仍持续移动 | 在 Y 接近 0 且接触时大幅强化 `velocity_damping`，可配合接触锁死奖励 |

通过观测上述信号调整各职责权重，可在不依赖真实 success flag 的条件下逐步提升智能体性能。



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

