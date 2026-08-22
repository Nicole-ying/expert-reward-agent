# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主体为一个2D飞行器（vehicle-like），初始位置在视口顶部中央附近，带有随机初始作用力。  
核心目标是 **尽快到达并稳定停靠在中央目标着陆平台上**，同时尽可能少使用引擎推力。  
智能体需要学习：趋近目标、降低线速度、保持姿态竖直、实现两腿同时安全接触（软着陆）。  
不应混淆的目标：单纯求快但忽略安全着陆，或一味省燃料而无法到达目标。  

## 2. 任务类型选择
**selected_route_id**: `navigation_goal_reaching`  
**confidence**: high  
**reason**: 任务核心是到达指定目标位置（中央着陆平台）并完成稳定停靠，符合“导航目标到达”族定义；附属的省燃料、动作小等属于次要优化目标，不构成多目标冲突的主体。  

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: [8]  
- **dtype**: 默认为 float64 或 float32（取决于环境实现，但通常为 float64）  
- **obs[0]**: `x_position`，水平方向相对于目标着陆平台中心的偏移量，可用于奖励趋近目标，reward_usable: true  
- **obs[1]**: `y_position`，垂直方向相对于平台高度（接触面）的偏移量，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，reward_usable: true  
- **obs[4]**: `body_angle`，机体倾角（如弧度），reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，reward_usable: true  
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true  

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **动作/索引 0**: `no_engine` (不做任何事)，语义：无推力，用于滑行或停靠后保持  
- **动作/索引 1**: `left_orientation_engine` (左姿态引擎)，语义：产生逆时针或顺时针旋转力矩（具体方向取决于环境）  
- **动作/索引 2**: `main_engine` (主引擎)，语义：产生纵向（向上）推力  
- **动作/索引 3**: `right_orientation_engine` (右姿态引擎)，语义：产生与左引擎相反的旋转力矩  

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 可能表示机体静止稳定，通常意味着已着陆并静止，可能视为成功；  
- **failure-like termination**: `crash_or_body_contact` （如机身碰撞地面或平台以外部分）、`horizontal_position_outside_viewport` （漂出水平边界）很可能表示失败；  
- **ambiguous termination**: `crash_or_body_contact` 若接触平台但判定为 crash 则为失败，但描述未区分成功接触与失败接触；我们需要从“到达并稳定停靠”推断理想行为是两腿接触且低速，但不能直接从终止信号中获知成功。  
- **truncation**: 源代码未显示截断（max_steps），但多数环境有步数限制，此处未给出，视为不存在或不可直接用于奖励。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空字典 `{}`，无可用字段 。  
- **forbidden_or_uncertain_info_fields**: 任何 info 字段均不可用（因为提供的信息为空）。终止原因也不能从 info 获取。  

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
**允许使用**：  
- `obs`（当前观测）  
- `action`（当前动作）  
- `next_obs`（下一帧观测）  
- `info` 中明确允许的字段（实际为空，等于不可用）  
- `training_progress` 仅在 prompt 明确允许使用时可用，此处未允许，应禁止使用。  

**禁止使用**：  
- `original_reward`（被屏蔽的官方奖励）  
- 未声明的 `info` 字段（所有字段）  
- 未声明的 `obs` 切片（即不得依赖未记录的额外观测）  

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（可直接组成距离度量）  
- **velocity**: `x_velocity`, `y_velocity`（线速度）  
- **orientation**: `body_angle`, `angular_velocity`（角速度）  
- **contact**: `left_support_contact`, `right_support_contact`（双腿触地标志），可构造着陆条件  
- **action/engine**: 可以基于动作选择构建惩罚（如鼓励 no_engine，惩罚主引擎和姿态引擎）  
- **other**: 无  

## 8. 不确定或不可用的信号
- **官方奖励/任务终止标志**：被屏蔽，不可用  
- **绝对时间/步数**：无直接可用信号（training_progress 不允许使用）  
- **成功标志**：info 中无 success 或 failure 字段，不可用  
- **视口边界信息**：观测中无边界值，只能从位置推测越界但越界后 episode 已终止，故不能在奖励中直接使用边界信号  
- **风或其他扰动**：被省略，不可用  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D flying vehicle with two legs
  actuator_type: one main thruster (vertical), two orientation thrusters (torque)
  contact_structure: two-point support (left/right legs)
primary_objectives:
  - reach the central target landing pad (zero x,y relative position)
  - achieve stable contact with both legs, near-zero linear and angular velocity
  - keep body angle upright (close to 0)
secondary_objectives:
  - minimize total engine thrust usage (actuator effort)
  - arrive as fast as possible (implicit through fast approach)
main_failure_risks:
  - crash: body contacts ground or obstacle outside target pad
  - horizontal drift out of viewport
  - landing with single leg contact or high velocity, leading to instability
  - overshoot or oscillation around target, causing unnecessary fuel waste
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `approach_goal`  
  **purpose**: 鼓励向目标着陆平台靠近，减少位置误差。  
  **why_required**: 前往目标是任务核心，距离奖是引导基础。  
  **usable_signals**: `x_position`, `y_position` （下一帧或当前帧）。  
  **risks**: 若权重过高，可能忽视软着陆条件，导致高速撞击；需要与速度/姿态惩罚协调。  

- **role_id**: `soft_landing_condition`  
  **purpose**: 检测着陆成功特征（双腿接触、低线速度、低角速度、竖直姿态），给予正向激励。  
  **why_required**: 任务要求稳定停靠，单纯到达不够，必须形成软着陆。  
  **usable_signals**: `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity`, `angular_velocity`, `body_angle`。  
  **risks**: 条件设定过于严苛会导致延迟奖励，可能需配合稠密分量。  

- **role_id**: `stability_penalty`  
  **purpose**: 惩罚非竖直姿态和大角速度，鼓励飞行平稳。  
  **why_required**: 防止剧烈翻滚，便于着陆控制。  
  **usable_signals**: `body_angle`, `angular_velocity`。  
  **risks**: 与着陆时的必要姿态调整冲突；可考虑在接近目标后才加强。  

### 10.2 条件职责 conditional_roles
- **role_id**: `thrust_penalty`  
  **condition_to_use**: 当 agent 接近目标且速度已降至较低水平时逐渐启用，或在全程作为温和正则项。  
  **usable_signals**: `action`（是否使用主引擎或姿态引擎）。  
  **risks**: 过早惩罚会抑制探索；忽略则燃料浪费严重。建议随逼近程度动态调节权重。  

- **role_id**: `velocity_smoothing`  
  **condition_to_use**: 当需要更稳定轨迹时可加入，但不是必须。  
  **usable_signals**: `x_velocity`, `y_velocity` 的突变（需要历史信息，若无法获得则不可用）。  
  **risks**: 需要存储上一帧速度，增加复杂度；当前环境未提供历史，可能难以直接实现。  

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `time_or_step_penalty`  
  **reason**: 无可用的步数或时间信号；`training_progress` 不允许使用，真实步数未暴露。  
  **forbidden_or_missing_signals**: 步数计数器。  

- **role_id**: `original_reward_mimic`  
  **reason**: 官方奖励被屏蔽，严禁尝试复现。  

- **role_id**: `safe_zone_boundary`  
  **reason**: 没有视口边界坐标，超出边界即终止，无法在 episode 内构建渐进惩罚。  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `approach_goal` | `x_position`, `y_position` (from next_obs or obs) | — | `dense_state_signal` (e.g. -dist), `bounded_signal` | 可结合当前位置与下一帧位置计算趋近量 |
| `soft_landing_condition` | `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity` (all from next_obs) | — | `gated_reward` (if all conditions met, large bonus) | 所有特征可从 next_obs 直接读取 |
| `stability_penalty` | `body_angle`, `angular_velocity` | — | `quadratic_penalty`, `abs_penalty` | 可直接加负平方或负绝对值 |
| `thrust_penalty` | `action` (Discrete index) | — | `action_cost` (e.g. penalty for action 1,2,3) | 动态权重可随与目标距离、速度调节 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 策略过早触发“crash”终止而无法获得正奖励 | 回报曲线持平且 episode 长度很短；多数终止时双腿未均接触 | 加强距离引导，增加 soft_landing 稀疏奖励，或减免早期动作惩罚 |
| 漂出水平边界 | 最终 x_position 数值超过视口边界（可从 episode 结束时观察估计） | 在接近边界附近增加基于横向位置的惩罚（结合 speed） |
| 长时间悬停而不着陆或燃料耗尽 | 回报不高，episode 步数很长且双腿未接触或接触不稳定 | 增加时间（步数）隐式压力，通过缓慢衰减奖励或加大 thrust_penalty |
| 着陆时只有单腿接触或倾斜 | 结束时 `left_support_contact` 与 `right_support_contact` 不一致，或 `body_angle` 较大 | 强化双腿同时接触的奖励条件，并加大 `body_angle` 惩罚 |
| 过度使用主引擎，省燃料目标未达成 | 动作统计中动作 2 使用频率高 | 适度调高 thrust_penalty 的权重，尤其是在低速靠近目标时 |
| 角速度过大导致旋转漂移 | 观察 `angular_velocity` 直方图 | 加强 angular_velocity 惩罚，或增加 anti‑spin



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





# Validation repair

具体错误：Reward v1 failed validation: 出现禁止模式: import ; 修复要求：删除所有import和numpy调用；平方根使用**0.5，有界函数使用max/min或不需要库的代数表达式 (record: runs/env_001/rpv4_old_baseline/seed_1/iter_01/generation/validations/reward_v1.validation.json)

只修复代码合规问题，不重新分析环境，不改变原定奖励设计。直接输出完整合规的compute_reward函数。

# Invalid previous draft

# reward_v1.py

```python
import math

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar‑lander‑style environment.
    Task family: navigation_goal_reaching, dynamics_subtype: goal_approach_and_soft_contact.
    """
    # ---- unpack observations ----
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    x_vel_new, y_vel_new = next_obs[2], next_obs[3]
    body_angle_new = next_obs[4]
    ang_vel_new = next_obs[5]
    left_contact_new = next_obs[6]
    right_contact_new = next_obs[7]

    # ---- 1. Approach the goal (improvement_delta) ----
    dist_old = math.sqrt(x_old**2 + y_old**2)
    dist_new = math.sqrt(x_new**2 + y_new**2)
    approach_reward = 100.0 * (dist_old - dist_new)   # positive when getting closer

    # ---- 2. Stability penalty (quadratic_penalty) ----
    w_angle = 5.0
    w_angvel = 0.5
    stability_penalty = -w_angle * body_angle_new**2 - w_angvel * ang_vel_new**2

    # ---- 3. Thrust efficiency (discrete action cost) ----
    w_thrust = 0.03
    thrust_cost = -w_thrust if action != 0 else 0.0    # action 0 = no engine

    # ---- 4. Soft‑landing quality (joint_condition_proxy) ----
    contact_both = float(left_contact_new > 0.5 and right_contact_new > 0.5)
    if contact_both > 0.5:
        # how “soft” the landing is
        vel_sum = abs(x_vel_new) + abs(y_vel_new)
        vel_factor = 1.0 / (1.0 + 10.0 * vel_sum)
        ang_factor = 1.0 / (1.0 + 5.0 * abs(ang_vel_new))
        angle_factor = 1.0 / (1.0 + 5.0 * abs(body_angle_new))
        landing_quality = vel_factor * ang_factor * angle_factor
        landing_bonus = 200.0 * landing_quality
    else:
        landing_bonus = 0.0

    # ---- assemble ----
    total_reward = approach_reward + stability_penalty + thrust_cost + landing_bonus
    components = {
        'approach_reward': approach_reward,
        'stability_penalty': stability_penalty,
        'thrust_cost': thrust_cost,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与职责选择
- **task_family**：`navigation_goal_reaching`  
- **dynamics_subtype**：`goal_approach_and_soft_contact`  
- **selected reward roles**（从 `reward_role_decomposition` 中选取）：
  1. `approach_goal` – 主学习信号（mandatory）
  2. `soft_landing_condition` – 任务完成近似信号（mandatory）
  3. `stability_penalty` – 稳定/安全约束（mandatory）
  4. `thrust_penalty` – 效率/动作代价（conditional，v1 以极小权重加入）

## role‑to‑signal 映射与公式算子
| role | 使用的信号（next_obs） | formula operator | 说明 |
|------|------------------------|------------------|------|
| `approach_goal` | `x_position`, `y_position` | `improvement_delta` | 奖励每一步距离的缩减量，连续且稠密 |
| `soft_landing_condition` | `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity` | `joint_condition_proxy` | 仅在双腿同时接触时激活，用多个连续因子的乘积衡量着陆质量 |
| `stability_penalty` | `body_angle`, `angular_velocity` | `quadratic_penalty` | 轻量抑制机体倾角和旋转，避免剧烈翻滚 |
| `thrust_penalty` | `action`（离散索引） | 动作代价（常数惩罚） | 非零动作施加极小负值，鼓励引擎空闲 |

## 排除的 roles 及原因
- `velocity_smoothing` – 需要历史速度信息，当前接口无此能力，不可用。
- `time_or_step_penalty` – 无可用的步数或时间信号，且 `training_progress` 不允许使用。
- `original_reward_mimic` – 官方奖励被屏蔽，明确禁止。
- `safe_zone_boundary` – 视口边界坐标未暴露，无法在 episode 内构建渐进惩罚。
- **terminal_success_reward / terminal_failure_penalty** – `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 `false`，且环境 `info` 为空，无法从终止原因中获得差异化信号。

## 为何未使用 terminal 奖励
环境不提供显式的成功/失败 flag，也不能从 `info` 推断 episode 结束原因。直接硬编码终端事件奖励风险极大（可能把失败当作成功奖励，或反之）。因此 v1 完全依赖 per‑step 信号：`approach_reward` + `stability_penalty` 驱动趋近与稳定，`landing_bonus` 在接触双腿且各条件满足时提供强正反馈。成功着陆的 episode 会因为 `landing_bonus` 获得高回报，失败的 episode 则缺少此项且通常累积较多负向惩罚，这种差异足以引导学习。

## 后续迭代方向
- 若 agent 频繁在成功前 crash，可考虑加入 **soft_health_gate**，在姿态/速度接近危险阈值时衰减主奖励。
- 若燃料消耗仍然过高，可将 `thrust_cost` 改为随目标接近度动态增强的权重。
- 若出现单腿着陆或倾斜着陆，可微调 `landing_quality` 的因子权重，或为 `body_angle` 增加更严的 hinge 惩罚。
- 若需要更强的时间压力，将来可在允许使用 `training_progress` 的迭代中加入 **curriculum‑based decay**。

## 训练后应重点观察的 failure modes
- **高速撞击平台**：`approach_reward` 过度激励快速下降，检查速度直方图，必要时动用 velocity hinge 或增大 `stability_penalty`。
- **水平漂移出界**：`approach_reward` 的横向分量不足（取决于横纵尺度的差异），可通过分开横纵 weight 缓解。
- **悬停/徘徊**：episode 很长且回报停滞，可能 `approach_reward` 的系数过小或 `thrust_cost` 略大；可提升 `w_approach` 或降低 `w_thrust`。
- **单腿接触倾斜着陆**：检查 `body_angle` 和 `contact_both` 的末端统计，强化 `body_angle` 惩罚或调整 `landing_quality` 中 angle 因子斜率。
- **动作统计**：若姿态引擎使用率过高而实际上无需旋转，则 `stability_penalty` 可能偏弱，可适当加大 `w_angvel`。