# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 刚体飞行/着陆任务。一个身体从视口顶部中心附近被随机初始冲量射出，目标是以**最快的速度**到达并稳定留在**中心目标平台**上，同时**尽可能少用引擎推力**。智能体必须学会接近目标、降低速度、保持合适身体姿态、最终让双脚安全接触平台并静止。

**主任务**：导航到目标位置并稳定着陆（到达 + 软接触）。  
**次任务**：减少燃料消耗（少用推进器）、缩短飞行时间。  
**不应混淆的目标**：单纯追求高速撞击、持续在空中徘徊、仅保持姿态不移动。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: high  
reason: 核心目标是到达并稳定在一个空间固定目标点（平台）上，具有明确的到达与接触完成条件。附属优化（省燃料、求快）不构成权重冲突的第二个独立目标，因此属于导航目标到达族。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (假设)
- obs[0]: `x_position` 相对目标平台的水平坐标，reward_usable: true（可用于距离/误差信号）
- obs[1]: `y_position` 相对平台高度的垂直坐标（上方为正），reward_usable: true（可用于高度/加速度控制）
- obs[2]: `x_velocity` 水平线速度，reward_usable: true（速度惩罚、减速信号）
- obs[3]: `y_velocity` 垂直线速度，reward_usable: true（下降速度控制、接近软着陆）
- obs[4]: `body_angle` 身体绝对倾斜角，reward_usable: true（姿态稳定，保持直立）
- obs[5]: `angular_velocity` 角速度，reward_usable: true（姿态剧烈变化惩罚）
- obs[6]: `left_support_contact` 左支撑接触标志（0/1），reward_usable: true（接触奖励、着陆成功条件）
- obs[7]: `right_support_contact` 右支撑接触标志（0/1），reward_usable: true（同上）

所有观测均可直接用于奖励设计。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不点火，依靠惯性运动
- action 1: `left_orientation_engine` — 点燃左侧姿态发动机，产生旋转力矩（可能是逆时针？）
- action 2: `main_engine` — 点燃主发动机，产生指向主要方向的推力（通常向下反推）
- action 3: `right_orientation_engine` — 点燃右侧姿态发动机，产生相反旋转力矩

连续控制维度：无，纯离散。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 source 中 `terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`：
- `crash_or_body_contact`：身体主体与地面或障碍物碰撞（非预期的强接触），典型失败终止。
- `horizontal_position_outside_viewport`：水平位置超出屏幕边界，失败终止。
- `body_not_awake_or_settled`：身体进入休眠或已稳定（速度/角速度极低，接触平台）。可能代表成功着陆，但也可能出现在空中或远离目标提前停止（如游戏超时或因某种冻结），因此是**模糊终止**，必须结合位置和接触条件判断。
- 没有显式的 `truncation` 标志，环境未提供截断信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无 `"success"` 字段）
- explicit_failure_flag_available: false （同上）
- allowed_info_fields: `{}` （无可用字段）
- forbidden_or_uncertain_info_fields: `{"success": "缺失", "failure": "缺失", "termination_reason": "缺失"}`
- 因此所有成功/失败判定必须从观测中自行组合逻辑，例如：
  - 成功候选：`body_not_awake_or_settled` 且 双脚接触 `(obs[6]==1 and obs[7]==1)` 且 相对位置在允许范围内。
  - 失败候选：`crash_or_body_contact` 或 `horizontal_position_outside_viewport`，或错误的静止情况。

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` 当前观测 (shape [8])，全部维度可用
- `action` 当前动作 (0..3)
- `next_obs` 下一观测
- `info` 中目前为空字典，不可依赖
- `training_progress` 除非 prompt 明确要求，否则**不应使用**

禁止使用：
- `original_reward` 或任何形式的官方奖励
- 未声明的 `info` 字段
- 未声明的观测切片外的值

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position` — 距目标平台误差、高度控制
- **velocity**: `x_velocity`, `y_velocity` — 速度量级、下降速率
- **orientation**: `body_angle`, `angular_velocity` — 姿态稳定
- **contact**: `left_support_contact`, `right_support_contact` — 着陆/接触状态
- **action/engine**: 动作选择可用于燃料消耗惩罚（如 `action != 0` 为使用引擎）
- **other**: 无额外信号

## 8. 不确定或不可用的信号
- 无明确的“得分”或“完成”指示器
- 环境不提供平台范围或具体接触法向量
- 没有外部定位系统（如 landmark）
- 没有引擎推力大小或燃料剩余量（仅知动作是否使用引擎）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with bilateral supports (two legs/pads)
  actuator_type: discrete thrusters (main engine + two orientation engines)
  contact_structure: two-point contact (left & right support feet)
primary_objectives:
  - Reach the target platform (minimize relative horizontal/vertical position)
  - Achieve stable, safe contact (both support contacts active, low velocity, upright attitude)
secondary_objectives:
  - Minimize fuel usage (reduce engine actuation count/intensity)
  - Complete the task quickly (implicit via episode length, not explicit reward required)
main_failure_risks:
  - Crashing into ground/obstacle due to excessive downward speed or bad orientation
  - Drifting outside allowed horizontal range
  - Premature "settled" state far from target or without proper contact, causing early termination
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `progress_toward_target`  
  purpose: 鼓励智能体沿最短路径向目标平台移动，缩短位置误差  
  why_required: 任务是到达目标，无进度信号则无引导  
  usable_signals: `x_position`, `y_position`（相对误差），`next_obs` 的相同维度  
  risks: 若只给正奖励，可能诱导高速冲撞目标，需配合减速职责

- **role_id**: `soft_landing_and_contact`  
  purpose: 奖励在接近目标且满足接触条件时以低速、小姿态着“陆”，确保稳定停靠  
  why_required: 仅到达不足以成功，需要双脚接触且无反弹/侧翻  
  usable_signals: `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`, `left_support_contact`, `right_support_contact`, `x_position`, `y_position`  
  risks: 过于严格的减速要求可能使学习前期难获得奖励，需分区间或结合课程

- **role_id**: `crash_and_boundary_penalty`  
  purpose: 阻止智能体因失控导致提前失败终止（碰撞、出界）  
  why_required: 终止事件会结束回合并切断学习信号，必须明确惩罚  
  usable_signals: 由终止模式触发（但无显式标志），可通过对 `next_obs` 状态推测（如速度突变、位置飞速变化）或在 episode 结束处基于 `terminated` 与位置/速度判断，但准确度受限。建议设计时用**临界值警告**：对过大的垂直速度、水平位置超阈提供密集惩罚。  
  risks: 在缺失显式失败标志时可能不精确，但密集惩罚可弥补。

### 10.2 条件职责 conditional_roles
- **role_id**: `fuel_efficiency`  
  condition_to_use: 在整个 episode 中启用，但权重不宜过高，避免智能体选择完全不动作。  
  usable_signals: `action`（通过 action != 0 判断用引擎），可加和并除以步数或直接惩罚每次点火  
  risks: 强惩罚可能导致智能体永远不点燃主引擎而无法到达目标，需与进度职责平衡；初始学习阶段可减轻权重。

- **role_id**: `orientation_stability`  
  condition_to_use: 全程使用，但严格倾斜角度限制可在接近目标时加强。  
  usable_signals: `body_angle`, `angular_velocity`  
  risks: 在远处过度要求零倾角会限制机动，可用宽松规范 + 靠近目标时收紧。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `velocity_maximization`  
  reason: 任务需要快速到达，但简单的速度最大化会鼓励冲撞和不稳定行为；速度应作为“在安全范围内的适当快速”而不是目标。该职责与软着陆冲突，禁用。  
  forbidden_or_missing_signals: 缺乏显式的“安全速度上限”，需从接触动能反向推导，不可直接设高速度奖励。

- **role_id**: `direct_success_flag_reward`  
  reason: 无显式 `success` 标志，不能以该 flag 发放大型奖励。  
  forbidden_or_missing_signals: `info["success"]` 不存在。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| progress_toward_target | `x_position`, `y_position` 及其变化量 | 无 | `distance_to_goal`, `negative_l2_norm`, `delta_distance` | 越靠近越好，可用连续负值惩罚距离 |
| soft_landing_and_contact | 速度、角度、角速度、双脚接触、与目标的接近度 | 无 | `bounded_signal` (速度阈值), `indicator` (接触), `quadratic_penalty` (倾角) | 接近目标且双脚接触时给予大正奖励，否则严格惩罚高速、大倾角 |
| crash_and_boundary_penalty | 推测的越界条件（位置、速度）或 episode 结束判断 | 显式 fail flag | `threshold_indicator` (|x|>max_x, vy < -danger_limit) | 当成密集约束使用，比如当 y_vel 过负时持续惩罚 |
| fuel_efficiency | action (0=no_engine) | 剩余燃料量 | `episode_action_count`, `per_step_penalty` if action!=0 | 轻柔惩罚，以免抑制必要动作 |
| orientation_stability | `body_angle`, `angular_velocity` | 无 | `quadratic_cost` (angle), `abs_penalty` (angvel) | 后期可随接近目标增加权重 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 盘旋不下降、过度节约燃料 | 平均 episode 长度很长，y_position 初始后几乎不变，不落地 | 增加对高度维持的惩罚（鼓励减少 y_pos），或增加进度奖励中高度的权重 |
| 高速垂直冲击导致 crash | y_velocity 终末巨大负值，随即 episode 结束（crash） | 引入下降速度上限的密集惩罚，并对 soft_landing 奖励放松距离条件，先奖轻微接触 |
| 姿态不稳翻滚出界 | angular_velocity 持续高值，body_angle 大幅振荡，x_position 很快越界 | 加强姿态稳定惩罚，尤其是接近目标时；也可添加对 angular_velocity 的 L1/L2 惩罚 |
| 仅一脚接触便静止（body_not_awake 但只有一侧接触） | 一条腿接触，另一条腿悬空，速度极低，agent 误以为成功 | 明确 rewarding 必须双脚同时接触，单脚接触给予正奖励但不足以终止训练，同时检查 episode 结束时的接触状态 |
| 过度依赖一种方向引擎导致轨迹弯曲 | 动作分布极不均衡，x_position 偏移过大 | 增大 x_position 偏离的惩罚，必要时对动作分布加熵奖励或课程引导平衡控制 |



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
- best_score_so_far: -27.460

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | 3 | -27.460 | -27.460 | unsolved |
| goal_proximity + landing_progress + orientation_penalty + velocity_penalty | 1 | -111.620 | -111.620 | unsolved |

## Previous interventions

- iter 4 (score=-111.620, structure=goal_proximity + landing_progress + orientation_penalty + velocity_penalty): selected_level**：**L2: 理由：需要改变 contact_bonus 的数学形态——从稠密状态奖励改为“接触事件 + 一次性成功着陆事件”，属于 L2 语义修改。权重会适当微调，但核心是公式形态变化。 | selected_intervention**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
