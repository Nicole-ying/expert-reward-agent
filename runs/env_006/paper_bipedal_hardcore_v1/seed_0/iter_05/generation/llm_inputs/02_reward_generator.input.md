# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
双足机器人在布满阶梯、树桩、坑洞等不规则障碍物的崎岖地面上持续前进，走的越远且越高效越好。机器人配备 LIDAR 测距仪，可提前感知前方地形高度变化。每条腿的髋关节和膝关节均由独立的力矩控制器驱动。任务是让机器人学会在不规则地面上稳定行走，利用 LIDAR 信息提前调整步态以应对前方障碍，同时避免摔倒并尽量减少不必要的关节力矩消耗。当机器人摔倒或到达地形尽头时，回合结束。

核心目标：持续、稳定地通过崎岖地面继续前进（或到达尽头），次要目标：避免摔倒、降低关节力矩消耗。不应将“到达终点”视作唯一导航目标，而是作为行进完成的自然结果。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务核心是在复杂地形上连续移动，主目标是保持稳定并尽量走远，没有明确的坐标导航目标；附属能耗和姿态要求不影响主目标性质。最匹配 locomotion_continuous_control（持续前进通过地形）。

动力学子类型 dynamics_subtype: planar_bipedal_gait  
解释：双足、髋膝四关节力矩驱动，平面步态前进，尽管地形粗糙，但动力学本质仍是平面双足行走。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推断）
- 字段详解（index 从 0 开始）：

| Index | 名称 | 含义 | reward_usable |
|-------|------|------|---------------|
| 0 | hull_angle | 身体姿态角（俯仰） | true |
| 1 | hull_angular_velocity | 身体姿态角速度 | true |
| 2 | horizontal_speed | 质心水平速度（前进方向） | true |
| 3 | vertical_speed | 质心垂直速度 | true |
| 4 | hip_1_angle | 髋关节 1 角度 | true |
| 5 | hip_1_speed | 髋关节 1 角速度 | true |
| 6 | knee_1_angle | 膝关节 1 角度 | true |
| 7 | knee_1_speed | 膝关节 1 角速度 | true |
| 8 | hip_2_angle | 髋关节 2 角度 | true |
| 9 | hip_2_speed | 髋关节 2 角速度 | true |
| 10 | knee_2_angle | 膝关节 2 角度 | true |
| 11 | knee_2_speed | 膝关节 2 角速度 | true |
| 12 | leg_1_ground_contact | 腿 1 触地指示（0/1） | true |
| 13 | leg_2_ground_contact | 腿 2 触地指示（0/1） | true |
| 14‑23 | lidar_1 … lidar_10 | 前方 10 束激光测距读数（地形高度） | true（高级用法） |

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [4]
- action bounds: [-1.0, 1.0]
- 各维度含义：

| Action dim | 名称 | 含义 |
|------------|------|------|
| 0 | hip_1_torque | 施加于第一个髋关节的力矩 |
| 1 | knee_1_torque | 施加于第一个膝关节的力矩 |
| 2 | hip_2_torque | 施加于第二个髋关节的力矩 |
| 3 | knee_2_torque | 施加于第二个膝关节的力矩 |

控制类型：continuous（力矩控制，值被截断在 [-1,1] 内）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination：reached_end_of_terrain（到达地形尽头）
- failure-like termination：body_fallen_over（身体摔倒）
- ambiguous termination：无
- truncation：不含截断终止（可能仅在达到最大时间步时截断，但未说明）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（info 为空）

终止原因可通过最后一步的观测信号间接推断，属于 derived_possible：
- 摔倒：可通过 hull_angle 绝对值超过阈值（如 > 0.7 rad）、或 vertical_speed 突然大幅向下等判断。由于终止仅发生于身体摔倒，所以 terminated 且满足摔倒观测条件时一定是摔倒。
- 到达终点：episode 在正常的行进中提前结束，且观测中无摔倒迹象（hull_angle 较小、速度正常），可合理推断为 reached_end_of_terrain。

在奖励函数设计中，可以在终态时利用推导信号分配成功奖励或失败惩罚，但不可依赖显式 info 标签。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，24维）
- action（当前动作，4维）
- next_obs（下一观测，24维）
- info 中明确允许的字段（当前无允许字段，info 始终为空，故 info 不可用）
- training_progress 仅在 prompt 明确允许时使用（此处未提及，建议禁用）

禁止使用：
- original_reward（官方奖励被屏蔽）
- official_reward（同源）
- 任何未声明的 info 字段
- 任何超出 obs[0:24] 范围的数据

## 7. 可用于奖励函数的信号
以下信号可直接作为奖励或惩罚的输入：

- **姿态/稳定信号**：hull_angle（理想为 0），hull_angular_velocity（用作平滑项）
- **前进速度**：horizontal_speed（正值表示前进，可做前进奖励核心）
- **垂直运动**：vertical_speed（接近 0 的稳定状态可给予奖励）
- **关节状态**：各关节角与角速度（可用于步态规范、能耗惩罚的基础）
- **触地信号**：leg_1_ground_contact、leg_2_ground_contact（监测正确支撑模式，避免双足同时离地等）
- **动作量**：action（torque）的 L2 范数或绝对值（作为能量消耗的代理）
- **LIDAR**：lidar_1 … lidar_10（可间接用来鼓励根据地形提前调整姿态，但设计复杂，属于进阶信号）
- **摔倒推断**（derived_possible）：终止时的 hull_angle 或 vertical_speed 异常，可构建摔倒惩罚
- **到达终点推断**（derived_possible）：终止时正常姿态，给予到达奖励

## 8. 不确定或不可用的信号
- 绝对世界坐标（x,y）：环境未提供
- 与终点的距离或进度百分比：无
- 显式 done_reason 或 success flag：info 为空，不可用
- 地形高度图或路径信息：除 10 束 LIDAR 外无全局信息
- 能量总消耗计量：只能通过累积力矩间接计算
- 任何官方奖励函数组件：被屏蔽

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: 单刚体躯干 + 两条腿
  actuator_type: 力矩驱动髋膝（4个独立电机）
  contact_structure: 两腿交替触地（提供二值接触信号）
primary_objectives:
  - 在崎岖地面上尽可能远地前进（或到达尽头）
  - 避免摔倒
secondary_objectives:
  - 最小化关节力矩（能耗）
  - 维持平稳姿态（减少垂直振荡）
main_failure_risks:
  - 身体倾角过大导致摔倒
  - 步态失调，前进缓慢或原地打转
  - 关节过度用力，能量浪费
  - 未能利用 LIDAR 信息导致在障碍物处失稳
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: progress_reward**
  purpose: 鼓励机器人获得正向水平位移，推进时间越长越好
  why_required: 这是任务核心，没有前进奖励策略不会产生持续行走行为
  usable_signals: [horizontal_speed（obs[2]），可以通过时间差分 next_obs[2] - obs[2] 获得加速度，但 best 是直接累积或速度加权；也可以配合触地事件细化每步前进量]
  risks: 过度奖励水平速度可能鼓励机器人用危险姿态猛冲；需要限定速度上限或结合姿态约束

- **role_id: fall_penalty**
  purpose: 惩罚摔倒行为，迫使机器人学会稳定步态
  why_required: 摔倒直接导致 episode 结束，极大降低收益；明确惩罚是保证安全性的关键
  usable_signals: [终止时观测的 hull_angle（obs[0]）、vertical_speed（obs[3]），derived_possible 摔倒推断（角度超出阈值+垂直速度下拉）]
  risks: 惩罚设计不当可能使机器人过于谨慎而不敢迈步；最好只在 episode 结束时施加一次大的负奖励，而非每步都惩罚姿态

### 10.2 条件职责 conditional_roles
- **role_id: energy_efficiency_penalty**
  purpose: 降低关节



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

- target_score: 300.000
- best_score_so_far: -49.540

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| gated_forward_speed + stability_tilt_hinge_penalty | 2 | -49.540 | -49.540 | unsolved |
| gated_forward_speed + posture_hinge_penalty | 1 | -57.470 | -57.470 | unsolved |
| gated_forward_speed + stability_quad_penalty | 1 | -62.540 | -62.540 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
