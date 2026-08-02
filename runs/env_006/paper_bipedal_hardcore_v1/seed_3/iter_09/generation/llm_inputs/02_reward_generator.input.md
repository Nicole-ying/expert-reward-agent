# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主目标：让双足机器人在崎岖地形（含梯子、树桩、坑洞、不平表面）上尽可能快地稳健前进，最终到达地形尽头。次目标：在保证稳定不摔倒的前提下，最小化关节力矩消耗。不应混滑的目标：无需精确到达特定坐标，无需抓取物体，也无需保持绝对静止；地形只是阻碍前进的扰动，不应被当作需要精确避开的多目标之一。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心要求是双足机器人跨越崎岖地面持续前进，没有指定的目标点，只有“尽可能远和高效”的进度目标；地形多样性只是增加了控制的难度，附属的能量节省不构成任务族切换。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推断)
- obs[0] hull_angle： 身体倾角，reward_usable: true
- obs[1] hull_angular_velocity： 身体角速度，reward_usable: true
- obs[2] horizontal_speed： 质心水平速度，reward_usable: true
- obs[3] vertical_speed： 质心垂直速度，reward_usable: true
- obs[4] joint_0_angle： 髋关节1角度，reward_usable: true (可能用于姿态约束)
- obs[5] joint_0_speed： 髋关节1角速度，reward_usable: true (可能用于平滑/冲击惩罚)
- obs[6] joint_1_angle： 膝关节1角度，reward_usable: true
- obs[7] joint_1_speed： 膝关节1角速度，reward_usable: true
- obs[8] joint_2_angle： 髋关节2角度，reward_usable: true
- obs[9] joint_2_speed： 髋关节2角速度，reward_usable: true
- obs[10] joint_3_angle： 膝关节2角度，reward_usable: true
- obs[11] joint_3_speed： 膝关节2角速度，reward_usable: true
- obs[12] leg_1_ground_contact： 腿1地面接触标志（1.0 接触，0 未接触），reward_usable: true
- obs[13] leg_2_ground_contact： 腿2地面接触标志，reward_usable: true
- obs[14..23] lidar_1..lidar_10： 前方地形高度测量（激光雷达），reward_usable: true (用于预判障碍，但当前不作为主奖励)

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- bounds: [-1.0, 1.0]
- action_dim 0: hip_1_torque，施加给第一个髋关节的力矩
- action_dim 1: knee_1_torque，施加给第一个膝关节的力矩
- action_dim 2: hip_2_torque，施加给第二个髋关节的力矩
- action_dim 3: knee_2_torque，施加给第二个膝关节的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain (到达地形尽头，简称为“成功到达”)
- failure-like termination: body_fallen_over (身体摔倒)
- ambiguous termination: 无
- truncation: 当前环境可能没有最大步数截断，或因超时截断但未被明确列出，暂不考虑

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（info 恒为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 当前观测数组
- action: 当前动作数组
- next_obs: 下一时刻观测数组
- info: 仅当明确允许其字段时才可使用，本环境 info 恒为空，故实质上禁止使用
- training_progress: 当前 prompt 未声明允许使用，故禁止

禁止使用：
- original_reward
- official_reward
- info 中任何字段（因为 info 为空）
- 未声明的任何其他变量

## 7. 可用于奖励函数的信号
- position/velocity:
  - horizontal_speed (obs[2], next_obs[2]): 可直接作为前进奖励
  - vertical_speed (obs[3], next_obs[3]): 可用于惩罚异常弹跳
- orientation:
  - hull_angle (obs[0], next_obs[0]): 可用于惩罚倾斜，间接预防摔倒
  - hull_angular_velocity (obs[1], next_obs[1]): 可用于惩罚急剧旋转
- joint state:
  - joint_*_angle / joint_*_speed: 可用于姿态约束或关节冲击惩罚
- contact:
  - leg_1_ground_contact, leg_2_ground_contact: 可用于步态健康约束（避免单腿停留过久或双腿同时离地），或检测摔倒征兆
- action/engine:
  - hip_1_torque .. knee_2_torque: 力矩大小可用于能效惩罚
- other:
  - lidar_1..lidar_10: 可用于预判前方陡坡，但更适合作为策略输入，作为奖励信号用途有限
- 间接推断（derived_possible）：
  - 摔倒可被推断：当 hull_angle 超过某个阈值（如 1.0 rad），或 vertical_speed 负向过大且接触信号突变，或 hull_angular_velocity 异常时，大概率已摔倒。可设计一个 penalty 但不依赖终止状态本身。
  - 到达终点可被推断：如果在连续前进过程中 episode 突然 truncated 且未检测到明显摔倒信号，可能意味着到达终点。但无法在单步奖励中准确获得该事件，仅可用于 hindsight 分析，不适合做实时单步奖励。

## 8. 不确定或不可用的信号
- info 字段全部不可用：无法获得 rewards、success、failure、distance_to_goal 等
- 无法获得精确的全局位置或里程计（除非从速度积分，但噪声大）
- 无法获得地形类别或障碍物类型
- 无法获得能量、接触力等附加物理量

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal_agent
  actuator_type: torque_controlled_rotational_joints
  contact_structure: two_legs_with_ground_contact_signals
primary_objectives:
  - make fast and stable forward progress across irregular terrain
secondary_objectives:
  - minimize joint torque consumption
main_failure_risks:
  - falling over on rough ground
  - getting stuck in pits or against tree stumps
  - wasting energy by excessive joint torque
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress_reward
  purpose: 鼓励机器人向前移动，奖励水平速度
  why_required: 这是任务核心，缺少会导致机器人原地踏步或后退。
  usable_signals: [horizontal_speed (next_obs[2])]
  risks: 若权重过高可能导致机器人用不安全的高速冲刺，增加摔倒风险；需结合姿态约束。

- role_id: posture_stability_penalty
  purpose: 惩罚身体过度倾斜和快速旋转，以避免摔倒
  why_required: 摔倒直接导致终止且无法继续前进，是主要失败模式。
  usable_signals: [hull_angle (next_obs[0]), hull_angular_velocity (next_obs[1])]
  risks: 若惩罚过强可能抑制必要的身体摆动，导致步态僵硬，反而不利于越障；需适度。

### 10.2 条件职责 conditional_roles
- role_id: joint_effort_penalty
  condition_to_use: 训练后期或已有稳定步态后，为提升效率而启用，训练初期可关闭或衰减
  usable_signals: [action (hip_1_torque..knee_2_torque)]
  risks: 早期加入可能阻碍探索出大幅度的越障动作，导致 stuck 在局部最优。

- role_id: vertical_bounce_penalty
  condition_to_use: 当检测到 vertical_speed 过大或频繁上下波动时启用，以减少多余弹跳
  usable_signals: [vertical_speed (next_obs[3])]
  risks: 过度惩罚可能妨碍跳过小障碍，建议仅对负面弹跳或不合理的高频垂直运动施加微调

### 10.3 慎用/禁用职责 avoid_roles
- role_id: step_pattern_constraint
  reason: 虽然接触标志可用，但强制规定“左右交替”步态可能限制应对复杂地形的自然非对称步态，且没有证据表明不交替会导致失败；不适合早期强制
  forbidden_or_missing_signals: 没有步态周期的标记，仅凭接触信号难以设计稳健的步态约束

- role_id: termination_based_completion_reward
  reason: 无法从 info 获得确切的成功/失败标志，且终止事件不能直接用于单步 reward 反馈（会导致稀疏 reward）；到达终点的信号也无法可靠推导，故排除
  forbidden_or_missing_signals: 缺少 explicit_success_flag

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress_reward | next_obs[2] (horizontal_speed) | None | dense_state_signal, bounded_signal (可clip) | 简单线性或对数映射 |
| posture_stability_penalty | next_obs[0] (hull_angle), next_obs[1] (hull_angular_velocity) | None | quadratic_penalty, hinge_penalty | 超过阈值再施加惩罚，避免干扰小振幅自然摆动 |
| joint_effort_penalty | action (all 4 dims) | None | squared_sum, bounded_signal (loss=sum(action^2)) | 建议在初始训练阶段关闭，后期逐步引入 |
| vertical_bounce_penalty | next_obs[3] (vertical_speed) | None | quadratic_penalty, if |v_speed|>threshold | 若地形允许小幅度跳跃，需谨慎选择阈值 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 频繁摔倒 (hull_angle 过大) | episode 长度短，平均回报低；hull_angle 轨迹频繁触及高值；接触信号频繁消失 | 增大 posture_stability_penalty 权重，或降低 forward_progress_reward 权重 |
| 机器人原地粘滞或前进极慢 | horizontal_speed 均值接近零，episode 很长但没有前进；lidar 读数几乎不变 | 检查是否因 penalty 过高抑制了动作；可暂时关闭 vertical_bounce_penalty，并降低 joint_effort_penalty |
| 用过高力矩换速度，能耗大 | action 各维度的绝对值常接近 1.0；动作平滑性差 | 适时加入 joint_effort_penalty 并逐步加码，或施加 joint_speed 变化率惩罚 |
| 只迈一边腿的僵化步态 | leg_1_contact 和 leg_2_contact 长时间不均，前进速度依然慢 | 当前不考虑步态惩罚，但可观察；若成为问题，可引入轻微的接触对称性鼓励（但需谨慎） |



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
- best_score_so_far: -52.730

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_cost + air_penalty + ang_vel_penalty + posture_penalty + progress_reward | 1 | -52.730 | -52.730 | unsolved |
| action_cost + air_penalty + ang_vel_penalty + posture_penalty + progress_reward + vertical_speed_penalty | 1 | -52.730 | -52.730 | unsolved |
| air_penalty + angular_penalty + posture_gate + progress_reward + vertical_penalty | 2 | -59.200 | -65.670 | unsolved |
| angular_penalty + posture_penalty + progress_reward + vertical_penalty | 1 | -61.550 | -61.550 | unsolved |
| angular_penalty + posture_gate + progress_reward + vertical_penalty | 1 | -61.570 | -61.570 | unsolved |
| action_cost + ang_vel_penalty + posture_penalty + progress_reward | 1 | -65.160 | -65.160 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
