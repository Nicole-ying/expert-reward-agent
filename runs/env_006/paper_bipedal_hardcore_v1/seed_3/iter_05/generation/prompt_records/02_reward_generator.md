# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
双足机器人在布满梯子、树桩、坑洼等的不规则崎岖地形上向前行进，目标是走得尽可能远并在不摔倒的前提下到达地形尽头。同时，需要尽量减少不必要的关节力矩以避免能源浪费。机器人通过 LIDAR 传感器提前感知前方地形起伏，从而在接近障碍物前调整步态。**主目标**是持续稳定地向前移动（最大化前进距离），**次目标**是保持平衡避免摔倒以及降低关节能耗。不应将“到达尽头”视为唯一的点型成功目标，而应把它看作过程性前进的自然终结。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标不受特定目标位置约束，而是尽可能在困难地形上持续前进；附属目标包括防摔倒和降低能耗，都属于连续移动任务的常见额外偏好，并不构成等价的多目标冲突，因此不属于 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（隐含，从描述推断为连续数值）
- obs[0]: hull_angle, 机体俯仰角(rad)，reward_usable: true
- obs[1]: hull_angular_velocity, 机体俯仰角速度(rad/s)，reward_usable: true
- obs[2]: horizontal_speed, 机体水平前进速度(m/s)，reward_usable: true
- obs[3]: vertical_speed, 机体垂直速度(m/s)，reward_usable: true
- obs[4]: joint_0_angle, 髋关节1角度(rad)，reward_usable: true（可做姿态约束）
- obs[5]: joint_0_speed, 髋关节1角速度(rad/s)，reward_usable: true
- obs[6]: joint_1_angle, 膝关节1角度(rad)，reward_usable: true
- obs[7]: joint_1_speed, 膝关节1角速度(rad/s)，reward_usable: true
- obs[8]: joint_2_angle, 髋关节2角度(rad)，reward_usable: true
- obs[9]: joint_2_speed, 髋关节2角速度(rad/s)，reward_usable: true
- obs[10]: joint_3_angle, 膝关节2角度(rad)，reward_usable: true
- obs[11]: joint_3_speed, 膝关节2角速度(rad/s)，reward_usable: true
- obs[12]: leg_1_ground_contact, 腿1触地标志（0/1），reward_usable: true（可用于检测支撑相）
- obs[13]: leg_2_ground_contact, 腿2触地标志（0/1），reward_usable: true
- obs[14]: lidar_1, 前方地形高度测量1 (m)，reward_usable: false（高程信息难以直接翻译成标量奖励，除非有复杂地形适应目标）
- obs[15]: lidar_2, reward_usable: false
- obs[16]: lidar_3, reward_usable: false
- obs[17]: lidar_4, reward_usable: false
- obs[18]: lidar_5, reward_usable: false
- obs[19]: lidar_6, reward_usable: false
- obs[20]: lidar_7, reward_usable: false
- obs[21]: lidar_8, reward_usable: false
- obs[22]: lidar_9, reward_usable: false
- obs[23]: lidar_10, reward_usable: false

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- action[0]: hip_1_torque, 髋关节1力矩（归一化）
- action[1]: knee_1_torque, 膝关节1力矩
- action[2]: hip_2_torque, 髋关节2力矩
- action[3]: knee_2_torque, 膝关节2力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain`（到达地形尽头），可视为成功完成移动任务。
- failure-like termination: `body_fallen_over`（机体摔倒），应被视为失败。
- ambiguous termination: 无。只有上述两种明确终止。
- truncation: 无额外时间截断说明，但往往有 episode 长度上限，未明确说明，在任务分析中暂不作为独立终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无法直接读取到达尽头标志）
- explicit_failure_flag_available: false（info 为空，无法直接读取摔倒标志）
- allowed_info_fields: []（空列表，所有 info 字段不可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

终止类型必须通过观测信号间接推断：
- **摔倒可能推断路径**：`hull_angle` 的绝对值急剧增大（例如超过 0.5~1.0 rad），或 `hull_angular_velocity` 在终止前骤增；同时 `leg_1_ground_contact` 和 `leg_2_ground_contact` 同时消失（腿离地）可作为辅助证据。记为 **derived_possible**。
- **到达尽头可能推断路径**：episode 提前结束且观测并未出现上述摔倒特征，尤其是 `horizontal_speed` 从前一时刻的正常前进突然终止，可视为达到尽头。也记为 **derived_possible**。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段（此处 info 为空，故**不可用**任何 info 字段）
- training_progress 仅在后续明确指示允许时可用（当前任务未指明，应视为**禁用**）

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片（例如不能假设存在不在观测列表中的绝对位置）

## 7. 可用于奖励函数的信号
以下信号可从 obs、next_obs、action 及 derived_possible 推断中获得：
- position: 无绝对位置可直接使用（没有 x 坐标）
- velocity: 
  - next_obs[2] (horizontal_speed) —— **直接可用**
  - next_obs[3] (vertical_speed) —— 可用作平滑惩罚
- orientation:
  - next_obs[0] (hull_angle) —— **直接可用**，惩罚过大倾斜
  - next_obs[1] (hull_angular_velocity) —— 直接可用，惩罚急剧转动
- contact:
  - next_obs[12] (leg_1_ground_contact)
  - next_obs[13] (leg_2_ground_contact)
  可用于检测腿是否着地，或奖励交替支撑。
- action/engine:
  - action[0..3]（关节力矩）—— **直接可用**，惩罚能耗
- other:
  - **derived_possible: fall_detected** —— 从 hull_angle 和 angular_velocity 超过阈值推断摔倒
  - **derived_possible: end_reached** —— 推断到达尽头，但风险较高，一般不单独作为奖励信号，而由外部截断终止处理。

## 8. 不确定或不可用的信号
- 绝对位置 x, y, z 不存在于观测中
- 地形高度真值（只有相对 lidar 测量，无法直接得 reward）
- 成功/失败标志不存在（info 空）
- training_progress 当前不可用
- 能量消耗的积分量不存在
- 目标速度或目标位置不存在

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_rough_terrain_gait
control_type: continuous
morphology:
  body_type: bipedal_planar_body
  actuator_type: torque_controlled_joints (2 hips, 2 knees)
  contact_structure: two_feet_ground_contact
primary_objectives:
  - maximize_forward_progress_on_rough_terrain
secondary_objectives:
  - avoid_falling
  - minimize_joint_torque_energy
main_failure_risks:
  - falling_over_due_to_obstacle_impact_or_imbalance
  - getting_stuck_and_not_moving_forward
  - excessive_energy_consumption_without_progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 激励 agent 向前移动
  why_required: 任务核心是走得远，主 reward 必须驱使前进
  usable_signals: [next_obs[2] horizontal_speed]
  risks: 可能导致 agent 忽略平衡，摔倒时仍追求速度；需要配合 fall_prevention

- role_id: fall_prevention
  purpose: 抑制导致摔倒的姿态
  why_required: 一旦摔倒 episode 即终止，无法继续前进；必须通过惩罚危险姿态来保护 agent
  usable_signals: [next_obs[0] hull_angle, next_obs[1] hull_angular_velocity, derived_possible fall_detected]
  risks: 过强的惩罚可能使 agent 不敢迈步，导致保守的原地踏步

- role_id: energy_efficiency
  purpose: 降低不必要的大关节力矩，节约能量
  why_required: 任务明确要求 minimize unnecessary joint torque
  usable_signals: [action[0..3]]
  risks: 若权重过大，可能迫使 agent 用极小动作，导致前进不足

### 10.2 条件职责 conditional_roles
- role_id: contact_symmetry
  condition_to_use: 当 agent 已经有初步前进能力后，可鼓励双腿交替支撑以形成自然步态，避免单腿过度依赖
  usable_signals: [next_obs[12], next_obs[13] leg ground contact indicators]
  risks: 初期加入可能干扰探索，因为稳定步态尚未形成

- role_id: vertical_smoothness
  condition_to_use: 当地形过于崎岖时，加入轻微惩罚垂直速度剧烈波动，可提高运动平稳性
  usable_signals: [next_obs[3] vertical_speed]
  risks: 过度平滑可能阻碍 agent 通过障碍所需的跳跃动作

### 10.3 慎用/禁用职责 avoid_roles
- role_id: alive_bonus
  reason: 简单生存奖励会鼓励原地待命而不前进，与主目标冲突
  forbidden_or_missing_signals: 无对应终止信号可度量生存

- role_id: target_velocity_tracking
  reason: 没有预设的目标前进速度；此环境要求尽可能远而不是恒定速度，不应使用
  forbidden_or_missing_signals: 缺少目标速度值

- role_id: lidar_based_terrain_penalty
  reason: 奖励函数中很难将 lidar 高程图直接翻译成标量惩罚项，若设计不当反而会引入噪声；可在后续迭代中考虑，但当前不建议
  forbidden_or_missing_signals: 无法将多维度 lidar 信号简化为合理的标量奖励

## 11. role_to_signal_mapping
| role_id | usable



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
- best_score_so_far: -59.200

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| air_penalty + angular_penalty + posture_gate + progress_reward + vertical_penalty | 2 | -59.200 | -65.670 | unsolved |
| angular_penalty + posture_penalty + progress_reward + vertical_penalty | 1 | -61.550 | -61.550 | unsolved |
| angular_penalty + posture_gate + progress_reward + vertical_penalty | 1 | -61.570 | -61.570 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
