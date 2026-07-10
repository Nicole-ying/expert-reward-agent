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

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
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
本环境要求一个平面双足身体在起伏不平的地形上持续前进。主目标是**尽可能远、尽可能快地向前移动**，同时**尽可能减少能量消耗**（动作幅度/力矩）。次要目标是**维持上身稳定、避免摔倒**，因为一旦身体倾覆或到达地形尽头，回合即终止。因此奖励设计需引导智能体学习高效、平稳的前进步态，而不是到达某个固定终点。

## 2. 任务类型选择
- selected_route_id: **locomotion_continuous_control**
- confidence: **high**
- reason: 核心驱动是"前进超远+高速+低能耗"，没有指定的目标位置或终点坐标，终止条件仅有摔倒或走完整个地形（而非导航成功）。属于通过连续关节力矩控制实现步行能力的典型移动任务，故归入连续运动控制族。

动力学子类型细化：
- dynamics_subtype: **planar_bipedal_gait**  
  理由：两腿、髋与膝关节、身体倾斜角度与接触信息，任务要求产生稳定的双足步态在2D纵向平面内前进。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推断）
- 各维度说明与 reward_usable 标记：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | hull_angle | 主体相对于竖直方向的倾角（越大越倾斜） | true |
| 1 | hull_angular_velocity | 主体倾角的角速度 | true |
| 2 | horizontal_velocity | 前进/后退方向的线速度（正值=向前） | true |
| 3 | vertical_velocity | 上下方向线速度（正值=上升） | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1是否触地（1.0=触地，0.0=悬空） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
| 10 | hip2_speed | 腿2髋关节角速度 | true |
| 11 | knee2_angle | 腿2膝关节角度 | true |
| 12 | knee2_speed | 腿2膝关节角速度 | true |
| 13 | leg2_contact | 腿2是否触地 | true |
| 14–23 | lidar_0 … lidar_9 | 10个激光雷达测距值，描述前方地形轮廓 | true |

所有维度均可作为奖励信号来源，但雷达数据仅反映地形几何，直接用于奖励需要谨慎映射（如惩罚前方陡坡）。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 连续，各维范围 [-1.0, 1.0]
- 动作维度含义：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | hip_torque_leg1 | 施加在腿1髋关节上的力矩 |
| 1 | knee_torque_leg1 | 施加在腿1膝关节上的力矩 |
| 2 | hip_torque_leg2 | 施加在腿2髋关节上的力矩 |
| 3 | knee_torque_leg2 | 施加在腿2膝关节上的力矩 |

动作幅度的平方和可作为能量消耗的代理指标。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：`reached_end_of_terrain` – 走完前方全部地形，通常代表任务圆满达成。
- **failure-like termination**：`body_fallen_over` – 身体倾覆，失去平衡，代表失败。
- **ambiguous termination**：默认 `terminated` 为布尔值，在 step 返回时无法区分上述两种原因。
- **truncation**：未提及超时截断，推测由环境内部最大步数控制，但在本次 `step` 接口中未见传递。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无法获取明确的成功标记）
- **explicit_failure_flag_available**: false  
  （同上）
- **allowed_info_fields**：无（info = {}）
- **forbidden_or_uncertain_info_fields**：全部 info 子字段均为不可用，不得依赖任何未声明的键。

**关键结论**：奖励函数**不可基于终止原因**来区分成功/失败（如不可给到达终点额外奖励），因为无法从 `terminated` 布尔值可靠分离两种终止情况。奖励逻辑必须全部基于状态‑动作历程的内部信号构建。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

- **允许使用**：
  - `obs`（shape [24]）
  - `action`（shape [4]）
  - `next_obs`（shape [24]）
  - 允许使用 `info` 中**明确已声明的字段**（当前为空，故不可用任何 info 键）
  - `training_progress` 仅在 prompt 明确允许时使用，当前约束下视为可选但非必要
- **严格禁止使用**：
  - `original_reward`（被屏蔽的官方奖励）
  - 任何未在任务描述中给出的 obs 切片含义
  - 任何未在 step 源代码中出现的 `info` 键（如伪装的 "success"、"failure"、"reason"）

## 7. 可用于奖励函数的信号
- **位置相关**：雷达测距（lidar_*）提供前方地形相对距离，可间接反映位置与地形起伏，但不提供绝对位置（无x坐标）。
- **速度相关**：`horizontal_velocity`（前进速度，可直接奖励）、`vertical_velocity`（可用于检测跳跃、坠落或异常运动）。
- **姿态/平衡相关**：`hull_angle`、`hull_angular_velocity`，身体倾角与角速度。
- **接触相关**：`leg1_contact`、`leg2_contact`，可用于鼓励交替触地，或惩罚双腿同时离地。
- **动作/力矩相关**：`action` 各维（力矩），平方和或绝对值可作为能耗惩罚。
- **关节状态**：四条关节角度与角速度，可用于惩罚过大的关节运动或极限位置。
- **地形预览**：lidar_0 … lidar_9，可辅助判断是否为陡坡或平台，谨慎作为奖励调整项。

## 8. 不确定或不可用的信号
- **绝对行进距离**：未提供前进方向的位置坐标（如 x 坐标），不能直接按位移给奖励；水平速度积分可作为替代。
- **地形终点信号**：`reached_end_of_terrain` 终止原因不可读，无法作为奖励。
- **成功/失败标签**：无显式标记。
- **能量消耗真值**：无直接能量传感器，只能由动作力矩幅值近似。
- **高度/海拔**：无主体绝对高度（除垂直速度、倾角与雷达），不可直接获得海拔。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: single_rigid_body_with_two_legs
  actuator_type: torque_controlled_hip_and_knee_joints
  contact_structure: two_foot_ground_contact (contact flags)
primary_objectives:
  - maximize forward horizontal velocity (and thus displacement)
  - maintain stable upright posture (minimize tilt/angular rate)
  - avoid episode termination by falling
secondary_objectives:
  - minimize energy consumption (low joint torques)
  - optionally adapt to uneven terrain ahead (lidar-informed)
main_failure_risks:
  - falling over (excessive body tilt or loss of contact)
  - oscillating in place without forward progress
  - high energy cost with jerky motions
  - inability to handle steeper terrain sections
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: forward_velocity_reward**
  - purpose: 鼓励快速向前移动，直接关联主要目标。
  - why_required: 任务明确要求“尽可能远、尽可能快”。
  - usable_signals: `next_obs[2]` (horizontal_velocity)
  - risks: 若为纯线性奖励，可能驱动大步伐高能耗；若不加限制，可能忽略平衡。

- **role_id: stability_balance_penalty**
  - purpose: 惩罚身体倾斜与快速旋转，防止跌倒，保证步态稳健。
  - why_required: 摔倒会直接终止回合，持续行走的基础是保持直立。
  - usable_signals: `next_obs[0]` (hull_angle), `next_obs[1]` (hull_angular_velocity)
  - risks: 惩罚过重会压抑动作幅度，导致原地站立不动。

- **role_id: energy_efficiency_penalty**
  - purpose: 最小化关节力矩的使用，鼓励节能步态。
  - why_required: 任务描述中包含“while minimizing energy consumption”。
  - usable_signals: `action` 各维（平方和或绝对值和）
  - risks: 过早加入可能导致学习迟缓，可在训练后期加大权重。

### 10.2 条件职责 conditional_roles
- **role_id: gait_contact_pattern**
  - condition_to_use: 在初步站稳或移步后用，辅助生成交替触地的步态（避免双足同时离地或长期单腿）。
  - usable_signals: `next_obs[8]` (leg1_contact), `next_obs[13]` (leg2_contact)
  - risks: 信号较稀疏，奖励形式需精心设计，否则可能干扰刚启动时的探索。

- **role_id: terrain_aware_caution**
  - condition_to_use: 在基础步态稳定后，利用激光雷达预兆惩罚即将面临的陡坡或障碍以提前准备。
  - usable_signals: `next_obs[14:24]` (lidar_0..lidar_9)
  - risks: 雷达原始数值无法直接反映“陡峭度”，需要经过加工（如差分），且噪声可能误导。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id:



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子和少量动力学类型示例，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive: `w * signal`
  - penalty: `-w * abs(error)` 或 `-w * error**2`
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：无界状态值可能支配总奖励；状态值可能被占据/刷分，而不代表任务完成。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - `x / (1 + abs(x))`
  - `1 / (1 + k * abs(error))`
  - `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或 velocity/proximity 类信号容易被刷。
- 风险：threshold 过小会导致反馈饱和或无梯度。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * (1 / (1 + k * abs(posture_error)))`
- 使用条件：前进/接近奖励导致不健康冲刺、翻倒或失稳。
- 风险：gate 太严格会抑制探索；跳跃类任务尤其要轻。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 风险：乘积容易塌缩；单一接触或单一事件不能当成功。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。


```
