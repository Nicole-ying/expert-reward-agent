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
这是一个三维四足机器人在连续空间中的运动控制任务。机器人拥有8个扭矩控制关节（4条腿各2关节），必须控制关节扭矩使机器人**稳定地向前行走或奔跑**。**主要目标**是产生持续的向前运动（forward locomotion），**必要存活约束**是保持身体高度在健康范围内（0.2 m ~ 1.0 m），一旦高度超出该范围即提前终止。同时机器人需要尽量**保持躯干直立**，避免侧翻或倾覆。任务强调“维持稳定前进而非仅仅保持站立”，因此奖励设计应鼓励有效的向前位移和步态稳定性，而非简单奖励存活时间。**注意**：不存在显式的位置目标或到达点位，核心是持续、稳健的向前推进。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务明确要求“向前行走/奔跑”作为核心目标，没有固定的目标位置，机器人需要在连续的平地上持续产生前进运动，符合 locomotion_continuous_control 族特征。附属的存活、直立、能耗等均为安全性或平滑性约束，不是权重相当的多目标。四足运动属于典型的多腿持续前进控制。

动力学子类型（dynamics_subtype）进一步细分为 **multi_legged_body_locomotion**，因为该环境具有多足（四足）、高维身体状态、需协调步态完成向前行进的特点。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float32 (推断)
- 各维度含义：

| 索引 | 名称 | 含义 | 可用于奖励 |
|------|------|------|-------------|
| 0 | body_z | 主体中心的垂直高度 (m) | true |
| 1 | quat_w | 身体姿态四元数实部 | true (组合可得竖直分量) |
| 2 | quat_x | 身体姿态四元数 x 分量 | true |
| 3 | quat_y | 身体姿态四元数 y 分量 | true |
| 4 | quat_z | 身体姿态四元数 z 分量 | true |
| 5 | joint_1_angle | 第一髋关节角度 | true (运动学使用) |
| 6 | joint_2_angle | 第一踝关节角度 | true |
| 7 | joint_3_angle | 第二髋关节角度 | true |
| 8 | joint_4_angle | 第二踝关节角度 | true |
| 9 | joint_5_angle | 第三髋关节角度 | true |
| 10 | joint_6_angle | 第三踝关节角度 | true |
| 11 | joint_7_angle | 第四髋关节角度 | true |
| 12 | joint_8_angle | 第四踝关节角度 | true |
| 13 | body_x_velocity | 主体在世界 x 方向的速度（前进方向） | true（核心信号） |
| 14 | body_y_velocity | 主体侧向速度 | true（可抑制侧移） |
| 15 | body_z_velocity | 主体垂直速度 | true（着陆冲击等） |
| 16 | body_roll_velocity | 翻滚角速度 | true（稳定性） |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true（航向保持） |
| 19 | joint_1_velocity | 第一髋关节角速度 | true |
| 20 | joint_2_velocity | 第一踝关节角速度 | true |
| 21 | joint_3_velocity | 第二髋关节角速度 | true |
| 22 | joint_4_velocity | 第二踝关节角速度 | true |
| 23 | joint_5_velocity | 第三髋关节角速度 | true |
| 24 | joint_6_velocity | 第三踝关节角速度 | true |
| 25 | joint_7_velocity | 第四髋关节角速度 | true |
| 26 | joint_8_velocity | 第四踝关节角速度 | true |

备注：没有接触力、足端信息、相对位置等，仅提供纯本体状态量。

## 4. 动作空间 action_space
- type: Box (连续)
- shape: [8]
- dtype: float32
- 范围: 每个维度 [-1.0, 1.0]

| 维度 | 名称 | 含义 |
|------|------|------|
| 0 | hip_1_torque | 第一条腿髋关节力矩 |
| 1 | ankle_1_torque | 第一条腿踝关节力矩 |
| 2 | hip_2_torque | 第二条腿髋关节力矩 |
| 3 | ankle_2_torque | 第二条腿踝关节力矩 |
| 4 | hip_3_torque | 第三条腿髋关节力矩 |
| 5 | ankle_3_torque | 第三条腿踝关节力矩 |
| 6 | hip_4_torque | 第四条腿髋关节力矩 |
| 7 | ankle_4_torque | 第四条腿踝关节力矩 |

动作为直接扭矩控制，没有目标角度、目标角速度等高层接口，属于低层扭矩级控制。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **没有显式成功终止条件**。任务不存在“到达目标”概念，只有持续存活和前进。因此唯一可能被认为是“成功”的情况是 episode 自然截断（time limit），但需配合前向移动和高度稳定来推断其质量。
- failure-like termination: 
  - 身体高度超出健康范围（min_z=0.2, max_z=1.0），原因可能是摔倒、跃起过高等。
  - 状态值变为 NaN 或 Inf。
- ambiguous termination: 无。
- truncation: 达到最大步数限制（时间截断），不属于失败，但也不一定代表任务完成良好（可能仅仅维持站立不动）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  没有任何 info 字段指示成功，且任务本身没有二值成功定义。
- explicit_failure_flag_available: true  
  失败可以通过 terminated 且不是截断来间接推断，但不能在 compute_reward 中使用 terminated 标志（因为 compute_reward 不直接接收 terminated 标志，除非由环境在每步结束时提供）。在奖励函数中，只能利用 next_obs 中 body_z 是否在健康范围内来惩罚即将发生的失败；不能直接使用 episode 的终止状态。
- allowed_info_fields: [] （空，无任何可用于奖励的 info 字段）
- forbidden_or_uncertain_info_fields: （明确禁止）
  - reward_forward
  - reward_ctrl
  - reward_contact
  - reward_survive
  - x_position
  - y_position
  - distance_from_origin

**说明**：奖励函数必须完全基于 obs、action、next_obs 和禁止使用 original_reward、info 中的任何字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: numpy array shape (27,)
- action: numpy array shape (8,)
- next_obs: numpy array shape (27,)
- info: dict，但当前环境下为**空字典**，禁止使用任何字段
- training_progress: float 0.0~1.0，仅在 prompt 明确要求时使用（当前未要求，谨慎）

禁止使用：
- original_reward（已遮盖）
- official_reward
- info 中的任何字段（包括任何 reward_*）
- 未声明的 obs 切片扩展语义（如基于物理引擎的内部状态）
- 任何全局/非马尔可夫状态（如累积距离）

## 7. 可用于奖励函数的信号
- **position**:
  - 身体高度：obs[0] (body_z)；next_obs[0]
  - 身体姿态四元数：obs[1:5]；从中可计算躯干 z 轴在世界系投影 upright_proj = 1 - 2*(quat_x² + quat_y²)
  - 各个关节角度：obs[5:13]，用于构造姿态偏好或关节限位惩罚（若有边界隐式，但未给出具体限位；可假设一定范围）
- **velocity**:
  - 前进速度（主方向）：obs[13] (body_x_velocity)，可直接作为正向激励
  - 侧向速度：obs[14]，鼓励接近 0
  - 垂直速度：obs[15]，结合触地判断（无触地信息）或避免过大冲击
  - 本体角速度：obs[16:19] (roll, pitch, yaw rate)，用于惩罚翻倒、急转
  - 关节角速度：obs[19:27]，可用于动作平滑性或能耗惩罚
- **orientation**:
  - 通过四元数计算 upright = 1 - 2*(q_x²+q_y²)，范围为 [-1, 1]，1.0表示完全直立
- **contact**: 无接触力、无足底压力等信号
- **action/engine**:
  - action 本身可用于惩罚过大扭矩或 torch 变化率（相邻步 action 差值在智能体中可用，但 reward 函数是单步无记忆，且没有提供 prev_action，故不能直接使用相邻步差。若需要平滑，只能用相邻步 action 差，但 compute_reward 只接收当前步和下一步的 obs, action。没有 prev_action，无法计算 action 平滑。但可以通过 next_obs 的关节加速度间接推断？不直接。我们只能利用当前 action 大小。）
- **other**:
  - termination 标记无法在 compute_reward 中使用，因为函数不接收 terminated 布尔值（但可通过 next_obs 有效状态推断，如果 next_obs 无效？但 step 返回终止后应停止调用 compute_reward，所以不会用到无效 next_obs。不过可以通过观察当前 obs 和 next_obs 的高度是否进入危险区来塑造）。

## 8. 不确定或不可用的信号
- 世界绝对位置 x, y 被禁止，不能重建
- 无接触信息，无法给足端触地奖励或空翻检测
- 无能耗量（电机转矩*速度）可直接计算，但可利用 action（力矩）和关节速度乘积近似，但无关节电流等精确能耗
- 无显式存活奖励计数器或步数

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous (torque)
morphology:
  body_type: 3D quadruped, four legs, each with 2 DoF (hip + ankle)
  actuator_type: torque-controlled (8 independent joints)
  contact_structure: four point feet contacting ground, no contact force observation
primary_objectives:
  - maximize forward velocity (body_x_velocity)
  - maintain body height within safe range (0.2, 1.0)
secondary_objectives:
  - keep body upright (quaternion-derived up-projection close to 1.0)
  - minimize lateral drift (body_y_velocity near 0)
  - smooth locomotion (low action magnitudes or low body angular velocities)
main_failure_risks:
  - falling forward/backward or sideways causing early termination
  - standing statically without meaningful forward progress (low velocity)
  - excessive jumping or launching that triggers upper height bound
  - high-frequency torque oscillation causing instability or energy waste
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward
  purpose: 驱动机器人产生正向前进速度，使 agent 不止于站立。
  why_required: 这是任务的核心目标（locomotion），若无此奖励，策略可能只学会稳定站立。
  usable_signals: [obs[13] (body_x_velocity)]，必要时可做限幅或死区。
  risks: 若无限激励，可能导致机器人奔跑不稳、频繁摔倒，需结合存活和稳定性约束。

- role_id: height_survival_reward (or penalty)
  purpose: 将身体高度维持在健康范围 (0.2, 1.0) 内，避免提前终止；靠近边界时给予惩罚，正常范围给予中性或小额奖励。
  why_required: 高度超出直接终止，策略必须学会避免，属于安全硬约束。
  usable_signals: [obs[0], next_obs[0]]，可定义区域，如接近0.25或0.9时给予负奖励。
  risks: 如果惩罚太强，可能抑制探索；太弱则不能有效防止摔倒。

- role_id: upright_orientation_reward
  purpose: 保持躯干基本直立（世界z轴），防止侧翻。
  why_required: 摔倒与高度崩塌高度相关，且对向前运动有破坏。
  usable_signals: [obs[1:5] 四元数]，计算 body_up_z = 1 - 2*(quat_x²+quat_y²)，当该值 > 0.7 左右视为稳定。
  risks: 过于苛刻会阻碍轻微倾斜步态的自然学习，需要容忍一定范围的倾角。

### 10.2 条件职责 conditional_roles
- role_id: lateral_motion_penalty
  purpose: 抑制侧向漂移，使前进方向更纯净。
  condition_to_use: 当训练初期出现明显侧向漂移、或因侧向速度过大导致不稳定时加入；稳定后可降低系数。
  usable_signals: [obs[14] (body_y_velocity)]，可加绝对值或平方。
  risks: 若双侧对称运动天然有小幅侧向震荡，惩罚过大会抑制正常步态，应保守使用。

- role_id: action_magnitude_penalty
  purpose: 降低能量消耗、抑制高频振荡，提高样本效率。
  condition_to_use: 当观察到动作大幅震荡或 motor overheat 时加入；或作为微调后的细化项。
  usable_signals: [action](8维向量)，可求和平方。
  risks: 过早加入会抑制探索，导致步伐极弱或站姿；需在基础前进能力稳定后使用。

- role_id: joint_velocity_penalty
  purpose: 与动作惩罚类似，通过关节速度间接鼓励平滑动作。
  condition_to_use: 若无法获取上一动作差值，可替代平滑项使用。
  usable_signals: [obs[19:27] 关节速度]。
  risks: 可能误伤需要的快速摆动，最好与前进速度奖励协同调权。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_survival_time_reward
  reason: 无 info 字段，无法获取生存步数。若想用，可能扭曲为“尽量不动”，不可行。
- role_id: distance_from_origin_reward
  reason: info 中的 x_position 被禁止，无法安全构造；利用速度积分也违背无状态约束。
- role_id: contact_foot_reward
  reason: 没有任何接触信号，无法判断哪只脚着地，故禁用。
- role_id: multi_objective_weights_learning
  reason: 不属于当前环境的需求，且无辅助信号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes



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
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

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
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |





# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: 67.710

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_reward + height_reward + upright_reward | 1 | 67.710 | 67.710 | unsolved |
| action_penalty + forward_gated_height | 1 | -55.540 | -55.540 | unsolved |
| action_penalty + forward_gated + height_reward | 1 | -112.410 | -112.410 | unsolved |
| forward_gated + height_reward | 1 | -271.100 | -271.100 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
