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
控制一个3D四足机器人向前稳定行走/奔跑。核心目标是产生持续的前向速度，同时保持身体高度在安全范围（0.2 ~ 1.0）内不摔倒。次要目标包括维持直立姿态、减少侧向漂移、控制能耗和动作平滑。任务 **不要求** 到达某个指定位置，仅要求长期存活并向前移动。不能混淆为“仅站立不动”或“最小化能量消耗”，前进是刚性主目标。

## 2. 任务类型选择
selected_route_id: **locomotion_continuous_control**  
confidence: high  
reason: 核心目标是驱动四足机器人产生持续向前的运动，动力学是连续力矩控制的步行/跑步，属于典型的运动控制类任务。无到达点或明确终点，前进本身就是唯一主目标，其他高度、姿态是约束和次级优化项。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 连续浮点数（具体精度由环境决定）
- obs[0] (body_z): 身体高度，reward_usable: true，可用作安全高度监控
- obs[1] (quat_w): 身体姿态四元数实部，reward_usable: true，参与直立度计算
- obs[2] (quat_x): 四元数虚部 x，reward_usable: true
- obs[3] (quat_y): 四元数虚部 y，reward_usable: true
- obs[4] (quat_z): 四元数虚部 z，reward_usable: true
- obs[5] (joint_1_angle): 髋关节1角度，reward_usable: true（可做动作平滑或参考姿态）
- obs[6] (joint_2_angle): 踝关节1角度，reward_usable: true
- obs[7] (joint_3_angle): 髋关节2角度，reward_usable: true
- obs[8] (joint_4_angle): 踝关节2角度，reward_usable: true
- obs[9] (joint_5_angle): 髋关节3角度，reward_usable: true
- obs[10] (joint_6_angle): 踝关节3角度，reward_usable: true
- obs[11] (joint_7_angle): 髋关节4角度，reward_usable: true
- obs[12] (joint_8_angle): 踝关节4角度，reward_usable: true
- obs[13] (body_x_velocity): 世界x轴（前向）速度，reward_usable: true，**主前向奖励信号**
- obs[14] (body_y_velocity): 世界y轴（侧向）速度，reward_usable: true，可惩罚侧向
- obs[15] (body_z_velocity): 垂直速度，reward_usable: true，可惩罚剧烈上下起伏
- obs[16] (body_roll_velocity): 滚转角速度，reward_usable: true，用于稳定性惩罚
- obs[17] (body_pitch_velocity): 俯仰角速度，reward_usable: true
- obs[18] (body_yaw_velocity): 偏航角速度，reward_usable: true，转弯惩罚
- obs[19] (joint_1_velocity): 关节1角速度，reward_usable: true（动作平滑/能耗）
- obs[20] (joint_2_velocity): 关节2角速度，reward_usable: true
- obs[21] (joint_3_velocity): 关节3角速度，reward_usable: true
- obs[22] (joint_4_velocity): 关节4角速度，reward_usable: true
- obs[23] (joint_5_velocity): 关节5角速度，reward_usable: true
- obs[24] (joint_6_velocity): 关节6角速度，reward_usable: true
- obs[25] (joint_7_velocity): 关节7角速度，reward_usable: true
- obs[26] (joint_8_velocity): 关节8角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- 连续动作，每个维度范围 [[-1.0, 1.0]]
- action_dim 0: hip_1_torque — 第一髋关节扭矩
- action_dim 1: ankle_1_torque — 第一踝关节扭矩
- action_dim 2: hip_2_torque — 第二髋关节扭矩
- action_dim 3: ankle_2_torque — 第二踝关节扭矩
- action_dim 4: hip_3_torque — 第三髋关节扭矩
- action_dim 5: ankle_3_torque — 第三踝关节扭矩
- action_dim 6: hip_4_torque — 第四髋关节扭矩
- action_dim 7: ankle_4_torque — 第四踝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确的成功终止标志；可默认为“在时间限制（truncation）内始终保持健康姿态”视为一次成功完整运行。
- **failure-like termination**:  
  - body_height_outside_healthy_range：身体高度 z ≤ 0.2（摔倒）或 z ≥ 1.0（过度跃起）。  
  - state_value_outside_finite_range：任何状态值变为 NaN 或 inf，通常代表物理崩溃。  
  两类均直接终止回合，属于硬失败。
- **ambiguous termination**: 无。
- **truncation**: time_limit_reached（达到最大步数），表示存活完全程。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** 
- explicit_failure_flag_available: **false** （`info` 字段为空，不能直接获得终止原因，仅能从环境返回的 `terminated` 或 `truncated` 在 RL 循环中判断，但奖励函数接口不提供这些标志）
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs` (当前状态向量，27维)
- `action` (施加的动作向量，8维)
- `next_obs` (下一状态向量，27维)
- `info` 中明确允许的字段（当前允许字段为空）
- `training_progress` 只有任务提示明确允许时才使用（此处未明确允许，但可忽略）

**禁止使用**：
- `original_reward` （官方奖励被屏蔽）
- 任何被禁止的 info 字段：reward_*, x/y_position 等
- 未声明的 obs 切片（例如假想中有全局 x 坐标，实际上不存在）

## 7. 可用于奖励函数的信号
- **位置相关**：身体高度 `body_z`（obs[0]）；身体姿态四元数 `quat_w,x,y,z`（obs[1:5]），可计算 body_up_z。关节角度（obs[5:13]）可构造姿态正则化或对称性惩罚。
- **速度相关**：前向速度 `body_x_velocity`（obs[13]）——直接前进奖励；侧向速度 `body_y_velocity`（obs[14]）——侧向漂移惩罚；垂直速度 `body_z_velocity`（obs[15]）——起伏惩罚；角速度 `body_roll/pitch/yaw_vel`（obs[16:19]）——稳定性和转向惩罚；关节角速度（obs[19:27]）——动作平滑/能耗。
- **动作/执行器**：`action`（8维扭矩）可用于计算力矩大小、变化量。
- **其他**：训练进度（若环境描述明确需要，但此处未强调，谨慎使用）。

## 8. 不确定或不可用的信号
- 无全局 x/y 位置，无法奖励绝对前进距离。
- 无脚与地面接触力信息，无法使用接触软着陆或 gait pattern 奖励。
- 无显式 success/failure 标志，只能从观测预防。
- 无身体质量、惯性等物理参数，无法计算精确能耗（仅能估算扭矩×速度）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped
  actuator_type: torque_controlled_joints (8 DOF)
  contact_structure: leg-ground contact (not observable)
primary_objectives:
  - forward_velocity: 最大化身体前向速度（或达到目标速度）
  - health: 保持身体高度在 (0.2, 1.0) 区间内，防止摔倒
  - upright: 保持躯干近乎垂直，投影 close to 1
secondary_objectives:
  - lateral_stability: 减小侧向速度
  - yaw_stability: 减小偏航角速度
  - energy_efficiency: 减小关节力矩和运动幅度
  - smoothness: 动作连续，避免高频震荡
main_failure_risks:
  - 高度过低（摔倒）或过高（失控跳跃）导致终止
  - 姿态过度倾斜（翻滚）导致物理崩溃
  - 前进速度停滞，长时间原地踏步
  - 动作剧烈震荡，能耗高且不稳定
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: forward_velocity**  
  purpose: 鼓励持续向前运动，主目标  
  why_required: 任务描述明确要求“must walk or run forward”，否则只是平衡站立  
  usable_signals: body_x_velocity (obs[13])  
  risks: 若权重过大可能导致机器人采取极端前倾姿态或不稳定步态；若仅正向奖励无上限可能导致非物理跳跃

- **role_id: body_height_safety**  
  purpose: 防止身体高度超出健康范围导致提前终止  
  why_required: 高度超出区间直接终止，必须作为硬惩罚或饱和惩罚来保障生存  
  usable_signals: body_z (obs[0])  
  risks: 仅用中心化吸引力可能无法提供足够的边界紧急惩罚，可能需要非对称分段惩罚（对过低区域更重）

- **role_id: upright_orientation**  
  purpose: 维持躯干竖直，防止侧翻或倒立  
  why_required: 连续大幅度倾斜会破坏前进并容易导致摔倒，且隐式帮助对称步态  
  usable_signals: quat (obs[1:5]) 计算的 body_up_z = 1 - 2*(quat_x² + quat_y²)  
  risks: 过度强调可能抑制正常步态中必要的轻微身体摆动，需留出一定容差

### 10.2 条件职责 conditional_roles
- **role_id: lateral_velocity_penalty**  
  condition_to_use: 当侧向速度 |body_y_velocity| 的经验数值明显偏离零（非偶然噪声）时加入。若早期就加入可能过度约束探索  
  usable_signals: body_y_velocity (obs[14])  
  risks: 可能限制初始随机探索时必要的侧向调整，建议较小权重或延迟激活

- **role_id: action_smoothness**  
  condition_to_use: 训练中后期，步态初步形成后，若观察到动作剧烈震荡时引入  
  usable_signals: action (当前动作) 与上一动作对比（需要状态存储，但 reward 接口可访问上一动作？只能在 RL 循环中实现，或利用 next_obs 与 action 结合？



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


```
