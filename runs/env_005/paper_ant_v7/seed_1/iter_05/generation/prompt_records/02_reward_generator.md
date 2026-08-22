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
这是一个连续控制运动任务。一个 3D 四足机器人拥有四条腿和八个扭矩控制关节，必须持续向前行走或奔跑并保持身体直立。机器人的身体高度必须保持在健康的范围（0.2 到 1.0）内；低于最小高度（趴下）或超过最大高度（弹射）均会终止回合。策略的主要目标是产生稳定、持续的前向运动，而不仅仅是保持直立不倒。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是让四足机器人在平面地形上产生持续的前进运动。速度、姿态和高度等附属约束服务于这一前向移动主任务，不构成权重相当的多目标冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float32（默认连续环境）
- obs[0]: body_z，身体垂直高度，reward_usable: true（用于生存/高度约束）
- obs[1]: quat_w，身体朝向四元数 w 分量，reward_usable: true（用于计算直立投影）
- obs[2]: quat_x，四元数 x 分量，reward_usable: true（用于计算直立投影）
- obs[3]: quat_y，四元数 y 分量，reward_usable: true（用于计算直立投影）
- obs[4]: quat_z，四元数 z 分量，reward_usable: true（用于计算直立投影）
- obs[5]: joint_1_angle，第一个髋关节角度，reward_usable: true（用于姿态/能耗惩罚）
- obs[6]: joint_2_angle，第一个踝关节角度，reward_usable: true
- obs[7]: joint_3_angle，第二个髋关节角度，reward_usable: true
- obs[8]: joint_4_angle，第二个踝关节角度，reward_usable: true
- obs[9]: joint_5_angle，第三个髋关节角度，reward_usable: true
- obs[10]: joint_6_angle，第三个踝关节角度，reward_usable: true
- obs[11]: joint_7_angle，第四个髋关节角度，reward_usable: true
- obs[12]: joint_8_angle，第四个踝关节角度，reward_usable: true
- obs[13]: body_x_velocity，世界坐标系 x 方向线速度（前进速度），reward_usable: true（前向进度主信号）
- obs[14]: body_y_velocity，世界 y 方向线速度（侧向速度），reward_usable: true（侧向漂移惩罚）
- obs[15]: body_z_velocity，世界 z 方向线速度（垂直速度），reward_usable: true（降低震动）
- obs[16]: body_roll_velocity，滚转角速度，reward_usable: true（姿态稳定）
- obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true
- obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true（保持方向）
- obs[19]: joint_1_velocity，第一个髋关节角速度，reward_usable: true（能耗惩罚）
- obs[20]: joint_2_velocity，第一个踝关节角速度，reward_usable: true
- obs[21]: joint_3_velocity，第二个髋关节角速度，reward_usable: true
- obs[22]: joint_4_velocity，第二个踝关节角速度，reward_usable: true
- obs[23]: joint_5_velocity，第三个髋关节角速度，reward_usable: true
- obs[24]: joint_6_velocity，第三个踝关节角速度，reward_usable: true
- obs[25]: joint_7_velocity，第四个髋关节角速度，reward_usable: true
- obs[26]: joint_8_velocity，第四个踝关节角速度，reward_usable: true

注：所有维度均可用于奖励函数。

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- continuous：每维取值范围 [-1.0, 1.0]，对应归一化扭矩指令
- action 0: hip_1_torque，第一个髋关节扭矩
- action 1: ankle_1_torque，第一个踝关节扭矩
- action 2: hip_2_torque，第二个髋关节扭矩
- action 3: ankle_2_torque，第二个踝关节扭矩
- action 4: hip_3_torque，第三个髋关节扭矩
- action 5: ankle_3_torque，第三个踝关节扭矩
- action 6: hip_4_torque，第四个髋关节扭矩
- action 7: ankle_4_torque，第四个踝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确的成功终止。回合通常因时间截断 (truncation) 结束，若在截断前未触发失败终止，可视为存活至结束，但并不保证已完成前向移动目标。
- failure-like termination:
  - body_height_outside_healthy_range：身体高度 z 不在 (0.2, 1.0) 范围内（趴下或飞起），对应失效。
  - state_value_outside_finite_range：观测中出现 NaN 或 inf，数值不稳定，应为失败。
- ambiguous termination: 无机械定义的模糊终止。
- truncation: 时间限制到达，属于正常结束，存活且未触发以上失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空且不允许使用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何官方奖励项 reward_forward、reward_ctrl 等以及 x/y 位置等。终止原因无法从 info 中提取。

备注：尽管没有显式 success/failure 标记，但可以通过观测到的 next_obs[0] (body_z) 是否在 (0.2,1.0) 内以及数值是否有限来推断失败终止，但这需要 reward 函数在训练时能访问 terminated 标志（即使如此，此处分析仅基于允许使用的 obs/action/next_obs/info，不能假设 terminated 可作为输入）。在 stateless reward 设计中无法直接使用 termination 信号，但可以通过 body_z 和有限性检查接近该信息。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- `obs`：全部 27 维观测均可使用。
- `action`：全部 8 维动作均可使用。
- `next_obs`：与 obs 同结构的观测，可用于计算变化或检查终止条件相关的信号（如高度、有限性）。
- `info`：不允许使用任何字段（info 为空且被禁止）。
- `training_progress`：仅在 prompt 明确允许时使用，本环境未提及，故谨慎使用或不用。

禁止使用：
- `original_reward`（已遮蔽）。
- 任何来自 info 的官方奖励项或位置信息。
- 任何未在上述观察空间中声明的隐含信息。

## 7. 可用于奖励函数的信号
- 位置/姿态信号：
  - body_z (obs[0])：可作为生存/高度奖励的基础。
  - 姿态直立度：从四元数 obs[1:5] 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)，值域 [-1, 1]，1 表示完全直立。可用于 orientation 奖励。
- 速度信号：
  - body_x_velocity (obs[13])：前进速度，直接反映任务进度。
  - body_y_velocity (obs[14])：侧向速度，可用于惩罚漂移。
  - body_z_velocity (obs[15])：垂直速度，惩罚跳跃或剧烈振动。
  - body_roll_velocity (obs[16])、body_pitch_velocity (obs[17])、body_yaw_velocity (obs[18])：惩罚姿态晃动。
- 关节角度/角速度信号：
  - 关节角度 obs[5:13]：可惩罚过大关节偏离。
  - 关节角速度 obs[19:27]：可惩罚抖动或高能耗。
- 动作/控制信号：
  - action[0:8]：扭矩指令，可惩罚大幅动作、动作变化率（若结合上一动作）或直接作为能耗惩罚。
- 其他：可通过 next_obs 推断是否进入危险区域（body_z 过低/过高）进行惩罚，但注意 reward 是 step-wise 且无累积状态。

## 8. 不确定或不可用的信号
- 绝对水平位置（x, y）：不可用，禁止使用。
- 接触力/触地信息：不可用，环境中无相关观测。
- 官方奖励分量（reward_forward, reward_ctrl 等）：不可用，禁止。
- 明确的 success/failure 标记：不可用，info 为空。
- 躯干俯仰/滚转角度：不能直接从 27 维观测中获得欧拉角（除非从四元数转换，但需要额外计算），但可间接通过四元数投影获得 body_up_z。
- 动作变化率（jerk）：需要存储上一动作，而 compute_reward 为无状态函数，无法直接获取，除非将上一动作作为额外状态传入，但这不是标准接口。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D quadruped (four legs)
  actuator_type: eight torque-controlled joints (hip + ankle per leg)
  contact_structure: legs interact with ground, but no contact force/state observable
primary_objectives:
  - maximize forward velocity (body_x_velocity) while remaining stable
  - maintain body height within healthy range [0.2, 1.0] to avoid termination
  - keep body upright (body_up_z close to 1)
secondary_objectives:
  - minimize lateral drift (body_y_velocity magnitude)
  - suppress excessive body angular rates (roll, pitch, yaw) for smooth gait
  - reduce energy consumption (action magnitudes, joint velocities)
  - avoid large joint angle deviations that could lead to instability
main_failure_risks:
  - body_z falls below 0.2 (collapse), or exceeds 1.0 (launch)
  - orientation flips (body_up_z near -1) leads to upside-down state then often triggers height violation
  - numerical divergence (NaNs/infs) due to unstable joint configurations
  - agent learns to stay still to avoid termination but fails to make forward progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward  
  purpose: 激励机器人向前移动，直接体现任务核心目标。  
  why_required: 任务要求稳定前向运动，如果没有这个奖励，智能体可能学会静止不动以保持存活。  
  usable_signals: [obs[13] body_x_velocity]  
  risks: 若奖励设计不当（如单纯使用原始速度），可能导致机器人采用危险的跑步策略（易翻倒）或学习“冲刺然后摔倒”的局部最优。需要与其他稳定职责结合。

- role_id: survival_and_height_reward  
  purpose: 维持身体高度在健康区间内，避免摔倒或飞起终止。  
  why_required: 终止条件直接与高度相关，必须让智能体学会保持安全高度，否则无法存活。  
  usable_signals: [obs[0] body_z]  
  risks: 若只奖励待在某个固定高度，可能抑制前向运动，需要设计为仅惩罚偏离安全区的部分。

- role_id: upright_orientation_reward  
  purpose: 保持躯干大致直立，防止翻转。  
  why_required: 翻倒会迅速导致高度违规或失去前进能力；直立是稳定运动的基础。  
  usable_signals: [obs[1:5] quaternion; 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)]  
  risks: 四足动物在快速奔跑时可能有自然的俯仰，对直立约束过严可能妨碍高速步态，可通过适度容忍稍低的 body_up_z 或分级奖励缓解。

### 10.2 条件职责 conditional_roles
- role_id: lateral_drift_penalty  
  purpose: 惩罚侧向移动，使机器人沿正前方行走。  
  condition_to_use: 当观测到 body_y_velocity 偏离 0 时作用。最好在中后期加入，早期可放宽以保证探索。  
  usable_signals: [obs[14] body_y_velocity]  
  risks: 若惩罚过重可能限制机器人必要的侧向平衡调整步态，导致僵硬行走，需要系数调优。

- role_id: angular_rate_penalty  
  purpose: 减小身体滚转、俯仰、偏航角速度，让运动更平滑，降低摔倒风险。  
  condition_to_use: 始终可用，但初期可降低权重，待基本行走出现后再加强。  
  usable_signals: [obs[16] body_roll_velocity, obs[17] body_pitch_velocity, obs[18] body_yaw_velocity]  
  risks: 过大会抑制动态平衡所需的微小角速度，使运动过于僵硬；偏航角速度惩罚在需要转弯时可能不合适，但本任务只需直线前进故无问题。

- role_id: action_regularization  
  purpose: 降低扭矩指令能量，鼓励节能、连贯的动作。  
  condition_to_use: 当基本前进行为建立后加入更有效，早期可能阻碍探索。  
  usable_signals: [action[0:8]]  
  risks: 太



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
- best_score_so_far: 13.550

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_penalty + forward_gated + lateral_penalty | 3 | 13.550 | -464.620 | unsolved |
| action_penalty + forward_gated | 1 | 2.910 | 2.910 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
