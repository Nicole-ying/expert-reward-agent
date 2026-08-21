# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；

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
```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code

**2. components: dict**，记录 individual reward components。

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个匿名 2D 双足运动任务。智能体控制一个两腿躯干，通过髋、膝共 4 个连续力矩动作，在不规则地形上稳定向前行走。任务主目标是尽可能远、尽可能快地前进；必要健康约束是保持身体不倒，因为摔倒会立即终止回合。次要目标是降低能量消耗或关节力矩成本。

该任务不应被理解为纯平衡任务：保持平衡是完成前进目标的条件，而不是最终目标。也不应把“站立不动但保持平衡”当作成功。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

confidence: high

reason: 核心目标是控制腿部关节产生连续步态，在不规则地形上持续前进；动作是连续力矩；没有固定到达点，没有抓取操作，也没有多个权重相当且冲突的核心目标。摔倒只是失败/终止约束，不构成独立的多目标核心。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: 未显式给出，按常规推断为 float32
- obs[0]: hull_angle，躯干相对竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，躯干角速度，reward_usable: true
- obs[2]: horizontal_velocity，前向/后向线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，leg1 髋关节角度，reward_usable: true
- obs[5]: hip1_speed，leg1 髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，leg1 膝关节角度，reward_usable: true
- obs[7]: knee1_speed，leg1 膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，leg1 接地标志，1.0=接地，0.0=未接地，reward_usable: true
- obs[9]: hip2_angle，leg2 髋关节角度，reward_usable: true
- obs[10]: hip2_speed，leg2 髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，leg2 膝关节角度，reward_usable: true
- obs[12]: knee2_speed，leg2 膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，leg2 接地标志，1.0=接地，0.0=未接地，reward_usable: true
- obs[14..23]: lidar_0..9，10 个前方地形 LIDAR 测距读数，reward_usable: true，但属于间接/地形预览信号，使用需谨慎

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: 每个关节 [-1.0, 1.0]
- action_dim 0: hip_torque_leg1，施加在 leg1 髋关节的力矩
- action_dim 1: knee_torque_leg1，施加在 leg1 膝关节的力矩
- action_dim 2: hip_torque_leg2，施加在 leg2 髋关节的力矩
- action_dim 3: knee_torque_leg2，施加在 leg2 膝关节的力矩

动作是连续力矩控制，不存在离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain，到达地形终点，可能表示成功，但未提供显式成功标志。
- failure-like termination: body_fallen_over，躯干摔倒，任务失败，应被惩罚。
- ambiguous termination: step 中 terminated=True 可能来自 body_fallen_over 或 reached_end_of_terrain；info={} 为空，无法直接从 info 区分原因。
- truncation: 给定 step 返回 truncated=False；未显示外部时间限制。不过真实运行环境可能包了 TimeLimit wrapper，但当前信息中不可依赖。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无 / []
- forbidden_or_uncertain_info_fields: success、failure、termination_reason、reached_goal、any info key not explicitly declared

必须注意：终止原因只能从下一状态或状态轨迹推断，不能假设 info 中存在成功/失败字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段；本环境当前为空，因此 info 不应使用
- training_progress 仅在外部 prompt 明确允许时才使用；当前 prompt 未明确允许，因此默认禁止使用

禁止使用：
- original_reward
- official_reward
- 任何未声明的 info 字段
- 任何未声明的 obs 切片
- 不得尝试重构被 mask 的官方 reward

## 7. 可用于奖励函数的信号
- position:
  - 无绝对 root x/y 位置
  - 无累计行进距离
  - lidar 只提供前方地形距离，不提供绝对位置
- velocity:
  - horizontal_velocity，obs[2]
  - vertical_velocity，obs[3]
  - hull_angular_velocity，obs[1]
  - hip1_speed，obs[5]
  - knee1_speed，obs[7]
  - hip2_speed，obs[10]
  - knee2_speed，obs[12]
- orientation:
  - hull_angle，obs[0]
- contact:
  - leg1_contact，obs[8]
  - leg2_contact，obs[13]
- action/engine:
  - action[0..3] 力矩向量，可用于能量/平滑惩罚
- other:
  - hip1_angle, knee1_angle, hip2_angle, knee2_angle
  - lidar_0..9，obs[14..23]，可用于地形感知 shape，但不是直接前进目标信号

## 8. 不确定或不可用的信号
- 绝对位置 / 已行进距离
- explicit success flag
- explicit failure flag
- 地面反作用力或摩擦系数
- 真实能量消耗或执行器功率；可用 torque * joint_speed 近似，但未直接给出
- 参考步态轨迹或目标关节角度序列
- 地形的完整高度图；仅有前方 LIDAR 读数
- info 中的任何字段，当前 info 为空

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: planar two-legged walker with torso and two legs
  actuator_type: continuous torque motors at hip and knee joints
  contact_structure: two leg ground contact flags (leg1_contact, leg2_contact)
primary_objectives:
  - maximize forward progress over uneven terrain (as far/fast as possible)
  - maintain stable bipedal gait and avoid falling
secondary_objectives:
  - minimize energy consumption / torque cost
main_failure_risks:
  - torso pitch instability leading to fall
  - stationary or too-slow gait due to over-penalizing action/energy
  - jerky or aggressive gait that wastes energy or destabilizes on uneven terrain
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 奖励智能体沿前进方向移动的速度或正向位移代理。
  why_required: 任务主目标是尽可能远、尽可能快前进。
  usable_signals: [horizontal_velocity, next_obs[2]]
  risks:
    - 只奖励速度可能诱发震荡或摔倒，必须与稳定性职责配合。
    - 没有绝对位置，无法直接奖励累计距离。
    - 需要校准 horizontal_velocity 的正方向是前进方向。

- role_id: balance_and_fall_prevention
  purpose: 惩罚躯干过度倾斜、剧烈旋转或异常垂直运动，避免摔倒。
  why_required: 摔倒会终止回合，阻碍前进目标。
  usable_signals: [hull_angle, hull_angular_velocity, vertical_velocity]
  risks:
    - 对小幅 hull_angle 惩罚过重会限制正常步态。
    - 只用 terminal fall 进行奖励不可行，因为没有显式 fall flag。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 在已能驱动前进后加入；或检测到动作过大、能耗过高时启用。作为低权重次级正则，不应压过前进目标。
  usable_signals: [action[0..3], joint_speeds]
  risks:
    - 权重过高会导致智能体站立不动或迈步过小。
    - 与 forward_progress 冲突时，必须降低该职责权重。

- role_id: gait_symmetry_or_contact_pattern
  condition_to_use: 当观察到一条腿闲置、拖行或步态明显不对称时启用。
  usable_signals: [leg1_contact, leg2_contact, hip1_angle, hip2_angle, knee1_angle, knee2_angle, joint_speeds]
  risks:
    - 强制对称可能限制自然步态。




# expert_reward_context.md

# Expert Schema Disabled

This run is the w/o Expert Schema ablation. Design from environment facts only; do not use the fixed expert task templates or formula-operator library.

```
