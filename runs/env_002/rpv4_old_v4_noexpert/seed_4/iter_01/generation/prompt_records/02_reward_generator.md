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

本环境要求一个平面双足身体在不平坦地形上向前行走，尽可能走得远、走得快，同时降低能量消耗。智能体需要协调两条腿的髋关节与膝关节力矩，形成稳定的双足步态；一旦身体摔倒，回合立即终止。

- 主目标：持续向前通过不平坦地形，最大化前进距离或前进速度。
- 次目标：降低能量消耗，并保持身体直立/稳定、避免摔倒。
- 不应混淆：这不是一个静态到达目标点任务，也不是只要不摔倒即可的纯平衡任务；能量消耗是次要目标，不应取代前进与稳定。

## 2. 任务类型选择

selected_route_id: locomotion_continuous_control

confidence: high

reason: 环境核心是让双足身体通过关节力矩持续前进，通过地形；动作空间为连续关节力矩，观察空间包含身体线速度、躯干姿态、关节状态、触地与地形感知。摔倒终止、到达地形尽头终止。虽然存在能量最小化目标，但它是附属优化，不是与前进目标权重相当且冲突的多目标核心，因此不选 multi_objective_task。

## 3. 观察空间 observation_space

- type: Box
- shape: [24]
- dtype: 未在匿名说明中明确给出；连续 Box 通常为 float32，但本卡不将 float32 当作已确认事实
- obs[0]: hull_angle，主躯干相对竖直方向的倾角，reward_usable: true
- obs[1]: hull_angular_velocity，主躯干角速度，reward_usable: true
- obs[2]: horizontal_velocity，前向/后向线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿 1 髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿 1 髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿 1 膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿 1 膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿 1 地面接触标志，1.0=接触，0.0=未接触，reward_usable: true
- obs[9]: hip2_angle，腿 2 髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿 2 髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿 2 膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿 2 膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿 2 地面接触标志，1.0=接触，0.0=未接触，reward_usable: true
- obs[14]: lidar_0，第 1 条前方 LIDAR 测距距离，reward_usable: true
- obs[15]: lidar_1，第 2 条前方 LIDAR 测距距离，reward_usable: true
- obs[16]: lidar_2，第 3 条前方 LIDAR 测距距离，reward_usable: true
- obs[17]: lidar_3，第 4 条前方 LIDAR 测距距离，reward_usable: true
- obs[18]: lidar_4，第 5 条前方 LIDAR 测距距离，reward_usable: true
- obs[19]: lidar_5，第 6 条前方 LIDAR 测距距离，reward_usable: true
- obs[20]: lidar_6，第 7 条前方 LIDAR 测距距离，reward_usable: true
- obs[21]: lidar_7，第 8 条前方 LIDAR 测距距离，reward_usable: true
- obs[22]: lidar_8，第 9 条前方 LIDAR 测距距离，reward_usable: true
- obs[23]: lidar_9，第 10 条前方 LIDAR 测距距离，reward_usable: true

## 4. 动作空间 action_space

- type: Box
- shape 或 n: [4]，连续动作
- action 0: hip_torque_leg1，施加到腿 1 髋关节的力矩，范围 [-1.0, 1.0]
- action 1: knee_torque_leg1，施加到腿 1 膝关节的力矩，范围 [-1.0, 1.0]
- action 2: hip_torque_leg2，施加到腿 2 髋关节的力矩，范围 [-1.0, 1.0]
- action 3: knee_torque_leg2，施加到腿 2 膝关节的力矩，范围 [-1.0, 1.0]

控制类型为连续关节力矩控制，不是位置/速度目标控制。

## 5. step 与终止条件分析

### 5.1 终止模式

- success-like termination:
  - reached_end_of_terrain：到达地形尽头，可能视为完成任务。
- failure-like termination:
  - body_fallen_over：身体摔倒，回合失败中止。
- ambiguous termination:
  - reached_end_of_terrain 可作为成功，但 step 返回的 info 为空，不能直接确认成功原因；只能从环境事件定义判断。
- truncation:
  - masked step source 中 truncated 恒为 False，未观察到显式时间截断或步数截断。

### 5.2 success/failure 信号可用性

- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无；本环境 info 返回空字典 `{}`，未声明任何信息字段
- forbidden_or_uncertain_info_fields: 任何未声明的 termination_reason、success、failure、timeout 等字段都不可用于 reward

## 6. reward 函数接口契约

函数签名：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：

- obs
- action
- next_obs
- info 中明确允许的字段；根据当前 task spec 与 step source，尚无任何明确允许的 info 字段
- training_progress 仅在 prompt 明确允许时使用；本环境未明确允许，建议不使用

禁止使用：

- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片或任何外部状态

## 7. 可用于奖励函数的信号

- position:
  - 没有全局 x/y 坐标或全局身体位置。
  - 关节角度：hip1_angle, knee1_angle, hip2_angle, knee2_angle。
  - LIDAR 距离可提供前方地形相对距离，但不是身体全局位置。
- velocity:
  - horizontal_velocity：前向速度，契合主目标。
  - vertical_velocity：可辅助判断弹跳或异常上下运动。
  - hull_angular_velocity：躯干角速度，辅助摔倒前兆。
  - hip1_speed, knee1_speed, hip2_speed, knee2_speed：关节角速度。
- orientation:
  - hull_angle：主躯干倾角，与摔倒/直立高度相关。
- contact:
  - leg1_contact, leg2_contact：每条腿是否触地，可用于步态触地节奏。
- action/engine:
  - action 0..3：各关节力矩，可用于能量消耗近似。
- other:
  - lidar_0..lidar_9：前方地形测距，可用于地形感知辅助，但需要谨慎使用其物理含义。

## 8. 不确定或不可用的信号

- 真实能量消耗：没有电流、电压、功率或机械功字段，只能用动作力矩的模/平方等近似。
- 全局前进距离/里程计：观察中无 x,y 位置，不能直接计算精确前进位移。
- 显式摔倒标志：观察中没有直接的 body_fallen_over 标志，只能通过 hull_angle、hull_angular_velocity 等间接推断。
- LIDAR 对应的真实地形高度或障碍物成本：只有距离测量，缺少地面真实高度图或可行走代价定义。
- 接触力/地面反力：观察中只有 binary contact flag，没有法向力或摩擦幅值。
- 地形进度/完成比例：没有全局地形长度或剩余距离字段。

## 9. 专家任务画像 expert_task_profile

```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: planar_bipedal_body
  actuator_type: torque_controlled_hip_knee
  contact_structure: two_legs_discrete_contact_flags
primary_objectives:
  - walk_forward_as_far_and_fast_as_possible
  - cross_uneven_terrain_without_falling
secondary_objectives:
  - minimize_energy_consumption
  - maintain_upright_and_stable_gait
main_failure_risks:
  - falling_over_and_terminating
  - learning_jittery_or_in_place_gait
  - excessive_joint_torque_energy_cost
  - ignoring_terrain_perception_and_tripping
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles

- role_id: forward_progress
  purpose: 鼓励身体向前移动，形成持续前进的双足步态。
  why_required: 这是本任务主目标；没有前向速度信号，智能体可能学会原地平衡或抖动而不前进。
  usable_signals:
    - obs[2] horizontal_velocity
    - next_obs[2] horizontal_velocity
  risks:
    - 仅奖励前向速度可能诱导身体过度前倾或利用跳跃。
    - 需要与稳定性职责配合，避免只优化速度导致摔倒。

- role_id: upright_stability
  purpose: 惩罚主躯干过度倾斜或过快旋转，保持双足步态稳定。
  why_required: body_fallen_over 会终止回合；没有稳定性约束时，前向速度和力矩目标会迅速破坏平衡。
  usable_signals:
    - obs[0] hull_angle
    - obs[1] hull_angular_velocity
    - next_obs[0] hull_angle
    - next_obs[1] hull_angular_velocity
  risks:
    - 过强的直立约束可能限制跑步或前倾步态，使步态僵硬。
    - 早期过度惩罚会导致动作幅度过小，难以探索有效步态。

### 10.2 条件职责 conditional_roles

- role_id: energy_efficiency_penalty
  condition_to_use: 当智能体已出现有效前进但仍伴随高力矩/高关节速度时，或训练中后期需要降低能耗时加入。
  usable_signals:
    - action[0..3] 关节力矩
    - 可选 joint speeds：obs[5], obs[7], obs[10], obs[12]
  risks:
    - 训练早期加入能量惩罚会抑制探索，智能体可能只保持小动作、不前进。
    - 需要保证能量项不压过 forward_progress；否则会导致过慢甚至后退。

### 10.3 慎用/禁用职责 avoid_roles

- role_id: global_position_error
  reason: 观察空间没有全局 x/y 坐标或里程计，无法可靠计算精确位置误差。
  forbidden_or_missing_signals:
    - global_x_position
    - global_y_position
    - odometry_progress

- role_id: official_reward_reconstruction
  reason: 环境理解阶段不重建官方 reward；compute_reward 禁止使用 original_reward 或 official_reward。
  forbidden_or_missing_signals:
    - original_reward
    - official_reward

- role_id: terrain_height_error_shaping_from_lidar
  reason: LIDAR 只给出前方距离而非真实地形高度或可通行代价，直接构造地形高度误差容易引入错误先验。
  forbidden_or_missing_signals:
    - terrain_heightmap
    - obstacle_cost_map
    - exact_terrain_elevation

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | obs[2], next_obs[2] | global_x_position, odometry_progress | dense_state_signal, linear_surplus | 建议与 upright_stability 联合使用 |
| upright_stability | obs[0], obs[1], next_obs[0], next_obs[1] | explicit_fallen_flag | bounded_signal, quadratic_penalty | 关注 hull_angle 与 hull_angular_velocity 的异常幅值 |
| energy_efficiency_penalty | action[0..3], optionally obs[5], obs[7], obs[10], obs[12] | true_mechanical_power, motor_current | quadratic_penalty, bounded_penalty | 仅作为 conditional role，不覆盖主目标 |
| global_position_error | none | global_x/y, odometry_progress | prohibited | 信号缺失，不建议使用 |
| official_reward_reconstruction | none | original_reward, official_reward | prohibited | 禁止使用官方奖励 |
| terrain_height_error_shaping_from_lidar | lidar_0..9 | terrain_heightmap, obstacle_cost_map | prohibited | LIDAR 语义不足以支撑高度误差 |

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 原地抖动或原地摆腿，不前进 | horizontal_velocity 接近 0，但 joint speeds 或 action 幅值较高 | 减小能量 penalty，加入更强的 forward_progress 信号 |
| 频繁摔倒，回合很快结束 | hull_angle 或 hull_angular_velocity 经常超过较大范围，回合长度短 | 加强 upright_stability 职责，或降低前向速度奖励的激进程度 |
| 动作过小导致步态过于保守 | action 幅度长期接近 0，horizontal_velocity 很低 | 减小能量惩罚强度，或增加探索噪声 |
| 前倾过快、利用不稳定姿态冲速度 | horizontal_velocity 高但 hull_angle 长期偏离直立中心 | 调整 upright_stability 的惩罚阈值，使其允许适当倾角但避免极端 |
| 忽视地形感知，遇到坡/障碍后摇晃 | lidar 信号变化与 hull_angle 突变同时出现 | 如使用地形相关辅助 reward，应基于距离变化而非绝对高度；否则不要对 lidar 直接奖惩 |
| 能量项过强导致不学习前进 | 回合表现为小力矩、几乎不移动 | 将 energy_efficiency_penalty 改为 conditional，或根据最低前进速度动态启用 |



# expert_reward_context.md

# Expert Schema Disabled

This run is the w/o Expert Schema ablation. Design from environment facts only; do not use the fixed expert task templates or formula-operator library.

```
