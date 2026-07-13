# Response Record

# 匿名环境理解卡片

## 1. 任务目标
该环境是一个 2D 飞行器／着陆器的轨迹优化任务。主体从视口顶部中央附近起始，并受到一个随机初始力。主要目标是**以最短时间**将主体移动到视口中心的着陆平台上，并**稳定停靠**（即速度归零、姿态平稳、支撑接触）。次要目标是**尽可能少用引擎推力**，以节省燃料。智能体需要学会：
- 朝目标移动（x,y 位置逼近 0）；
- 在接近目标时减速；
- 保持竖直姿态（body_angle ≈ 0）；
- 最终让左、右支撑腿同时接触目标平台（left_support_contact=1 且 right_support_contact=1）且主体静止。

**不要混淆**：仅快速到达而不稳定停靠不算成功；仅节省燃料而不及时到达也不算成功。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定的目标位置（着陆平台）并稳定停靠，附属有速度、姿态、燃料消耗的优化，但主要成功标准是到达并停靠，符合导航目标到达类的定义。无多目标冲突权重。

动力学子类型：goal_approach_and_soft_contact  
（强调逼近目标、减速、稳定垂直姿态与安全接触）

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（接触标志为 0.0/1.0）
- obs[0]: x_position，相对于目标着陆平台中心的水平坐标，reward_usable: true
- obs[1]: y_position，相对于着陆平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，无推力，主体自由漂移
- action 1: left_orientation_engine，启动一个姿态引擎（产生偏转力矩和微小推力）
- action 2: main_engine，启动主引擎（产生沿机体方向的推力）
- action 3: right_orientation_engine，启动另一侧姿态引擎（相反偏转力矩）

注意：动作是离散的，每个时间步只能选择一种引擎或不做任何操作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能表示主体已静止并停靠在平台上（左右支撑接触均为 1，速度很小，姿态稳定）。这很可能就是成功到达并停靠的结局。
- failure-like termination: crash_or_body_contact（碰撞或不当身体接触）、horizontal_position_outside_viewport（水平位置超出视口范围）
- ambiguous termination: body_not_awake_or_settled 在没有支撑接触时也可能因能量耗尽而“沉睡”，此时可视为失败；但由于环境没有提供更细的 info，我们只能依据接触和位置判断。
- truncation: 无明确时间截断，但可以限制最大步数，本环境未说明。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）
- explicit_failure_flag_available: false（无 failure 字段）
- allowed_info_fields: {} （空字典，无可用的额外字段）
- forbidden_or_uncertain_info_fields: 所有未在 step 源码中出现的字段，例如 "success", "failure", "reason" 等均不可用。终止信号仅由 terminated 布尔值给出，且 masked step source 未提供终止原因分离信息，因此我们不能直接依赖 terminated 的标签来区分成功/失败。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- obs 和 next_obs 的完整 8 维向量（或单步中分别使用）
- action（整数动作）
- info 中明确允许的字段（目前已知 info 为空）
- training_progress：仅在环境明确允许或作为额外指导时才使用，当前未声明，因此**不建议使用**。

**禁止使用：**
- original_reward（官方奖励已被掩盖，不可访问）
- 任何未在 step 源码中出现的 info 字段
- 任何试图从 terminated 标志直接推断成功/失败的方法（因为终止原因不可辨）

## 7. 可用于奖励函数的信号
- position: x_position, y_position（均相对于着陆平台）
- velocity: x_velocity, y_velocity
- orientation: body_angle, angular_velocity
- contact: left_support_contact, right_support_contact
- action/engine: 当前动作（可用来惩罚引擎使用）
- other: 从 obs 可计算的状态量（如到目标的距离、速度模、角度绝对值等）

## 8. 不确定或不可用的信号
- 官方奖励 original_reward：不可用
- 显式成功/失败标志：不可用
- 视口边界的具体阈值：未提供，只能从位置范围经验推断
- 燃料消耗量或剩余能量：未显式给出，动作本身只能表示“哪个引擎工作”，不能得到推力大小或燃料消耗的具体数值，只能定性惩罚引擎动作。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 飞行器／着陆器（可视为月球着陆器简化版）
  actuator_type: 离散推力器（主引擎 + 两个姿态引擎）
  contact_structure: 两腿支撑，左右独立接触检测
primary_objectives:
  - 到达目标着陆平台（x≈0, y≈0）
  - 稳定停靠（速度≈0，姿态角≈0）
  - 左右支撑腿同时接触平台
secondary_objectives:
  - 最小化总引擎使用步数（节省燃料）
  - 尽可能快速到达（隐含在密集奖励设计中）
main_failure_risks:
  - 水平飞出视口边界
  - 发生碰撞或与平台以外的物体接触
  - 静止在非目标区域或未同时保持双腿接触
  - 因过度使用引擎导致 energy/time-out 耗尽而沉睡
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_distance_reduction
  purpose: 鼓励 agent 减小与目标平台的水平与垂直距离，驱动其朝平台移动
  why_required: 任务是导航到达，必须引导位置趋近
  usable_signals: [x_position, y_position, next_x_position, next_y_position]
  risks: 可能鼓励高速冲向平台而忽略减速，需要配合速度惩罚

- role_id: soft_landing_velocity
  purpose: 当 agent 接近平台时惩罚较大的垂向和水平速度，促进软着陆
  why_required: 稳定停靠要求速度趋零，避免撞击
  usable_signals: [x_velocity, y_velocity, distance_to_target]
  risks: 过早压低速度会导致学习缓慢；需要根据距离调控惩罚强度

- role_id: upright_orientation
  purpose: 惩罚机体倾角偏离竖直，保持稳定姿态
  why_required: 安全接触需要竖直姿态，且倾斜易导致侧向滑动或翻倒
  usable_signals: [body_angle, angular_velocity]
  risks: 若过于敏感可能抑制必要的姿态调整动作

- role_id: dual_leg_contact
  purpose: 最终停在平台上时奖励双腿同时接触
  why_required: 环境成功条件隐含“安全接触”，双腿接触是着陆完成的信号
  usable_signals: [left_support_contact, right_support_contact]
  risks: 仅在终端时使用，或与距离结合以避免在平台外误触发

### 10.2 条件职责 conditional_roles
- role_id: fuel_penalty
  condition_to_use: 训练初期可以不使用（以免阻碍探索），中后期或当速度/位置已部分达标时引入
  usable_signals: [action (0: 无引擎, 1:左姿态, 2:主引擎, 3:右姿态)]
  risks: 连续动作下若惩罚过度可能导致 agent 不敢使用引擎，无法到达目标

- role_id: fast_arrival_bonus
  condition_to_use: 如果有步数限制或训练进度参数可用，可对在较少步数内完成给予一次性奖励，否则不强制
  usable_signals: [step count, termination reach with dual contact]
  risks: 环境未提供步数计数器，只能通过自定义 wrapper 引入；且可能鼓励不安全的莽撞行为

### 10.3 慎用/禁用职责 avoid_roles
- role_id: termination_success_only
  reason: 无法从 info 或 terminated 直接获取成功标志，故不能单纯依赖“终局成功=大奖”的设计，必须将成功信号分解为可观测状态。
  forbidden_or_missing_signals: [explicit success flag]

- role_id: safety_collision_penalty
  reason: crash_or_body_contact 触发终止，但我们无法在奖励步中获得该信息（仅在终止时），且无 info 区分碰撞类型，贸然使用可能导致噪声。可在终止后通过 obs 状态推断，但需谨慎。
  forbidden_or_missing_signals: [crash flag per step]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_distance_reduction | x_position, y_position | - | dense_state_signal, quadratic_penalty | 可用 -distance 或 -(x^2+y^2) |
| soft_landing_velocity | x_velocity, y_velocity, distance | - | bounded_signal, scale_by_proximity | 建议当 distance<threshold 时打开速度惩罚 |
| upright_orientation | body_angle, angular_velocity | - | quadratic_penalty, absolute_penalty | 角度惩罚即可，可加角速度阻尼 |
| dual_leg_contact | left_support_contact, right_support_contact | - | logical_and, sparse_bonus | 仅在两条腿同时为 1 时给正向奖励 |
| fuel_penalty | action | - | categorical_penalty | 对非零动作给予小惩罚 |
| fast_arrival_bonus | step count | （需 wrapper） | sparse_bonus | 条件使用，当前未启用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过早坠毁（水平飞出视口） | x_position 绝对值快速增长至边界 | 增加对 x_position 的惩罚系数，或加入边界接近警告 |
| 到达目标但速度过大导致弹跳 | 终局时速度非零，双腿接触闪烁 | 加强软着陆速度惩罚，且使用近距离调度 |
| 姿态失控翻滚 | body_angle 和 angular_velocity 持续偏离 | 增大姿态惩罚，或对姿态引擎滥用增加约束 |
| 只悬停不下降（y_position 保持正值） | y_position 长时间不减小 | 增加对 y 方向趋近的奖励权重 |
| 仅单腿接触停滞 | one leg contact=1, other=0 且主体不动作 | 引导双腿接触奖励，或在接近时引入微调引导 |
| 过度使用引擎消耗燃料 | action 频繁为非零 | 引入适度的 fuel_penalty，训练后期逐步增强 |
