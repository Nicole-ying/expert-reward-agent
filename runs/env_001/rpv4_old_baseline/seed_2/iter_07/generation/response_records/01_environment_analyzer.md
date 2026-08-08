# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主体是一个2D车体（类似着陆器），起始于画面上方中心附近，带有随机初始力。智能体需要尽快飞到画面中央的目标着陆垫，稳定降落并停止，同时尽量少使用引擎推力。核心目标是**安全到达目标垫并稳定停靠**，次要目标是最小化时间和燃料消耗。不应混淆的是：着陆成功不等于单纯到达垫子上方，还需低速、姿态垂直、接触平稳。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务主目标明确为“到达中央目标垫并停靠”，符合导航到目标点的定义，附属目标（快速、省燃料）为性能优化，不构成同等权重冲突的多目标，因此不属于multi_objective_task。动力学表现为接近目标、减速、姿态稳定后软接触，属于goal_approach_and_soft_contact子类型。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 连续浮点（具体由环境决定，通常float32）
- obs[0]: x_position（相对目标垫的水平位置），reward_usable: true
- obs[1]: y_position（相对垫高度的垂直位置，0表示与垫齐平，正为上方），reward_usable: true
- obs[2]: x_velocity（水平线速度），reward_usable: true
- obs[3]: y_velocity（垂直线速度），reward_usable: true
- obs[4]: body_angle（机体姿态角），reward_usable: true
- obs[5]: angular_velocity（角速度），reward_usable: true
- obs[6]: left_support_contact（左支撑腿接触标志，1.0接触，0.0未接触），reward_usable: true
- obs[7]: right_support_contact（右支撑腿接触标志，1.0接触，0.0未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，无任何引擎工作，仅靠惯性/重力演化
- action 1: left_orientation_engine，点燃左侧姿态引擎，主要用于纠正或产生逆时针旋转倾向
- action 2: main_engine，点燃主引擎，产生主要推力（通常向上方向，抵抗重力/减速）
- action 3: right_orientation_engine，点燃右侧姿态引擎，产生顺时针旋转倾向

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 身体稳定垫上停靠（由body_not_awake_or_settled可能触发），暗示成功着陆。
- failure-like termination: crash_or_body_contact（剧烈碰撞）、horizontal_position_outside_viewport（飞出视野，位置丢失）。
- ambiguous termination: body_not_awake_or_settled 仅说明身体不再移动或进入休眠，未明确是成功还是失败，但结合任务目标，大概率是成功，然而无法直接判断具体原因。
- truncation: 提供的step源码中未出现时间截断，但真实环境可能有步数上限，此处未暴露。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info字典为空，无任何字段可用）
- forbidden_or_uncertain_info_fields:
  - 任何未在allowed_info_fields中声明的字段
  - 官方奖励信号 original_reward 禁止直接使用或重构
  - 环境内部终止原因代码不可见

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs: 当前步观测
- action: 当前步动作
- next_obs: 下一步观测（可用于计算差分、判断着陆事件等）
- info: 仅允许使用明确声明的字段（当前为空映射）
- training_progress: 仅在任务额外说明允许时用于退火调度，此处允许使用以可能实现课程学习，但不可直接作为时间进度指标

禁止使用：
- original_reward（官方奖励被屏蔽）
- 任何未在允许列表中的info字段
- 未标记的obs切片（但全部obs均标记可用，故无此问题）
- 环境内部步数或时钟（未提供）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action (0-3) → 可判别是否使用主引擎、姿态引擎
- other: 可派生量如与目标的距离（√(x²+y²)）、是否接触垫（任意contact flag>0）、垂直速度符号、角度绝对值等。

## 8. 不确定或不可用的信号
- 碰撞/坠毁事件：crash_or_body_contact 只在终止时产生，且无法通过obs直接判断是否发生碰撞（除非结合位置/速度急剧变化，但无可靠阈值），因此不能用作奖励函数中的确定性信号。
- 任务完成标志：没有显式成功标志，body_not_awake_or_settled 是终止理由之一，但在step中未被传递为信息字段。
- 时间/步数计数：无可用变量，无法在奖励函数内获知当前episode已进行的步数，因此不能奖励“快速到达”。
- 视口边界值：不知道horizontal_position_outside_viewport的具体判定阈值，无法安全地在奖励中基于位置硬判定。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: vehicle/lander
  actuator_type: one main vertical engine + two lateral attitude engines
  contact_structure: two support legs, each with a binary contact flag
primary_objectives:
  - 到达目标垫上方并稳定降落（软接触）
secondary_objectives:
  - 最小化燃料消耗（减少引擎使用）
  - 保持姿态稳定性（避免翻滚）
  - 在可行前提下尽快到达（此处因缺少步数信号不能奖励时间）
main_failure_risks:
  - 高速撞击垫子（crash）
  - 飞出视口丢失
  - 着陆姿态过大导致倾覆
  - 过度使用引擎浪费燃料且姿态不稳
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: proximity_to_target  
  purpose: 引导机体向目标垫移动，奖励接近目标的行为  
  why_required: 任务核心是到达目标点，无此奖励无法完成导航  
  usable_signals: [x_position, y_position, 可选派生距离]  
  risks: 过度奖励靠近但高速撞击，需结合速度信号抑制

- role_id: soft_landing_on_pad  
  purpose: 当任一支撑腿接触垫时，要求低速且姿态垂直，鼓励安全着陆  
  why_required: 仅到达目标垫上方不足以成功，需要稳定接触停靠  
  usable_signals: [left_support_contact, right_support_contact, y_velocity, x_velocity, body_angle]  
  risks: 易导致过早接触奖励，需结合接触触发条件；可能使智能体贪图接触但忽略周围姿态

- role_id: orientation_stability  
  purpose: 惩罚过大的倾斜角度或角速度，防止翻滚失控  
  why_required: 姿态不稳定会导致无法精确着陆甚至崩溃  
  usable_signals: [body_angle, angular_velocity]  
  risks: 过度惩罚可能阻碍必要的姿态调整机动，需适度放宽

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency  
  condition_to_use: 始终可用，但系数可随 training_progress 增大以鼓励后期更省燃料  
  usable_signals: [action]（惩罚主动使用引擎，特别是主引擎）  
  risks: 初期过度惩罚引擎使用会抑制探索和必要的减速机动，需渐进加重或仅在接近目标时激活

- role_id: time_pressure_soft  
  condition_to_use: 若环境未来提供步数指标或允许通过训练步数计算“剩余时间”信号时可用，当前**因无可用信号，暂无法实现**  
  usable_signals: [缺失episode步数信息]  
  risks: 无信号直接导致无法实现

### 10.3 慎用/禁用职责 avoid_roles
- role_id: crash_penalty  
  reason: 缺少显式碰撞指示器，且无法从obs可靠推断碰撞事件；强行使用易造成错误负奖励  
  forbidden_or_missing_signals: [crash_event, 明确失败标志]

- role_id: out_of_bounds_penalty_early  
  reason: 视口边界值未知，无法安全地在奖励函数中基于位置硬判定越界风险  
  forbidden_or_missing_signals: [boundary limits]

- role_id: explicit_success_bonus  
  reason: 无显式成功标志可用，无法安全发放成功奖励，容易错误奖励非成功终止状态  
  forbidden_or_missing_signals: [success_flag]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| proximity_to_target | x_position, y_position | - | dense_state_signal (e.g., negative distance, bounded_signal), gaussian | 可从next_obs计算新距离给予差分奖励 |
| soft_landing_on_pad | contact flags, y_velocity, x_velocity, body_angle | - | conditional_reward (if contact), bounded_signal, threshold_gate | 要求低y速度，小x速度，小角度 |
| orientation_stability | body_angle, angular_velocity | - | quadratic_penalty, bounded_signal | 可同时对角度绝对值与角速度施加惩罚 |
| fuel_efficiency | action | - | action_penalty (counter the use of engine actions) | 惩罚action=1,2,3，或仅惩罚main_engine(2) |
|
