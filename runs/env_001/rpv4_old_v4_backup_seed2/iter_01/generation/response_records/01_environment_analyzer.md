# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主体是一个2D类车辆刚体，从视口上方中央附近出发，带有随机初始作用力。任务是**尽快到达并稳定停靠在中央的目标垫上，同时尽可能减少发动机推力消耗**。理想行为是：平滑接近目标、降低线速度、保持垂直姿态（零角度）、以低速和正确朝向使左右两个支撑点同时接触目标垫并固化。次要目标包含节省燃料和/或快速完成，但不应混淆为主目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是“到达并稳定在目标垫上”，这是典型的到达型任务；附属有速度、姿态、能耗等约束，但到达仍是主目标，不含多个权重相当且冲突的核心目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (默认) 或 float32
- obs[0]: x_position — 相对目标垫的水平坐标 (reward_usable: true)
- obs[1]: y_position — 相对目标垫垫面高度的垂直坐标 (reward_usable: true)
- obs[2]: x_velocity — 水平线速度 (reward_usable: true)
- obs[3]: y_velocity — 垂直线速度 (reward_usable: true)
- obs[4]: body_angle — 车身朝向角 (reward_usable: true)
- obs[5]: angular_velocity — 角速度 (reward_usable: true)
- obs[6]: left_support_contact — 左支撑脚接触标志，接触为1.0 (reward_usable: true)
- obs[7]: right_support_contact — 右支撑脚接触标志，接触为1.0 (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，靠惯性滑行
- action 1: left_orientation_engine — 点燃左姿态发动机（提供转向力矩）
- action 2: main_engine — 点燃主发动机（提供向上/向前的推力）
- action 3: right_orientation_engine — 点燃右姿态发动机（提供相反力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact: 主体发生碰撞或非预期的身体接触（很可能为失败，但不排除目标垫接触也属于body_contact，需结合后续稳定信号区分）
- horizontal_position_outside_viewport: 水平位置超出视口边界（明确失败）
- body_not_awake_or_settled: 主体不再活动或已稳定（speed/angular接近零且可能接触垫）；此条件可能是成功（settled on pad）或失败（settled elsewhere or falling asleep）
- 没有显式截断（truncation）条件提及，但实际环境可能有最大步数限制，目前未在信息中给出。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info为空字典)
- explicit_failure_flag_available: false
- allowed_info_fields: （当前step返回的info = {}，无可用字段）
- forbidden_or_uncertain_info_fields: 无任何显式success/failure/termination_reason字段；不可使用info进行reward或终止判定

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (ndarray) – 当前步观察
- action (int) – 当前步动作
- next_obs (ndarray) – 下一步观察（可用于评估过渡效果）
- info 中仅允许环境卡片明确声明的字段（当前无任何字段被声明，故info不可用）
- training_progress (float) 仅当任务提示明确允许时可使用，当前未允许

禁止使用：
- original_reward（任何掩蔽的官方奖励）
- 未声明的 info 字段
- 未声明的 obs 切片（例如假定的额外维度）
- 任何内置评估指标或环境内部状态

## 7. 可用于奖励函数的信号
- position: x_position, y_position (相对目标垫)  
- velocity: x_velocity, y_velocity  
- orientation: body_angle  
- angular_velocity: angular_velocity  
- contact: left_support_contact, right_support_contact (二值浮点，1.0表示接触)  
- action/engine: action index (可用于动作惩罚，或结合下一步状态推断推力效果)
- other: 无额外传感器

## 8. 不确定或不可用的信号
- 目标垫的确切尺寸或接触阈值未知；仅能通过观测位置和接触信号推断是否真正着陆
- 步数上限或时间惩罚信息未提供
- official reward 的任何组成部分被屏蔽，不可获取
- 训练进度 training_progress 未授权使用
- 无任何 external 目标位置之外的指引信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two downward support legs
  actuator_type: thrusters (one main engine, two orientation engines)
  contact_structure: two point contacts (left and right support), target pad is flat
primary_objectives:
  - reach center of target pad (x≈0, y≈0)
  - settle with near-zero velocity and zero angle
secondary_objectives:
  - minimize engine usage (sparse or weak thrust preferred)
  - fast completion (implicit via episode length, but no explicit time reward given)
main_failure_risks:
  - crashing into ground or obstacle (outside pad)
  - sliding off viewport horizontally
  - failing to reduce velocity, bouncing or never settling
  - settling outside the pad (incorrect location)
  - excessive fuel burn without making progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_target  
  purpose: 引导主体向目标垫（x=0, y=0）移动，同时减速，并逐步减小姿态角。  
  why_required: 核心目标；没有该引导则无法收敛到目标区域。  
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle]  
  risks: 权重过高可能导致急停或只关心位置而忽略稳定；需结合 landing 阶段。

- role_id: stable_landing  
  purpose: 在接近目标垫时，鼓励低速下让左右支撑脚同时接触。  
  why_required: 单脚接触或高速接触不符合“settle”要求；必须引入接触与速度联合条件。  
  usable_signals: [x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact, x_position, y_position（用于判定是否在垫上）]  
  risks: 接触信号本身可能提前出现（如蹭到地面），需用位置过滤；阈值设计不好可能导致早停。

### 10.2 条件职责 conditional_roles
- role_id: action_penalty  
  purpose: 鼓励节省燃料，惩罚非必要的发动机使用（特别是主发动机）。  
  condition_to_use: 当 agent 已经学会基本到达和着陆后，引入轻微的动作成本以优化效率；初期可关闭或极弱。  
  usable_signals: [action]  
  risks: 过强的惩罚会阻止 agent 学习使用发动机；可能引起根本不飞行的懒惰策略。

- role_id: time_efficiency (implicit)  
  purpose: 鼓励快速完成，可通过每步小的负奖励实现。  
  condition_to_use: 如果环境无步数限制或你想鼓励短路径，但需保证不压倒其他目标。  
  usable_signals: [无直接信号，需要依赖函数外部步数计数；但环境不提供信息，可用固定的小常数奖励]  
  risks: 若未正确标定可能导致提前 crash 以赚取总奖励。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: official_reward_replica  
  reason: 明确禁止使用 original_reward；任何试图模仿官方 reward 的组件都不可靠且违反规则。  
  forbidden_or_missing_signals: [original_reward]
- role_id: info_based_success  
  reason: info 字典为空，无 success 标记可用。只能从观测推断成功，不得依赖 info。  
  forbidden_or_missing_signals: [info.success, info.failure]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | x_position, y_position, x_velocity, y_velocity, body_angle | — | dense_state_signal, bounded_signal, distance_penalty (欧氏距离到目标), velocity_penalty, angle_penalty | 可分阶段：远距离时侧重位置减少距离，靠近时侧重减速 |
| stable_landing | x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact, x_position, y_position | — | conditional_reward (当位置接近0且接触均成立时启用), speed_penalty, angular_penalty | 引入条件 mask，只有实际在目标垫上方/上且双脚接触才给正奖；可对速度和角度的范数施加penalty |
| action_penalty | action | — | action_cost (如主发动机动作 penalize) | 可仅对动作2（主发动机）施加成本，不影响姿态发动机或空动作 |
| time_efficiency | — | step counter in compute_reward (当前接口无) | step_constant (固定小负值) | 需谨慎；如果环境会truncate，每步负奖励有助于快速完成，但容易让 agent 选择 crash 提前结束 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 始终悬停不下降或水平漂移 | 垂直速度持续接近零，y_position 不减小 | 减弱对主发动机的惩罚，或增强高度减少的正奖励 |
| 高速撞击目标垫或地面 | 快速 y_velocity 变为负大值，然后 episode 终止（很可能 crash） | 在接近目标时显著增加速度惩罚；引入高度相关的衰减系数 |
| 到达目标垫附近但无法稳定，来回震荡 | body_angle 和 angular_velocity 持续变化，接触信号跳动 | 加强角度和角速度惩罚；着陆奖励需覆盖连续多个步以保证持续接触 |
| 只使用姿态发动机摇摆而不使用主发动机 | 主发动机频率低，x/y 位置难以下降 | 对“无推进”或仅姿态的动作给予轻微负奖励，或引入 progress 奖励（如高度减少） |
| 过早满足稳定条件（settled但未在垫上） | 终止时 x_position 或 y_position 明显偏离0 | 在 stable_landing role 中严格要求 x≈0, y≈0，且接触成立 |
| 左右支撑接触不平衡，导致倾斜 | 只有左或右接触成立，body_angle 显著 | 着陆奖励要求左右接触同时为1，并惩罚角度 |
