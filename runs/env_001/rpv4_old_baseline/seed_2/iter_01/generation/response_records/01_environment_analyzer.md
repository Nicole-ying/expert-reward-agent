# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个从画面顶部中央附近出发的飞行器，安全、稳定地降落在画面中央的目标平台上。要求着陆时速度接近于零、姿态接近竖直，且所有支脚平稳接触平台。

次要目标：在确保主目标达成的前提下，尽量缩短飞行时间，并尽量减少主引擎和姿态引擎的使用（即节省燃料）。

不可混淆的目标：不应将“快速到达”或“节省燃料”凌驾于“安全着陆”之上；也不能将“悬停”或“保持在目标上方”当作成功条件。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达指定的目标位置并稳定停留，属于典型的导航到达类任务。附属的燃料、时间要求均为次要，不改变主目标的定性。observations 直接提供相对于目标的坐标，符合目标导向的导航范式。

dynamics_subtype: goal_approach_and_soft_contact（接近目标并实现低速、稳定接触）

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- obs[0]: x_position (相对目标平台的水平坐标), 可直接用于距离/接近奖励，reward_usable: true
- obs[1]: y_position (相对目标平台高度的垂直坐标), 同上，reward_usable: true
- obs[2]: x_velocity (水平线速度), 可用于着陆软度控制，reward_usable: true
- obs[3]: y_velocity (垂直线速度), 同上，reward_usable: true
- obs[4]: body_angle (机体朝向角), 可用于姿态奖励，reward_usable: true
- obs[5]: angular_velocity (角速度), 可用于姿态稳定性惩罚，reward_usable: true
- obs[6]: left_support_contact (左侧支脚接触标志，1.0 表示接触), 可用于着陆状态判断，reward_usable: true
- obs[7]: right_support_contact (右侧支脚接触标志，1.0 表示接触), 同上，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力，仅受重力/物理影响)
- action 1: left_orientation_engine (点燃左侧姿态推进器，产生旋转力矩)
- action 2: main_engine (点燃主推进器，提供向上推力并可能产生力矩)
- action 3: right_orientation_engine (点燃右侧姿态推进器，产生反方向旋转力矩)

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: body_not_awake_or_settled 如果发生在飞行器已接触地面且速度/角速度极低时，极可能意味着成功着陆；但如果发生在半空中或刚碰撞后，则可能是早期终止。
- failure‑like termination: crash_or_body_contact（与地面或障碍的异常碰撞）、horizontal_position_outside_viewport（水平飞出边界）明确为失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分成功/失败，需要结合观察判断。
- truncation: 代码中未出现 episode length 截断，但实际部署时可能通过外部 wrapper 实现，当前源中未见。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {} 或未返回任何键）
- forbidden_or_uncertain_info_fields: 任何未在以上列出的字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前 step 的观察）
- action（当前 step 执行的动作）
- next_obs（下一 step 的观察，可用于检测变化或下一时刻状态）
- info 中明确且安全可用的字段（当前为空，故禁止使用任何 info 内容）
- training_progress（仅当 prompt 明确允许使用进度信息时才可用；此处未声明，禁止使用）

禁止使用：
- original_reward（官方奖励已遮罩）
- official_reward（同上）
- 任何未在观察空间声明中列出的 obs 切片
- 任何未在允许列表中的 info 字段

## 7. 可用于奖励函数的信号
- position: x_position, y_position（可直接计算到目标的距离、高度差）
- velocity: x_velocity, y_velocity（可衡量接近速度、着陆软度）
- orientation: body_angle（偏离竖直的角度），angular_velocity（旋转速度）
- contact: left_support_contact, right_support_contact（着陆脚是否触地，可判断着陆状态）
- action/engine: action 索引可映射到是否使用主引擎、姿态引擎，用于推力/燃料惩罚
- other: 可通过 (obs, next_obs) 的组合构造微分信号，如速度变化、角速度变化等

## 8. 不确定或不可用的信号
- 明确的连续接触力/碰撞力（没有）
- 燃料余量或推力大小（没有直接测量，仅能通过动作间接推测）
- 成功标志位（info 中无）
- 平台检测区域（仅通过相对坐标隐含，没有显式“目标区域”标记）
- 任何与“官方奖励”相关的隐式信息

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 带支脚的垂直起降飞行器 (lander-like)
  actuator_type: 一个主推进器 + 两个对称姿态推进器
  contact_structure: 两个独立支脚，可分别检测接触
primary_objectives:
  - 到达目标平台中心（x_position ≈ 0，y_position ≈ 0）
  - 实现稳定、低速着陆（速度 ≈ 0，且两支脚同时触地）
  - 保持直立姿态（body_angle ≈ 0）
secondary_objectives:
  - 尽量缩短到达时间（在安全前提下）
  - 最小化推进器使用（动作非零即燃料消耗）
main_failure_risks:
  - 高速撞击地面导致 crash_or_body_contact
  - 水平方向漂移超出视口
  - 着陆时姿态严重倾斜并侧翻
  - 长时间悬停不降落（可能导致超时截断，若加装 wrapper）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: proximity_to_target
  purpose: 引导飞行器向目标平台移动
  why_required: 是任务的核心到达需求，无此则难以收敛到目标
  usable_signals: [x_position, y_position]
  risks: 如果仅用距离奖励，可能导致高速撞地；必须与速度/姿态约束结合

- role_id: soft_landing_conditions
  purpose: 确保着陆时速度接近于零，姿态竖直，支脚平稳接触
  why_required: 防止以危险方式完成任务（高速冲击、侧翻）
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact]
  risks: 如果权重过高，可能导致飞行器不敢接近地面（奖励悬崖）

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 按需惩罚不必要的推进器使用，以节省燃料
  condition_to_use: 当飞行器已经足够接近目标或已经完成着陆时，可加大惩罚；在远离目标时不应过度惩罚，以免阻碍探索
  usable_signals: [action]
  risks: 过早或过强的燃料惩罚会阻碍飞行器学习上升和移动

- role_id: terminal_landing_bonus
  purpose: 在成功着陆（接触且稳定）时给予一次性奖励，强化最终行为
  condition_to_use: 仅当 next_obs 显示两支脚同时接触且速度/角速度都接近零时授予
  usable_signals: [next_obs 的 contact、velocity、angular_velocity]
  risks: 如果阈值设置不当，可能把不稳定着陆也判为成功

### 10.3 慎用/禁用职责 avoid_roles
- role_id: time_bonus_or_penalty
  reason: 鼓励快速到达容易导致飞行器以危险方式高速撞击，与安全着陆冲突。当前环境没有显式的时间步惩罚接口，且 safety 高于 speed。
  forbidden_or_missing_signals: [无可用全局时间步数限制，且 time 信号未显式提供（依赖外部进度）]

- role_id: exact_position_shape_reward
  reason: 环境只需到达中心，不存在复杂的形状奖励（如走廊、栅栏），不需要分段引导。
  forbidden_or_missing_signals: [无子目标点]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| proximity_to_target | x_position, y_position | – | distance_reward, shaped_sq_distance, bounded_progress | 可结合指数衰减奖励靠近过程 |
| soft_landing_conditions | x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact | – | velocity_penalty, angle_penalty, quadratic_penalty | 仅在有接近行为时激活，否则奖励为零 |
| fuel_efficiency | action | – | action_count_penalty, engine_usage_penalty | 可对 action≠0 施加轻量惩罚 |
| terminal_landing_bonus | next_obs[2:8] (速度、角度、接触) | – | threshold_bonus | 需要严格的成功条件判定，防止 false positive |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 学习到稳定悬停在空中而不下降 | y_position 均值远大于 0，且长时间未触发终止 | 减小距离奖励在远距离时的强度，增加“接近即下降”的引导；或增加对 y_position > 0 的轻微惩罚 |
| 高速撞击目标平台 | 终止前的 y_velocity 很大，奖励曲线在终止前突然上升但失败率高 | 大幅提高 soft_landing_conditions 的权重，尤其是 y_velocity 惩罚；加入速度上限逐级惩罚 |
| 使用主引擎一直向上冲，飞出视口 | y_position 异常大，然后 horizontal_position_outside 终止 | 确保 proximity 奖励使用相对坐标的绝对值，避免正反馈溢出；对 out_of_bounds 给予极重惩罚 |
| 着陆时向一侧大幅倾斜并侧翻 | body_angle 在终止时显著偏离 0，单侧接触 | 增强角度惩罚，且令 terminal bonus 要求两脚同时接触；可配合 angular_velocity 惩罚 |
| 反复点火但不移动（燃料浪费） | action 多非零，但位移很小 | 检查燃料或推力惩罚是否过弱；适当加入动作平滑惩罚 (action_change) 或对无效推力惩罚 |
