# 匿名环境理解卡片

## 1. 任务目标
主体从视野顶部中央附近以随机初始速度出发，目标是尽可能快速、稳定地降落在中央目标垫上并停稳（双腿接触、位置居中对齐、姿态水平、速度几乎为零）。次级目标是整个过程中尽量少使用发动机推力（省燃料）。不应混淆的目标：单纯的“存活”或“保持平衡”不是主目标，而是着陆成功的前提；单纯的“燃料为零”也不是主目标，只是在成功着陆前提下才受鼓励。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达并稳定在指定的目标位置（中央垫子）。尽管存在燃料效率这一次要目标，但任务成败仍由是否成功软着陆决定，因此属于导航到达族，而非多目标。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64 或 float32（由环境决定）  
- obs[0]: name=x_position, meaning=相对于目标垫水平位置的偏差坐标，reward_usable: true  
- obs[1]: name=y_position, meaning=相对于目标垫高度的垂直偏差坐标，reward_usable: true  
- obs[2]: name=x_velocity, meaning=水平线速度，reward_usable: true  
- obs[3]: name=y_velocity, meaning=垂直线速度，reward_usable: true  
- obs[4]: name=body_angle, meaning=机体倾斜角度（弧度制），reward_usable: true  
- obs[5]: name=angular_velocity, meaning=角速度，reward_usable: true  
- obs[6]: name=left_support_contact, meaning=左支撑腿接触标志（1.0 表示接触），reward_usable: true  
- obs[7]: name=right_support_contact, meaning=右支撑腿接触标志（1.0 表示接触），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: name=no_engine, meaning=不点火，无推力  
- action 1: name=left_orientation_engine, meaning=点燃左姿态发动机（产生一个力矩和微小横向力）  
- action 2: name=main_engine, meaning=点燃主发动机（产生向上推力）  
- action 3: name=right_orientation_engine, meaning=点燃右姿态发动机（产生反向力矩和微小横向力）  

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  - body_not_awake_or_settled：机体进入睡眠或完全稳定不动。当双腿安全接触目标垫且速度、姿态均在阈值内时可能触发，属于成功信号（但无法直接与失败情形区分）。  
- failure-like termination:  
  - crash_or_body_contact：机体与地面或障碍猛烈碰撞（可能由于姿态过度倾斜或高速冲击非垫区域）→ 典型失败。  
  - horizontal_position_outside_viewport：水平位置超出有效视野范围 → 失败。  
- ambiguous termination:  
  - body_not_awake_or_settled 若发生在垫外（比如在倾斜坡道上稳定）也可能被终止，此时实际是失败，但终止模式相同。  
- truncation: 未在代码片段中明确给出最大步数，但通常存在隐含截断。本分析不依赖截断信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 字段）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: 无。info 恒为空字典，不可用任何 info 键。  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：来自当前步的观察数组（8维）
- action：当前步采取的动作（0~3整型）
- next_obs：来自下一步的观察数组（8维），可用于检查接触变化、速度衰减等
- training_progress：0~1之间的训练进度系数（仅当 prompt 明确允许时使用，本环境当前未给出要求，保守起见可不使用）
- info：已明确为空，因此任何 info 字段均不允许

禁止使用：
- original_reward（已掩盖的官方奖励）
- 任何未声明的 obs 切片含义
- 环境名称、benchmark 名称
- info 的任何字段（因为 info 恒空）

## 7. 可用于奖励函数的信号
- position: obs[0] (x 偏差), obs[1] (y 偏差) → 目标接近程度
- velocity: obs[2] (vx), obs[3] (vy) → 移动速度，用于鼓励减速
- orientation: obs[4] (body_angle) → 倾斜程度，应接近0
- angular_velocity: obs[5] → 角旋转速率，可辅助惩罚快速旋转
- contact: obs[6] (左腿接触), obs[7] (右腿接触) → 用于鼓励双腿着垫、避免单腿着陆
- action/engine: action 值本身 → 可惩罚非零动作（燃料使用）或针对性地惩罚姿态发动机频繁使用
- other: 无额外可用信号

## 8. 不确定或不可用的信号
- 显式成功/失败标志：完全不可用（info 为空）
- 燃料剩余量：无观测
- 当前步数/剩余步数：未传入
- 与目标垫的精确接触质量（如接触力）：无
- 风速/扰动：无观测，但环境描述提及 wind，不过未在观测中暴露

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: main_thruster + left/right_orientation_thrusters
  contact_structure: two_leg_touch_sensors (left and right)
primary_objectives:
  - 水平位置收敛至 0（垫中心）
  - 垂直位置收敛至接近 0（垫面高度）
  - 机体角度收敛至 0（水平）
  - 几乎无平动和旋转角速度（软着陆）
  - 两条腿均稳定接触垫子
secondary_objectives:
  - 最小化总推力使用次数/总冲量（燃料效率）
  - 在较短时间内完成着陆（可通过隐式惩罚时间促进，但不直接可获取）
main_failure_risks:
  - 过度倾斜导致碰撞或 crash
  - 高速触垫或撞击地面造成 crash
  - 单腿着垫后倾覆
  - 飘出水平边界
  - 在垫外区域静止（假成功）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 引导机体向目标垫水平面和高度靠近，使 x_position→0，y_position→0。  
  why_required: 这是到达任务的核心，若无此奖励，机体可能留在任何位置。  
  usable_signals: [obs[0], obs[1]]（当前和下一状态均可）  
  risks: 过于强烈的奖励可能导致机体急速冲向垫子，忽视速度和姿态，因此需要配合 soft_landing。

- role_id: soft_landing  
  purpose: 鼓励线速度（尤其是垂直速度）接近零，姿态角度接近零，角速度接近零，确保安全接触。  
  why_required: 即使位置准确，高速撞击或倾斜着陆也会导致崩溃，不符合任务要求。  
  usable_signals: [obs[2], obs[3], obs[4], obs[5], next_obs[2], next_obs[3], next_obs[4], next_obs[5]]  
  risks: 若权重过大，可能使机体悬停而不下降，延长步数；需平衡目标接近与减速。

- role_id: contact_quality  
  purpose: 当接近目标且速度很低时，鼓励双腿同时接触（obs[6] 和 obs[7] 均为 1.0），避免单腿着地。  
  why_required: 任务明确要求“稳定着陆”，双腿着地是稳定性的重要指标；若只单腿着垫，极易倾覆。  
  usable_signals: [obs[6], obs[7], next_obs[6], next_obs[7], 以及位置、速度条件]  
  risks: 在远离垫子时鼓励接触可能错误；应与目标接近程度耦合使用。

- role_id: success_shaping  
  purpose: 当所有成功条件接近满足时（位置近零、速度极小、角度近零、双腿接触）给予额外的正向奖励，帮助策略向成功状态聚集。  
  why_required: 因为无显式成功标志，须用观测合成成功信号，以弥补稀疏的终止奖励。  
  usable_signals: [obs[0], obs[1], obs[2], obs[3], obs[4], obs[6], obs[7]]  
  risks: 阈值设置过严会导致奖励消失，过松会让策略在次优状态满足，需仔细调参。

### 10.2 条件职责 conditional_roles
- role_id: thrust_penalty  
  purpose: 对使用任何发动机的动作施加轻微惩罚，鼓励节省燃料。  
  condition_to_use: 整个 episode 内持续生效，但强度应低于引导性奖励，避免因省燃料而放弃着陆。  
  usable_signals: [action]  
  risks: 过强会导致 agent 不敢下降或不敢修正姿态，必须在成功着陆先决下引入。

- role_id: angular_damping  
  purpose: 显式惩罚角速度，防止机体旋转过快。  
  condition_to_use: 可与 soft_landing 合并，但在角度控制初期尤其重要；可与角度惩罚叠加。  
  usable_signals: [obs[5], next_obs[5]]  
  risks: 过度抑制角运动可能降低姿态发动机探索，需与姿态稳定目标协调。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_reward  
  reason: 没有可用的 success flag，无法仅在真正成功时给奖励。  
  forbidden_or_missing_signals: [info.success] 不存在。

- role_id: explicit_failure_penalty  
  reason: 没有可用的 failure flag，且终止原因无法直接传入 reward 函数；仅凭观测难以可靠判断所有失败模式，可能导致误处罚。  
  forbidden_or_missing_signals: [info.failure, info.termination_reason] 不存在。

- role_id: progress_time_penalty  
  reason: 无步数观测，环境未提供 episode 长度信息，无法构造基于时间的惩罚。  
  forbidden_or_missing_signals: [step_count, max_steps] 不可用。

- role_id: fuel_exact_penalty  
  reason: 缺少燃料表或推力大小的直接观测，只有动作索引，可间接实现简单的推力惩罚，但无法精确计算冲量。  
  forbidden_or_missing_signals: [fuel_remaining, thrust_magnitude] 不可用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | obs[0], obs[1] / next_obs[0], next_obs[1] | - | distance_to_target, bounded_signal, quadratic_penalty | 鼓励 x,y 同时接近 0 |
| soft_landing | obs[2], obs[3], obs[4], obs[5] / next_obs 对应 | - | abs_delta, quadratic_penalty, product_of_conditions | 可组合速度与角度的惩罚 |
| contact_quality | obs[6], obs[7], next_obs[6], next_obs[7], 位置/速度 | - | logical_and, conditional_bonus | 仅当接近目标、低速时奖励双腿接触 |
| success_shaping | obs[0..4], obs[6], obs[7] | 无直接成功标志 | thresholded_product, gaussian_kernel | 多个指标综合评分以近似成功 |
| thrust_penalty | action | 推力大小、燃料 | action_indicator, constant_penalty | 动作不为 0 时施加固定惩罚 |
| angular_damping | obs[5] / next_obs[5] | - | abs_delta, quadratic_penalty | 可与角度惩罚一起构成旋转稳定性 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 机体一直悬停不下降或下降极慢 | y_position 长期远大于 0，vy 接近 0；可能是 soft_landing 过重或推力惩罚过强 | 降低下降速度惩罚，增强位置奖励的垂直分量 |
| 高速撞击垫子或地面 | 接触瞬间 (obs[6] 或 obs[7] 跳变) 时 vy 或 vx 依然很大 | 增强速度惩罚，或在接近垫子时提高减速权重 |
| 姿态来回摆动，过度使用姿态发动机 | 角速度/角度频繁正负交替，大量动作=1或3 | 增加角度和角速度惩罚，或提高推力惩罚 |
| 在垫子周围晃荡但不落脚 | y_position 接近 0 但双腿不接触，水平位置不稳定 | 提升对双腿接触的奖励，或对位置稳定给出额外奖励 |
| 单腿着垫后翻倒 | 一条腿接触后另一条腿再未接触，角度迅速增大 | 强调双腿接触奖励，加重单腿着地时的角度惩罚 |
| 飘出水平边界 | x_position 绝对值持续增大 | 增加水平位置惩罚，或以边界惩罚引导回中 |
| 在垫外区域稳定停住（假成功） | x 或 y 偏差明显，但速度/角度很小且 legs 接触（非垫） | 强化位置精度要求，或减轻 contact 奖励对非中心区域的适用性 |