# Response Record

# 匿名环境理解卡片

## 1. 任务目标  
本环境要求控制一个具有主发动机和两个姿态发动机的 2D 飞行器（或类似着陆器），从视口顶部附近初始位置出发，最终在中心目标垫上安全着陆并停稳。  
**主目标**：尽快到达目标垫上方并使机体稳定接触目标垫（以合适速度与姿态完成软着陆）。  
**次目标**：在完成着陆的前提下，尽可能减少发动机推力使用（节省燃料）。  
**不应混淆的目标**：单纯的省燃料不是首要任务；若过早坠毁或飞出边界则失败；未接触目标垫的长时间悬停也不符合成功要求。

## 2. 任务类型选择  
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达预定目标区域（中心目标垫）并实现稳定接触，构成典型的 goal reaching 问题；附属的节能和姿态稳定仅为优化项，不改变任务主类型。

## 3. 观察空间 observation_space  
- type: Box  
- shape: (8,)  
- dtype: float32 或 int/enum 混合（contact flag 可能为 0/1 浮点）  
- obs[0]：x_position，水平相对目标垫中心的坐标，reward_usable: true  
- obs[1]：y_position，竖直相对目标垫高度的坐标，reward_usable: true  
- obs[2]：x_velocity，水平线速度，reward_usable: true  
- obs[3]：y_velocity，竖直线速度，reward_usable: true  
- obs[4]：body_angle，机体朝向角度，reward_usable: true  
- obs[5]：angular_velocity，角速度，reward_usable: true  
- obs[6]：left_support_contact，左支撑脚/左腿接触标志，reward_usable: true  
- obs[7]：right_support_contact，右支撑脚/右腿接触标志，reward_usable: true  

## 4. 动作空间 action_space  
- type: Discrete  
- n: 4  
- action 0：no_engine，不点火，滑行/惯性运动。  
- action 1：left_orientation_engine，点火一侧姿态发动机，产生偏转力矩。  
- action 2：main_engine，点火主发动机，提供主体升力/反向推力。  
- action 3：right_orientation_engine，点火另一侧姿态发动机，产生相反方向偏转力矩。  

## 5. step 与终止条件分析  
### 5.1 终止模式  
- success-like termination：body_not_awake_or_settled（机体安静/停稳）可能代表成功着陆，但具体需验证是否仅接触目标垫才算成功，以及是否结合接触标志。  
- failure-like termination：crash_or_body_contact（机体与地面或障碍非正常接触），horizontal_position_outside_viewport（水平位置超出边界）都极可能是失败。  
- ambiguous termination：body_not_awake_or_settled 若未伴随成功条件（如双足接触目标垫），可能只是意外停摆。  
- truncation：未提供显式 episode 长度截断，但目前信息未出现时间限制，视为无截断。  

### 5.2 success/failure 信号可用性  
- explicit_success_flag_available: false（info 为空字典，没有 success 字段）。  
- explicit_failure_flag_available: false。  
- allowed_info_fields: 无（info 返回 {}）。  
- forbidden_or_uncertain_info_fields: 任何未在 step source 中出现的 info 字段均禁止使用（如 success, failure, distance_to_target 等）。  

## 6. reward 函数接口契约  
函数签名：  
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：  
- obs, action, next_obs （其中 obs 为当前步观察，next_obs 为下一步观察，两者均可使用）  
- info 中无可用字段，禁止依赖 info  
- training_progress 可选使用，但本环境未明确要求 curriculum，仅可在组件化设计中按安全方式使用  

禁止使用：  
- original_reward（被屏蔽，不可访问）  
- 任何未声明的 info 字段或隐藏的内部状态  
- 真实环境名、gym ID、benchmark 名称  

## 7. 可用于奖励函数的信号  
- position: x_position, y_position（相对目标垫位置）  
- velocity: x_velocity, y_velocity  
- orientation: body_angle, angular_velocity  
- contact: left_support_contact, right_support_contact（标志位，0/1）  
- action/engine: 所选动作（engine usage 可从 action 推断）  
- other: 可构造派生信号，如距离 = sqrt(x^2+y^2)，速度矢量大小，接近度变化等  

所有原始 observation 字段均明确可用，无缺失。  

## 8. 不确定或不可用的信号  
- 无地面真值成功标志（info["success"] 不存在）  
- 无碰撞强度或接触力信息  
- 无燃料余量或推力水平计量（仅动作选择，无连续推力值）  
- 无目标垫是否被触碰的单独指示（只能通过接触脚标志和机体状态间接推断）  

## 9. 专家任务画像 expert_task_profile  
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body (lander/ spacecraft) with two support legs
  actuator_type: main engine (vertical thrust) + two attitude thrusters (differential)
  contact_structure: left and right foot contact sensors, target pad centred
primary_objectives:
  - reach target pad and make stable contact (both feet on pad, near-zero velocity)
  - avoid crashing (hard body contact with ground/wall) or flying out of viewport
secondary_objectives:
  - minimize engine usage (fuel efficiency)
  - minimize time to landing (implicit via episode length / shaping)
main_failure_risks:
  - overusing main engine → overshoot or unstable oscillation
  - insufficient orientation control → landing on side/crashing
  - ignoring attitude damping → excessive angular velocity prevents stable contact
```

## 10. 奖励职责拆解 reward_role_decomposition  
### 10.1 主职责 mandatory_roles  
- role_id: approach_progress  
  purpose: 引导 agent 靠近目标垫（减小相对位置误差）  
  why_required: 核心目标，不接近则无法完成着陆  
  usable_signals: x_position, y_position，可构造距离  
  risks: 过度依赖距离可能导致 agent 不学减速/姿态稳定就盲目冲撞  

- role_id: soft_landing_regularization  
  purpose: 在接近目标垫时鼓励低速和垂直方向的稳定接触  
  why_required: 以防高速撞击导致 crash，保证停在垫子上  
  usable_signals: x_velocity, y_velocity, 以及 possibly 与 y_position 结合判断接近阶段  
  risks: 若过早惩罚速度可能阻碍探索；必须与接近信号耦合（距离越小权重越高）  

- role_id: orientation_stability  
  purpose: 保持机体角度接近竖直（避免过大倾斜导致侧翻或不安全接触）  
  why_required: 安全接触要求姿态可控  
  usable_signals: body_angle, angular_velocity  
  risks: 角度惩罚过强可能导致 agent 完全不敢使用姿态发动机，影响接近轨迹  

### 10.2 条件职责 conditional_roles  
- role_id: fuel_efficiency_penalty  
  condition_to_use: 仅在 agent 已经能够基本完成着陆任务时，才加重对此职责的惩罚以优化燃料；或者在训练后期通过 training_progress 调度权重。  
  usable_signals: action (可区分 main_engine, orientation engine 使用)，可统计连续使用次数  
  risks: 早期加入会挫伤探索，因为不点火则无法移动  

- role_id: terminal_bonus_for_stable_contact  
  condition_to_use: 在达到终止状态（likely settled）且确认双足接触目标垫且机体稳定的条件下给出正向奖励。  
  usable_signals: left_support_contact, right_support_contact, angular_velocity, velocity (near-zero)  
  risks: 因为无 explicit success flag，需通过 episode 最后一步 next_obs 检测这些信号，可能不够鲁棒（agent 可能在垫子附近停稳但并未真正成功）  

### 10.3 慎用/禁用职责 avoid_roles  
- role_id: progress_by_original_reward  
  reason: original_reward 被完全屏蔽，不可访问，任何 replication 均违规。  
  forbidden_or_missing_signals: original_reward  

- role_id: curiosity_exploration  
  reason: 本环境无需探索大量状态空间，目标明确，内在探索奖励可能分散 agent 精力；且没有 novelty 信号源。  
  forbidden_or_missing_signals: 需要预测误差或 visit count，不可用  

## 11. role_to_signal_mapping  
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_progress | x_position, y_position → distance | none | dense_state_signal (negative distance), bounded_signal, radial shaping | 可直接取 -distance 或 -distance^2，注意不要使奖励尺度爆炸 |
| soft_landing_regularization | y_velocity, x_velocity, distance | none | velocity_penalty * gating(distance) | gating 可使用 1/(1+distance) 或 after a threshold |
| orientation_stability | body_angle, angular_velocity | none | quadratic_penalty | 角速度平方 + 角度偏差平方 |
| fuel_efficiency_penalty | action | continuous usage counter? | discrete action penalty per step | 可用动作索引区分 engine usage |
| terminal_bonus_for_stable_contact | left_support_contact, right_support_contact, angular_velocity, linear velocities | explicit success flag | binary conditional bonus | 仅在 terminated 且条件满足时加常数 |

## 12. 初始训练后应观察的 failure modes  
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 始终悬停不下降 | y_position 不减小或增加 | 检查距离 reward 是否足够，是否过度惩罚主发动机使用 |
| 高速撞击目标垫或地面 | 到 episode 末端仍有大绝对值 y_velocity 且 crash | 加强 soft landing 职责，尤其在接近时增加速度惩罚的权重 |
| 来回振荡无法稳定 | x_position 反复过零，角速度大 | 检查姿态稳定职责是否生效，可增加角速度 damp 权重 |
| 完全不使用姿态发动机导致旋转失控 | body_angle 发散，crash | 可考虑对长期不调整姿态的轻微负奖励或引入 cost_for_zero_attitude_control |
| 偏向只使用姿态发动机而不点火，长久滞留 | 进度极慢且 y_position 几乎不变 | 增强 approach_progress 对纵向距离的敏感性，并适度减小姿态 fuel penalty |
| episode 过早因出界终止而没有学到靠近 | 大量 episodes 因横向出界结束 | 可以为横向超出范围增加强负惩罚，或在边界区域给额外惩罚 |
| 接触垫子但未完全停稳（单脚接触） | left_support_contact xor right_support_contact true while settled | 可为双脚接触且静止给出明显 bonus，并确保接触检测正确 |
