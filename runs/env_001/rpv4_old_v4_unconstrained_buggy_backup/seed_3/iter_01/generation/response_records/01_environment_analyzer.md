# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是到达指定的目标位置（中央着陆平台）并稳定停止，完全符合 navigation_goal_reaching 的定义。虽然存在燃料优化和姿态控制等附属要求，但它们不构成权重相当的冲突多目标，主次分明。  
动力学子类型: goal_approach_and_soft_contact（要求接近目标并实现低速、姿态稳定的软接触/停靠）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推测为 float32  
- obs[0]: x_position，水平坐标（相对于目标平台的 x 偏移），reward_usable: true  
- obs[1]: y_position，垂直坐标（相对于平台高度，即飞行器底部到平台表面的距离），reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，身体倾斜角度（弧度 or 归一化角度），reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  

备注：所有维度均为连续数字，接触标志虽为 0/1 但以浮点形式给出。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine（不点火，靠惯性/重力运动）  
- action 1: left_orientation_engine（点火左向姿态调节引擎，产生向左旋转力矩）  
- action 2: main_engine（点火主引擎，产生向上的推力，通常用于减速/悬停）  
- action 3: right_orientation_engine（点火右向姿态调节引擎，产生向右旋转力矩）  

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能代表身体已静止，结合 y_position≈0 且左右支撑接触为 1 时可视为成功着陆。但 termination 本身是模糊的，因为 crash 后也可能导致静止。  
- failure-like termination: crash_or_body_contact 中“crash”部分，horizontal_position_outside_viewport（水平超出视口）  
- ambiguous termination: body_not_awake_or_settled 的具体触发条件未知，可能是身体进入休眠/静止状态；crash_or_body_contact 中的 body_contact 可能包含正常着陆接触，需依赖位置判断。  
- truncation: 未提供  

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）  
- explicit_failure_flag_available: false（同样无 failure 字段）  
- allowed_info_fields: 无（info 返回空字典 {}）  
- forbidden_or_uncertain_info_fields: 所有 internal 字段均不可用，如 termination_reason 等  

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）
- action（当前采取的动作）
- next_obs（下一观测）
- info 为空，不提供任何可依赖信号
- training_progress 仅在 prompt 明确允许时可用（本例中未明确，默认不使用）

禁止使用：
- original_reward（官方奖励被屏蔽）
- official_reward
- 未声明的 obs 切片（例如假设 obs 包含额外维度）
- 任何未在 observation_space 中声明的 info 字段

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action 值（0~3），可用于燃料惩罚或推力鼓励
- other: 通过 next_obs 计算上述量的变化率（如速度变化、位置变化）

## 8. 不确定或不可用的信号
- 原始的官方奖励（被 mask）
- 终止原因细分（无法获得 crash 类型或成功标识）
- 物理引擎内部细节（质量、推力大小、重力常数）
- 是否在目标平台正上方的精确判定（只能通过位置和接触估算）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_landing_legs
  actuator_type: main_engine + two_orientation_thrusters
  contact_structure: two_feet_bottom_contacts
primary_objectives:
  - reach target pad (x→0, y→0)
  - settle with low velocity & stable orientation
secondary_objectives:
  - minimize fuel usage (engine activations)
  - smooth trajectory (avoid aggressive angle changes)
main_failure_risks:
  - high-speed crash into ground (crash_or_body_contact with large velocity)
  - landing outside pad or toppled (horizontal out of viewport, or one leg not contacted when y≈0)
  - running out of fuel equivalent (excessive engine use leading to early termination?)
  - early stop away from pad (body_not_awake_or_settled far from x=0)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 引导飞行器向目标位置 (x=0, y=0) 靠近  
  why_required: 这是导航任务的最基本驱动力，缺失则无法学习到达  
  usable_signals: [x_position, y_position]  
  risks: 仅考虑位置可能导致高速掠过目标，需要与速度阻尼配合  

- role_id: velocity_damping  
  purpose: 接近目标时降低速度，避免撞击，实现 soft contact  
  why_required: 没有减速要求，智能体可能高速冲入目标区域导致失败  
  usable_signals: [x_velocity, y_velocity]  
  risks: 若整体惩罚，可能抑制必要机动速度；需设计为接近目标时才生效或权重随距离变化  

- role_id: orientation_penalty  
  purpose: 保持身体接近垂直（角度接近 0），以稳定着陆  
  why_required: 倾斜过大易导致侧翻或单脚触地，且不符合安全着陆姿态  
  usable_signals: [body_angle, angular_velocity]  
  risks: 过于严格可能限制机动灵活性，可在接触前放宽  

### 10.2 条件职责 conditional_roles
- role_id: safe_contact_bonus  
  condition_to_use: 当 y_position 接近0（例如绝对值小于阈值）且左右支撑接触均为1时激活  
  usable_signals: [left_support_contact, right_support_contact, y_position]  
  risks: 如果仅凭接触标志就奖励，可能鼓励在远处提前接触地面；必须与位置条件绑死  

- role_id: fuel_efficiency_penalty  
  condition_to_use: 当动作并非 no_engine 时给予小幅度负奖励  
  usable_signals: [action]  
  risks: 权重过高可能导致智能体不敢使用引擎，无法有效控制下降；建议保持为微小正则化项，且可在训练后期逐步引入或使用 training_progress 调度（如果允许）  

### 10.3 慎用/禁用职责 avoid_roles
- role_id: early_settle_penalty  
  reason: 终止条件 body_not_awake_or_settled 含义模糊，无法在 step 级判断是否为错误早停；且早停可能已导致 episode 结束，届时代理无法得到奖励  
  forbidden_or_missing_signals: 无明确的“早停”信号；只能从终止推断，但终止后的 step 不再计算 reward  

- role_id: final_bonus_only  
  reason: 缺少显式 success 标志，无法可靠给出最终一次性奖励；且可能鼓励 agent 追求终止本身而非正确着陆。不建议使用基于 termination 的奖励。  
  forbidden_or_missing_signals: 无 success 标志  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | 无 | negative_l2_distance, negative_abs_error, squared_penalty | 可以加权组合，y 方向权重可较高（防止硬着陆） |
| velocity_damping | x_velocity, y_velocity | 无 | squared_velocity_penalty, velocity_magnitude_penalty | 可乘以距离衰减因子，只在目标附近生效 |
| orientation_penalty | body_angle, angular_velocity | 无 | absolute_angle_penalty, squared_angular_velocity_penalty | 角速度惩罚可单独添加以抑制晃动 |
| safe_contact_bonus | left_support_contact, right_support_contact, y_position | 无 | conditional_reward(if both_legs_contact and y_near_0) | 应一次性给予较大正值，且 episode 可能立即终止，需注意 |
| fuel_efficiency_penalty | action | 无 | discrete_action_cost([•]) | 仅对非零动作施加微小负常数 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击地面 | 最终 step 的 y_velocity 很大，y_position≈0 但 crash | 加强 velocity_damping 权重，或在接近地面时加大惩罚 |
| 水平飞出视野 | x_position 绝对值持续增大并终止 | 增强 goal_proximity 中的 x 方向引导，加入水平位置约束 |
| 在非目标区域“假着陆” | 终止时左右接触均为1，但 x_position 远离0 | safe_contact_bonus 必须严格绑定 x 接近0 的条件；可增加 x 方向的接近奖励 |
| 过度摇摆或翻倒 | body_angle 振幅大，角速度高，或最终单脚接触 | 增加 orientation_penalty 或限制角速度上限；可在训练初期使用更重的角惩罚 |
| 完全使用 no_engine 自由落体 | 整个 episode 内 action 几乎全为 0 | 检查主目标引导是否过弱；可适度减轻 fuel penalty 或延迟引入 |
| 无法稳定结束，悬停振荡 | 在目标上空长时间不接触 | 调整速度阻尼与接触奖励的平衡，允许在 y_position 很小时给予接触鼓励，并可能使接触奖励略大于燃料惩罚 |
