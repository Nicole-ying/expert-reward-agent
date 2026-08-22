# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 动态着陆/停靠控制问题。主体从接近画面顶部中央的位置出发，并带有随机初始冲量。核心目标是**让主体尽快到达并稳定停靠在画面中央的目标垫上**，同时**尽可能减少引擎推力使用**。自主学习到的策略应当先接近目标，逐渐减速，保持身体姿态稳定，最终轻柔与目标垫接触。注意：本任务不以“最快到达”为唯一指标，也不要求严格控制绝对姿态轨迹，而是以成功稳定着陆为主，燃料效率为次。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是到达并稳定在指定的中央目标垫，属于典型的“到达目标位置”问题，燃料和时间是附加约束而非并列多目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断，实际 Box 默认为 float32)
- obs[0]: x_position，水平位置相对于目标垫的坐标，reward_usable: true
- obs[1]: y_position，垂直位置相对于目标垫高度的坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，主体朝向角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧支撑点接触标志 (0.0 or 1.0)，reward_usable: true
- obs[7]: right_support_contact，右侧支撑点接触标志 (0.0 or 1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action/action_dim 0: no_engine，无操作/空闲
- action/action_dim 1: left_orientation_engine，点燃左侧姿态引擎
- action/action_dim 2: main_engine，点燃主推进引擎
- action/action_dim 3: right_orientation_engine，点燃右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 可能表示成功着陆并稳定，但需要结合位置判断，未提供明确成功标签。
- failure-like termination: `crash_or_body_contact`（撞击或身体异常接触）和 `horizontal_position_outside_viewport`（水平离开视口），这两种明显为失败。
- ambiguous termination: `body_not_awake_or_settled` 在目标垫以外也可能发生（如坠地后静止），因此不能直接作为成功信号；是否为成功必须从观测与终止原因中进一步推断，但当前源码未提供终止原因字段。
- truncation: step 返回的第二个布尔值为 `False`，无截断标志，推测环境自然终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （info = {}，无任何字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用，因为源码返回空字典；禁止假设 info["success"]、info["failure"]、info["termination_reason"] 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（所有维度含义明确，可用于构造奖励）
- action（当前动作）
- next_obs（下一时刻观测，可用于形状性奖励）
- info 为空，故无可用字段；若接口要求 info 不得出现未声明字段，请保持 info 不使用。

禁止使用：
- original_reward（已屏蔽，禁止还原）
- 任何未在 observation_space 说明中出现的 obs 子维度
- 任何 info 字段（因为 info = {}）
- training_progress 除非 prompt 中明确说明允许使用

## 7. 可用于奖励函数的信号
- position: x_position, y_position（到目标垫的相对位置，可构成距离惩罚/奖励）
- velocity: x_velocity, y_velocity（可用于鼓励减速、软着陆）
- orientation: body_angle（可用于稳定朝向）
- contact: left_support_contact, right_support_contact（检测着陆，鼓励两脚同时接触）
- action/engine: 动作选择（离散 ID，可计算燃料消耗）
- other: 无额外信息

## 8. 不确定或不可用的信号
- 成功/失败标签：不存在，不能直接用于奖励。
- 距离目标垫的绝对坐标：仅提供相对值，但已经足够。
- 主体是否在垫正上方：需通过 x_position≈0 且 y_position≈0 并结合接触信号近似判断。
- 着陆时的冲击力：无直接测量，仅能通过速度推断。
- 最大步数：未提供截断信息，无法推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle-like body
  actuator_type: thrusters (main engine + left/right orientation engines)
  contact_structure: two support legs/contacts (left and right)
primary_objectives:
  - reach the central target pad and settle on it
secondary_objectives:
  - minimize time to reach the pad
  - minimize engine thrust usage (fuel efficiency)
main_failure_risks:
  - crashing/impacting with excessive force
  - drifting horizontally out of the viewport
  - failing to settle (continuous bouncing / spinning)
  - spinning out of control due to excessive orientation thrust
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_target  
  purpose: 鼓励主体向目标垫靠近。  
  why_required: 这是任务核心——必须移动到垫上。  
  usable_signals: [x_position, y_position]  
  risks: 单独使用可能导致高速撞击，需配合减速职责。

- role_id: safe_landing  
  purpose: 鼓励在目标垫上进行软着陆，即两脚同时接触、速度低。  
  why_required: 否则主体可能飞越目标，或仅单脚擦过而不稳定。  
  usable_signals: [x_velocity, y_velocity, left_support_contact, right_support_contact, maybe x_position, y_position]  
  risks: 过早施加可能使 agent 不敢机动，可在接近垫子时激活。

- role_id: settlement_stabilization  
  purpose: 在接触垫子后鼓励速度趋零、角度稳定，实现“settled”状态。  
  why_required: 终止条件包含 body_not_awake 或 settled，但未提供明确成功信号；通过奖励静止可引导学习可靠着陆。  
  usable_signals: [x_velocity, y_velocity, angular_velocity, body_angle]  
  risks: 权重过大可能抑制探索。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency  
  condition_to_use: 当主体接近目标垫时加入，避免在远距离时抑制推进。  
  usable_signals: [action (discrete ID: 0=no thrust, others have cost)]  
  risks: 过早使用可能导致 agent 不去探索。

- role_id: orientation_stability  
  condition_to_use: 当主体接近垫子时，鼓励保持 body_angle ≈ 0。  
  usable_signals: [body_angle]  
  risks: 远距离时强迫直立可能妨碍动力学，建议仅在接近垫子时激活。

- role_id: time_penalty  
  condition_to_use: 可全程施加一个很小的每步负奖励，促使快速完成任务。  
  usable_signals: [step constant]  
  risks: 若任务难以完成，会严重挫伤探索，需极小权重或仅在训练后期启用。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: crash_penalty_from_info  
  reason: info 为空，无法获得 crash 标签。  
  forbidden_or_missing_signals: [info crash flag]  
- role_id: velocity_tracking  
  reason: 任务无预设速度轨迹，不应跟踪特定速度剖面。  
  forbidden_or_missing_signals: [target velocity]  
- role_id: precision_docking  
  reason: 目标垫足够大，不需要亚厘米级精度。  
  forbidden_or_missing_signals: [high-precision target coordinates]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | x_position, y_position | none | -distance_to_target (negative l2 distance) 或 bounded_signal(negative norm) | 核心引导信号 |
| safe_landing | x_vel, y_vel, left_contact, right_contact, x_position, y_position | exact landing force | -|velocity|_penalty + contact_bonus (only if both contacts are true and x,y near zero) | 需结合位置判断防止误激励 |
| settlement_stabilization | x_velocity, y_velocity, angular_velocity, body_angle | none | -|velocity| -|angular_velocity| -|angle| | 鼓励完全静止，可在接触后加重 |
| fuel_efficiency | action (ID 0=good, others=bad) | none | -cost_per_nonzero_action (action != 0) | 仅接近垫子时生效 |
| orientation_stability | body_angle | none | -|body_angle| (or quadratic_penalty) | 接近垫子时激活 |
| time_penalty | constant per step | none | constant negative reward per step | 全局轻量惩罚 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主体不移动或只使用无推力动作 | 奖励上升但 x,y 变化极小，很快因出界而终止 | 增大approach_target权重，引入 explore bonus 或减小燃料惩罚 |
| 高速撞击目标垫后弹飞或 crash | 速度在接近目标时仍很高，终止于 crash | 增强 safe_landing 中的速度惩罚，或在接近垫子时额外减速 |
| 主体在垫子周围来回振荡但不落地 | x, y 反复过零，未获得接触信号 | 在接触奖励中加入“必须双脚接触且速度小”的条件，或增加 settlement 权重 |
| 过度使用姿态引擎导致旋转失控 | body_angle 和 angular_velocity 剧烈变化 | 加入 orientation_stability，限制旋转速度；可能需要对角度使用更重的惩罚 |
| 过早耗尽燃料或因固定时间惩罚不敢动作 | 动作 0 频率极高，主体迟迟不接近目标 | 降低 time_penalty，暂时禁用 fuel_efficiency，增加 approach 奖励 |

