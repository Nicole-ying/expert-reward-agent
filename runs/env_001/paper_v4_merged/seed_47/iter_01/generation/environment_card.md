# 匿名环境理解卡片

## 1. 任务目标
本环境是一个2D飞行器精确着陆任务。agent 从视野顶部中心附近出发，带有随机初始扰动。核心目标为安全、稳定地在中心目标平台上着陆——即到达指定相对水平位置 x≈0、高度 y≈0（平台高度），同时保持姿态接近竖直、双腿同时接触平台、速度几乎为零。次要目标为尽快完成着陆，并尽量少使用引擎推力（降低燃料消耗）。不应将存活时间或长时间悬停作为正面目标，也不应单纯最大化水平进度而忽略触地质量和姿态约束。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务的核心是到达指定的目标位置（平台中心）并满足姿态与接触约束，属于典型的带位姿终端约束的到达任务；尽管有“尽快、节能”的次级要求，但它们并不构成权重相当、互相冲突的多目标核心，因此不选 multi_objective_task。着陆过程中的平衡需求是到达目标的一部分，而非生存类任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（或 float64）
- obs[0]: x_position，相对于目标平台水平坐标，reward_usable: true
- obs[1]: y_position，相对于平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，身体朝向角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不激活任何引擎（保持当前惯性）
- action 1: left_orientation_engine，点燃左朝向引擎（产生转向或侧向推力）
- action 2: main_engine，点燃主引擎（一般提供向上的推力，但也可能产生旋转分量）
- action 3: right_orientation_engine，点燃右朝向引擎（转向或侧向推力，方向与左相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact：身体（除双腿外的部分）与地面发生碰撞 → 很可能为失败终止（坠毁）。
- horizontal_position_outside_viewport：水平位置超出视野边界 → 失败终止（出界）。
- body_not_awake_or_settled：身体不再活跃（例如静止且未触发其他终止）或满足平台稳定着陆条件（settled） → 若为 settled 则属于成功终止，若仅为不活跃但未满足着陆要求则可能为中立或失败。从任务目标推断，成功着陆的唯一途径就是触发 settled 条件（双腿接触、速度极低、姿态竖直等），因此该条件可视为成功类终止，但需要谨慎对待可能的非成功不活跃情形。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中无 success 字段，原始观测亦无直接标志）
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 为空字典，禁止读取任何字段）
- forbidden_or_uncertain_info_fields: info 内的任何内容均不可用；禁止使用 original_reward

补充推断路径（derived_possible）：
- 成功着陆可通过“终止时的 next_obs 满足两条腿接触、速度接近零、角度接近零、且未发生 crash 或出界”间接判断。
- 坠毁可通过突然的高加速度、body_angle 突变、或 body 位置骤然下降并伴随 contact 信号异常间接推断。
- 出界可从 x_position 超出视野范围推测。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）
- action（当前动作）
- next_obs（下一时刻观测）
- info 中明确允许的字段（当前无）
- training_progress 只在 prompt 明确允许时使用（此处未允许，禁用）

禁止使用：
- original_reward
- official_reward
- 未在允许信号中声明的 info 字段
- 未声明的 obs 维度

## 7. 可用于奖励函数的信号
位置相关：
- x_position, y_position（均可用于计算到目标点的欧氏距离、水平偏移、高度偏差；可构造距离进步量 delta_distance）
- 可通过 next_obs 与 obs 的 x/y 位置差获取位移方向

速度相关：
- x_velocity, y_velocity（可用于惩罚水平漂移、过大的垂直速度，特别是在接近目标时；可构造速度门控惩罚）
- 速度平方/模长可用于能量惩罚

姿态相关：
- body_angle（用于惩罚偏离竖直的姿态，着陆阶段应接近 0）
- angular_velocity（惩罚过大角速度，防止剧烈旋转）

接触相关：
- left_support_contact, right_support_contact（用于鼓励双腿同时接地，或惩罚单脚/belly着陆）

动作相关：
- action 的语义（no_engine、主引擎、偏转引擎）可用于燃料惩罚（如非零动作施加小惩罚）

间接推断成功的信号（derived_possible）：
- 当 next_obs 满足：双腿接触均为 1、x_velocity≈0、y_velocity≈0、|body_angle| ≈0、x_position≈0、y_position≈0，且当前步未检测到 crash 条件时，可以高置信度推断着陆成功，用于终端奖励。

## 8. 不确定或不可用的信号
- original_reward / official_reward：被屏蔽，不可用
- info 中的所有字段：空字典，无 success、failure、termination_reason 等标志
- 真实的 crash/body_contact 标记：没有直接的布尔标志，只能通过后续触地失败时的异常观测变化间接推断
- 燃料消耗量：观测中无直接度量
- 剩余时间/步数：不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_legs   # 2D 刚体，带有两个支撑腿
  actuator_type: main_engine_and_lateral_orientation_engines
  contact_structure: two_leg_contact_sensors_with_binary_flags
primary_objectives:
  - land_safely_on_target_pad           # 最终状态：x≈0, y≈0, 低速, 姿态竖直, 双腿同时着地
  - avoid_crash_and_out_of_bounds
secondary_objectives:
  - minimize_time_to_land               # 尽快完成
  - minimize_engine_usage               # 节省推力/燃料
main_failure_risks:
  - crash_due_to_body_contact_with_ground
  - failing_to_settle_while_hovering_indefinitely
  - tipping_over_or_losing_angular_control
  - drifting_out_of_viewport
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: distance_improvement
  purpose: 为非终止步提供密集的接近目标进度信号，衡量相对于目标位置的欧氏距离减小量。
  why_required: 核心导航信号，驱使 agent 向平台中心移动；可直接对抗悬停策略，因为悬停时距离不再减小，信号为零或负。
  usable_signals: [x_position, y_position]
  risks: 单纯依靠当前距离（proximity）容易形成悬停陷阱（停在较近高处持续得分而不完成着陆），因此必须