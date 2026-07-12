# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个二维飞行器着陆任务。飞行器从视口顶部中央附近以随机初速释放，目标是**尽快且节能地到达中央目标平台并稳定停靠**。具体要求包括：精确水平定位到平台上方，垂直速度接近零，保持姿态平稳，并让左右支撑足安全接触地面。飞行动力受离散引擎推力、姿态控制和物理约束。主要目标是安全着陆，次要目标是节省推力、快速完成。不应该追求极速而忽略平稳接触或倾角。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达并稳定在指定目标位置上，附带速度/姿态/能耗要求，典型的导航目标到达任务。没有多目标冲突，也不是纯粹的平衡或探索问题。

动力学子类型 dynamics_subtype: goal_approach_and_soft_contact
reason: 飞行器需从一定初始距离逼近平台，逐渐减速、调整姿态并实现左右支撑足同时接触的软着陆。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position，相对目标平台中心的水平偏移，reward_usable: true
- obs[1]: y_position，相对平台高度的垂直偏移，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧支撑足接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右侧支撑足接触标志（0/1），reward_usable: true

**注意**：所有8维均明确可用，reward_usable 为 true。但接触标志仅代表物理接地，并不代表成功停止，不能直接用作成功条件。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，无推力，仅靠惯性/重力。
- action 1: left_orientation_engine，启动左定向引擎（调节姿态）。
- action 2: main_engine，启动主引擎（提供主要推力，推测方向为机体下方或后方）。
- action 3: right_orientation_engine，启动右定向引擎。

**注意**：离散动作无幅度参数，每个动作是瞬时脉冲式发动机点火，持续时间由步长决定。不允许动作幅度或连续调节。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体停止并可能处于稳定接触状态，可能暗示成功着陆）——这是最可能的成功信号，但需结合接触和位置判断。
- failure-like termination: crash_or_body_contact（碰撞或机体与地面非支撑足接触），horizontal_position_outside_viewport（飞出视口水平边界）。
- ambiguous termination: body_not_awake_or_settled 可能因各种原因（悬挂、静止于非目标位置）发生，本身不保证成功着陆。
- truncation: 无显式最大步数，源中 truncation 恒为 False（见返回 `terminated, False, {}`），环境不设置时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，非平台提供的 success 字段）
- explicit_failure_flag_available: false
- allowed_info_fields: 无，info 恒等于空字典 {}，不可用于奖励。
- forbidden_or_uncertain_info_fields: 所有 info 字段，因为不存在。

**注意**：不能依赖 info["success"] 等，必须通过观测构建成功判断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (np.array of shape [8])
- action (int 0-3)
- next_obs (np.array of shape [8])
- info 中明确允许的字段：无，因为 info 恒为空 {}。
- training_progress 可选用，但当前 prompt 未明确强制使用以调度课程学习，谨慎使用。

禁止使用：
- original_reward
- official_reward
- 任何未在 obs 中声明的信号
- 任何 info 字段（info 为空）
- 环境内部的真实奖励数值

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4])
- angular_velocity: obs[5]
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: 当前 action（离散值 0-3）可反映推力使用情况
- other: 可构造 derived 信号，如距离目标平台的距离 = sqrt(x^2 + y^2)，水平偏角等。

## 8. 不确定或不可用的信号
- 能量消耗量：无直接能耗观测。可间接通过 action 使用次数推断。
- 机体质量、惯性参数：未观测。
- 平台精确坐标：已通过相对位置给出，无需。
- 隐式成功标志：无。
- info 任何内容：空字典，不可用。
- 环境内部奖励：已屏蔽，不可用。
- 时间截断标志：不存在。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (lander-like)
  actuator_type: discrete impulse engines (main + two orientation thrusters)
  contact_structure: two support feet (left/right) with binary contact detection
primary_objectives:
  - 到达目标平台正上方 (x≈0, y≈0)
  - 稳定停靠 (速度≈0, 倾角≈0, 双脚接触)
secondary_objectives:
  - 最小化发动机使用频率或总推力脉冲
  - 尽快到达 (速度适中，而非极速)
main_failure_risks:
  - 高速撞击目标平台，导致 crash_or_body_contact
  - 姿态失控翻滚导致接触地面或飞出视口
  - 过度使用推力导致高速或远离目标
  - 悬停过久耗尽推力后未能稳定接触
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity_and_arrival
  purpose: 推进机体向目标平台中心靠拢，并最终到达零位置。
  why_required: 环境核心任务是导航到达，无此职责无法引导智能体向目标移动。
  usable_signals: [x_position, y_position, derived distance_to_pad]
  risks: 若只奖励接近，可能鼓励以高速撞击而不减速，必须与软着陆职责配合。

- role_id: soft_landing_and_stabilization
  purpose: 在接近目标时强制减速、保持姿态水平、实现双脚平稳接触。
  why_required: 成功着陆要求低速和双脚接触，单纯到达可能忽略稳定。
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact]
  risks: 过早施加稳定压力可能导致贪婪行为，不愿离开初始位置。需要阶段门控：仅在靠近目标且速度较高时施加减速惩罚，或在接触不完整时惩罚。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当智能体已具备基本到达能力后，可作为辅助项，鼓励减少非必要推力使用。  
  usable_signals: [action (0=无推力, 1/2/3=使用发动机)]  
  risks: 若从一开始就过度强调，可能阻碍学习起飞或调整动作。应渐序引入或使用小系数。

- role_id: terminal_settlement_bonus
  condition_to_use: 在终止条件 `body_not_awake_or_settled` 且双脚接触、位置接近零时提供一次性正向反馈，明确成功。  
  usable_signals: [terminated flag from env + obs contacts and positions]  
  risks: 无法在奖励函数内部直接获取终止标志（因为 compute_reward 在每步调用，且不一定知道终止），需从当前步的 next_obs 预测是否可能已经满足条件，或依赖后续环境返回的 done 但 compute_reward 没有 done 参数。实际上，在标准 reward 接口中，done 通常不可用，除非我们定义 reward 为 per-step 并基于 next_obs 推断。因此这个职责可能对环境接口不可实现，应列为 avoid 或在外部利用 done 时由 env 提供，但当前不允许。目前 env info 为空，缺乏终止信息，故**不推荐**在 compute_reward 内部实现。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_settlement_bonus
  reason: 终止条件信息不在函数签名中（无 done 参数），无法可靠判断是否达到终止。若仅基于 next_obs 猜测“可能成功”，有可能错误奖励非终止步。当前环境在 compute_reward 阶段不应引入任意成功判断。
  forbidden_or_missing_signals: [done flag, info.success]

- role_id: angular_velocity_penalty_direct
  reason: 虽然可用 angular_velocity，但仅关注角速度而忽略绝对角度可能导致机体回旋时受惩罚，而倾斜但静止不罚。应优先使用 body_angle 稳定，angular_velocity 仅作为阻尼项，不易单独作为职责。

- role_id: pure_time_penalty
  reason: 无显式步数截断，施加每步负奖励可能导致策略倾向于尽快结束 episode（包括通过失败终止），安全性下降。慎用，除非经过训练后期校验。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_and_arrival | x_position, y_position (obs[0], obs[1]) | 无 | dense_state_signal, bounded_signal (e.g. negative L2 distance) | 可使用高斯、倒数或线性分段奖励形状，避免远距离奖励过大。 |
| soft_landing_and_stabilization | x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact (obs[2:8]) | 无 | quadratic_penalty (on velocity and angle), conditional_sparse_penalty | 结合目标距离门控：仅在距离 < threshold 时激活减速和姿态惩罚，否则忽略以允许自由调整。 |
| energy_efficiency | action (0-3) | 无能耗测量 | count_usage_penalty, action_magnitude (binary) | 可对动作非零施加小常数惩罚，或对主发动机 (action=2) 另加惩罚。需低权重或后期启用。 |
| terminal_settlement_bonus | 需要 done 和 info.success | done flag, info | sparse_terminal_reward | **不可实现**，弃用。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击平台导致 crash | episode 结束前 y_velocity 负值大，x,y 接近零，未稳定双脚接触 | 增加 soft_landing 阶段的减速惩罚强度，或在接近目标时对主引擎使用加负权重；引入垂直接近速度限制。 |
| 永远不尝试点火，自由落体失败 | action 长期为 0，y 和 x 较少变化 | 调整能量效率惩罚过重，暂时移除或降低，确保到达奖励足够引导。 |
| 过度使用主引擎导致飞出视口 | 水平位置超界，大量 action=2 | 增加越界惩罚（可通过位置信息），或限制 main_engine 使用频度（若引擎不适用横向控制）。 |
| 姿态振荡，无法稳定接触 | body_angle 和 angular_velocity 高频摆动，双脚接触短暂 | 强化角度稳定惩罚（绝对值），加入角速度阻尼，必要时代码中进行姿态稳定专项训练。 |
| 仅单脚接触就停止，未形成稳定双脚 | 终止时 left/right 接触不同时为 1 | 在 soft_landing 职责中增加双脚同时接触的条件奖励，并惩罚单脚悬空状态。 |
| 悬停在目标附近但永不接触 | y 接近 0 但 y_velocity 未归于零，双脚未接触 | 引入接近零速度的奖励，并强化 soft_landing 中接触信号作用，或施加时间压力（后期可选）。 |
