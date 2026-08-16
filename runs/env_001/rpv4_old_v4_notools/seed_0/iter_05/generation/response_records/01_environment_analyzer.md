# Response Record

# 匿名环境理解卡片

## 1. 任务目标
任务是在 2D 视口中控制一个带引擎和双腿的飞行器，从顶部中心附近出发，尽快且平稳地降落到位于中央的目标平台上，并保持稳定姿势。核心目标是**到达目标平台并安全停靠**；次要目标包括：**最小化引擎推力（燃料消耗）**、**缩短飞行时间**、**保持着陆时姿态水平**、**两腿平稳触地**。任务不要求连续行进，也不涉及抓取物体。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定位置（目标垫）并稳定停止，非连续行走或生存平衡。次要目标（时间、燃料、姿态）服从主目标，不属于权重相当的多目标冲突任务。

动力学子类型 dynamics_subtype: goal_approach_and_soft_contact  
（接近目标并低速、稳定接触/停靠）

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推断）
- obs[0]: x_position（相对目标垫水平坐标），reward_usable: true
- obs[1]: y_position（相对平台高度垂直坐标），reward_usable: true
- obs[2]: x_velocity（水平线速度），reward_usable: true
- obs[3]: y_velocity（垂直线速度），reward_usable: true
- obs[4]: body_angle（机体倾斜角度），reward_usable: true
- obs[5]: angular_velocity（角速度），reward_usable: true
- obs[6]: left_support_contact（左腿触地标志），值 0.0 或 1.0，reward_usable: true
- obs[7]: right_support_contact（右腿触地标志），值 0.0 或 1.0，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（不做任何推进）
- action 1: left_orientation_engine（点火左侧姿态引擎，产生一个方向的旋转力矩）
- action 2: main_engine（点火主引擎，产生垂直向上的推力）
- action 3: right_orientation_engine（点火右侧姿态引擎，产生相反方向的旋转力矩）

（注：动作空间为离散，每个动作代表同时启用特定引擎的组合，但无法同时启用多个。未提供引擎推力、力矩大小或燃料消耗速率的具体数值，只能从观测变化间接感知。）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（飞行器静止在目标平台上且两腿平稳接触，可视为成功着陆）
- failure-like termination: crash_or_body_contact（本体与地面或其他物体非法接触，或角度/速度过大导致坠毁），horizontal_position_outside_viewport（飞出水平边界）
- ambiguous termination: 无（依据三个条件区分）
- truncation: 未提供，假设没有时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无显式标志）
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空，不可使用任何 info 键）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用，因为 step 返回的是空字典

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，shape (8,)），可使用全部 8 维
- action（当前动作，0~3 整数）
- next_obs（下一时刻观测）
- training_progress（仅当环境定义明确说明可用时，此处未提供，应避免使用）

禁止使用：
- original_reward（官方奖励已掩蔽，不可使用）
- info（因为为空，任何尝试访问 info 字段均不可用）
- 未在观测空间中声明的任何其他信息

## 7. 可用于奖励函数的信号
- position: obs[0]（x坐标）, obs[1]（y坐标）
- velocity: obs[2]（x速度）, obs[3]（y速度）
- orientation: obs[4]（角度）, obs[5]（角速度）
- contact: obs[6]（左腿触地）, obs[7]（右腿触地），二值信号
- action/engine: 可获取当前动作，推断使用了哪个引擎（0=无，1=左姿态，2=主引擎，3=右姿态）。可用于最小化推力使用。
- other: 无其他明确信号

## 8. 不确定或不可用的信号
- 显式着陆成功或失败标志：不存在
- 累计燃料消耗或剩余燃料：不可用
- 时间步数或计时器：未提供（但可根据轨迹长度间接推断，不过非显式信号）
- 平台中心精确坐标：通过 obs[0] 和 obs[1] 接近 0 时表示位于目标上方，平台高度定义为某固定值（但未提供具体数值，可从观测范围估计）
- 安全着陆所需的速度/角度阈值：环境未给出，需通过尝试确定合理范围（如 |angle| < 阈值, |velocity| < 阈值）
- 地面高度或地形信息：不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: planar_vehicle_with_two_legs
  actuator_type: discrete_thrusters (main engine, two orientation thrusters)
  contact_structure: two_point_contact_on_legs
primary_objectives:
  - reach_target_pad (obs[0]≈0, obs[1]≈0)
  - soft_landing (low velocity, near-zero angle, both legs in contact)
secondary_objectives:
  - minimize_engine_usage (sparse action 2, 1, 3)
  - minimize_time_to_landing (implicit: quick settling)
  - keep_stable_orientation_during_flight (penalize large angles)
main_failure_risks:
  - crashing_body (excessive speed or angle when touching ground)
  - going_out_of_bounds (horizontal position too large)
  - landing_on_side_or_tipping_over (one leg contact, large angle)
  - excessive_fuel_use_or_oscillation (alternating actions frequently)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_reaching_progress  
  purpose: 鼓励向目标接近，减少水平与垂直距离  
  why_required: 任务是到达目标垫，直接衡量进度  
  usable_signals: [obs[0], obs[1], next_obs[0], next_obs[1]]  
  risks: 单纯距离奖励可能让飞行器高速撞向目标，需配合减速和姿态约束

- role_id: soft_landing_on_both_legs  
  purpose: 促进两腿同时触地，保持身体水平且低速  
  why_required: 成功着陆的定义，否则视为坠毁  
  usable_signals: [obs[4], obs[6], obs[7], obs[2], obs[3]]  
  risks: 不恰当的正奖励可能导致提前终止或奖励消失

- role_id: action_sparsity  
  purpose: 惩罚主引擎和姿态引擎的使用，鼓励无动作滑翔  
  why_required: 明确要求“尽可能少用推力”  
  usable_signals: [action]  
  risks: 过度惩罚可能阻碍必要调整，需要平衡

### 10.2 条件职责 conditional_roles
- role_id: terminal_success_bonus  
  condition_to_use: 只有通过观测判断成功着陆时（如 next_obs 满足 contact both, low speed, near-zero angle, position near zero），并且 done=True（但 done 信号不可直接获取，需外部传入或根据 next_obs 推断？由于无法获取 done 标志，可能需要从 last observation 判断）  
  usable_signals: [next_obs[6], next_obs[7], next_obs[2], next_obs[3], next_obs[4]]  
  risks: 错误识别成功会污染训练；若无 done 标志，可仅在最后一帧给予奖励（需环境提供 done 参数，但 reward 函数规范中未显式包含 done，需注意：调用时可能传入额外参数如 terminated。如果未提供，则无法使用。根据接口，只有 compute_reward(obs, action, next_obs, original_reward, info, training_progress)，所以 done 不可用，此条件职责可能无法安全实现，除非用 next_obs 的接触+零速度推测后由外部逻辑给予，但 reward 函数本身无法知道 episode 是否结束。因此，慎用）

- role_id: orientation_stabilization_during_flight  
  condition_to_use: 当飞行器尚未着陆时（根据接触标志和高度判断）  
  usable_signals: [obs[4], obs[5]]  
  risks: 着陆时角速度惩罚可能妨碍最后姿态调整

### 10.3 慎用/禁用职责 avoid_roles
- role_id: binary_crash_penalty  
  reason: 无法获取可靠 crash 信号（无 info 标志）；通过观测阈值硬判 crash 可能导致误判（如腿触碰但角度大，可能被环境视为 crash，但 reward 无法区分 crash 与正常着陆终止）。应依赖环境返回的终止和 next_obs 综合判断，但在仅 reward 函数中无法获得 done 和真实终止原因，故应避免用巨大负奖励直接惩罚 crash，改为通过 soft landing 引导

- role_id: explicit_success_flag_dependent_bonus  
  reason: 缺少任何 info 标志，无法使用

- role_id: time_penalty  
  reason: 没有时间步计数器信号，无法可靠施加，仅能通过总步数间接影响，但不建议在单步 reward 中硬编码

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_reaching_progress | obs[0], obs[1], next_obs[0], next_obs[1] | 目标位置具体值（但知道目标在0,0） | distance_change_reward, dense_state_signal, linear_progress | 使用 -（当前位置到目标距离）或距离缩减量 |
| soft_landing_on_both_legs | obs[6], obs[7], obs[4], obs[3], next_obs相应值 | 安全阈值（需实验定义） | bounded_signal, and_combination, product | 当双腿接触且角度速度都在允许范围内时给予正向奖励；可作为 done 条件外的连续塑造 |
| action_sparsity | action | 引擎推力大小（无法量化燃料消耗） | discrete_action_penalty, step_cost | 对 action 1/2/3 施加固定小惩罚，action 0 不罚 |
| terminal_success_bonus | next_obs[6], next_obs[7], next_obs[2], next_obs[3], next_obs[4] | done 标志，真实成功/失败状态 | sparse_event_bonus | 依赖外部在 episode 结束时传递额外标志或使用最后一帧观测，但奖励函数若无 done 参数则无法可靠实施，可能需降级为条件启用 |
| orientation_stabilization_during_flight | obs[4], obs[5] | 飞行/着陆分割 | quadratic_penalty on angle, angular velocity | 仅在 contact==0 时激活，避免干扰着陆 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速坠毁 | episode 长度极短，最终 y_velocity 很大，或角度严重倾斜 | 增加 soft landing 惩罚权重，降低 progress 权重 |
| 悬停不下降或过度盘旋 | 大量使用姿态引擎，y_position 不变或增加，燃料耗尽 | 调整 goal_reaching 信号优先竖直接近，加强主引擎鼓励 |
| 落入边界外 | x_position 超限终止 | 增加水平位置误差惩罚 |
| 仅单腿着陆并翻倒 | 终止时一条腿接触、角度大 | 强化双侧接触奖励，增加角度惩罚，甚至奖励 landing_legs_both |
| 引擎使用过于频繁/振荡 | action 频繁在 1/3 间切换 | 增大 action sparsity 惩罚，或加入动作变化平滑损失（但需连续观测前一个动作，此处未提供，可考虑使用动作保留奖励） |
| 未学会快速下降，拖延时间 | episode 很长但不 crash，最终依靠长时间滑翔才着陆 | 加入隐含时间惩罚（如每步微小负奖励），或整体 shaping 势能下降优先 |
