# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 着陆器轨迹优化任务。主体（着陆器）从视角顶部中央附近出发，受初始随机作用力，目标是以最快速度到达画面中心的着陆垫并稳定停靠（settle），同时尽可能少地使用引擎推力。智能体需要学会：  
- 精确接近目标垫（将相对位置向量驱近于零）  
- 减速至软接触（降低线速度与角速度，保持姿态水平）  
- 利用两条支撑腿与垫形成安全、稳定的接触  
- 附属优化：降低燃料消耗（减少引擎动作）和减少用时  

该任务不应被混淆为纯粹的平衡、抓取或多足行走；其核心是到达并稳定停留在指定目标位置。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达并稳定在中心着陆垫（位置目标），附属优化速度与能耗。不存在多个冲突等权目标，目标垫到达是硬性成功前提，其他均为辅助性偏好。

动力学子类型 dynamics_subtype: goal_approach_and_soft_contact （接近目标并低速、稳定接触/停靠）

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64  
- obs[0]: x_position – 相对于着陆垫的水平坐标，reward_usable: true  
- obs[1]: y_position – 相对于垫高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity – 水平线速度，reward_usable: true  
- obs[3]: y_velocity – 垂直线速度，reward_usable: true  
- obs[4]: body_angle – 主体姿态角（方向），reward_usable: true  
- obs[5]: angular_velocity – 角速度，reward_usable: true  
- obs[6]: left_support_contact – 左支撑腿是否触垫（0.0或1.0），reward_usable: true  
- obs[7]: right_support_contact – 右支撑腿是否触垫（0.0或1.0），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine – 不点火，依靠当前速度和重力演化  
- action 1: left_orientation_engine – 点燃左侧姿态引擎（产生扭矩，逆时针？）  
- action 2: main_engine – 点燃主发动机（产生向上的推力，可能同时影响姿态）  
- action 3: right_orientation_engine – 点燃右侧姿态引擎（产生反向扭矩，顺时针？）  

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（身体不再活跃/已稳定停靠）——若同时观察到 left_contact 且 right_contact 且速度/角速度极低，很可能为成功着陆。但 step 源码中 info 为空，不能直接作为显式成功标志。  
- failure-like termination: crash_or_body_contact（除垫外与其他表面碰撞，或非预期身体接触），以及 horizontal_position_outside_viewport（水平超出视口）。这两种情况可视为着陆失败。  
- ambiguous termination: body_not_awake_or_settled 可能因物理休眠触发，但需要结合接触和速度才能确认成功；若未触垫却休眠（例如悬停在半空停止计算）应视为异常终止，不可直接用作奖励依据。  
- truncation: 无时间截断迹象（源码未显示步骤限制），但实际环境中可能存在。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，没有 success 字段）  
- explicit_failure_flag_available: false  
- allowed_info_fields: （空）  
- forbidden_or_uncertain_info_fields: info 中任何未声明的字段均不可使用；严禁假设 info["success"]、info["failure"]、info["termination_reason"] 等存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (前一步观察)
- action (前一步动作)
- next_obs (下一步观察)
- (仅当调用者明确允许时) training_progress
- info 中当前已知没有字段，因此默认为不可用字段，除未来明确声明不可使用。

禁止使用：
- original_reward（原环境奖励）
- 未声明的 obs 切片（如 obs 维度大于 8 则不可使用）
- 未声明的 info 字段
- 对真实环境名的任何依赖

## 7. 可用于奖励函数的信号
- position: 相对目标垫的水平和垂直位置 (next_obs[0], next_obs[1])  
- velocity: 水平和垂直线速度 (next_obs[2], next_obs[3])  
- orientation: 姿态角 (next_obs[4])  
- angular velocity: 角速度 (next_obs[5])  
- contact: 左右支撑触垫标志 (next_obs[6], next_obs[7])  
- action/engine: 动作选择（0-3），可用于燃料消耗或动作惩罚  
- other: 无

## 8. 不确定或不可用的信号
- 显式成功标志（无）  
- 目标垫位置全局坐标（仅有相对位置）  
- 剩余燃料量（未提供）  
- 碰撞具体类型（crash_or_body_contact 为组合条件，无法在 obs 中区分）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D_rigid_body_with_two_legs
  actuator_type: main_vertical_thruster_and_two_side_orientation_thrusters
  contact_structure: bilateral_leg_contacts_on_target_pad
primary_objectives:
  - reach_target_position: drive (x, y) relative to pad to near zero
  - safe_settlement: achieve low velocity and low angular velocity while both legs contact the pad
secondary_objectives:
  - fuel_efficiency: minimize usage of main/orientation engines
  - time_minimization: complete task as quickly as possible
main_failure_risks:
  - high_velocity_impact: contacting pad or terrain with excessive speed causing crash
  - horizontal_excursion: drifting outside the allowed horizontal window
  - orientation_loss: excessive tipping leading to side contact or failure to settle
  - excessive_hovering: using too much fuel to maintain altitude without progressing
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: position_approach  
  purpose: 鼓励着陆器向目标垫位置靠拢，使相对位置 (x, y) 趋近于零。  
  why_required: 到达目标区是任务的根本成功条件。  
  usable_signals: [next_obs[0], next_obs[1]] （可使用距离或分量）  
  risks: 如果权重过大，可能导致高速冲向目标而忽略减速，引发撞击。

- role_id: soft_landing  
  purpose: 在接近目标区域时或全局鼓励降低线速度和角速度，同时倾向双腿同时接触垫。  
  why_required: 安全停靠（settle）要求速度低、姿态平且接触良好。  
  usable_signals: [next_obs[2], next_obs[3], next_obs[4], next_obs[5], next_obs[6], next_obs[7]]  
  risks: 过早激活软着陆奖励可能使代理不敢移动；应逐步增强或与位置接近关联。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency  
  condition_to_use: 全程可施加，但建议在接近着陆区或接触后降低权重，避免因惩罚动作而不敢推进。或仅在完成靠近/软着陆后施加回报。  
  usable_signals: [action] （非零动作视为耗油）  
  risks: 过度惩罚动作可能导致悬停或过早放弃引擎，使着陆器直接坠落。

- role_id: time_penalty  
  condition_to_use: 每步施加小的负奖励以鼓励快速完成，但需配合失败终止时的较大负奖励，防止代理选择早期撞击来减少总步数。  
  usable_signals: [none; constant per step]  
  risks: 可能误导代理优选快速撞毁而非安全着陆。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_bonus  
  reason: 环境未提供显式成功标志，info 为空；无法可靠检测何时达到成功状态。若自行推断（如速度<ε 且双触垫）可能误报。建议改为基于软着陆与位置奖励自然形成，避免在 reward 函数中加入 if done: bonus。  
  forbidden_or_missing_signals: [explicit success flag]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| position_approach | next_obs[0], next_obs[1] | 