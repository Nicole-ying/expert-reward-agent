# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 近似飞行器/着陆器轨迹优化任务。主体从画面顶部中央附近开始，承受随机初始力。核心目标是尽快到达画面中央的目标垫，并稳定停留（settle），同时尽量少用引擎推力。智能体需要学习：  
- 接近目标位置；  
- 降低速度，以较小的速度接触；  
- 保持身体角度稳定（接近垂直），避免倾覆；  
- 安全、轻柔地接触目标垫。  

不应混淆的目标：不存在碰撞回避、路径跟踪精度或动态障碍物规避；唯一位置目标是中心垫。节能与快速到达虽然是描述的一部分，但它们服务于主到达目标，不属于独立的多目标冲突。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心任务是到达指定目标位置并稳定，所有的动力学控制（速度、姿态、接触）均服务于该到达过程，附加的节能/快速是次要优化指标，不构成两个平等冲突的目标。布局与经典月球着陆器/着陆垫任务高度相似，属于典型导航到达。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float64  
- obs[0]: x_position（相对于目标垫的水平坐标），reward_usable: true  
- obs[1]: y_position（相对于垫面高度的垂直坐标），reward_usable: true  
- obs[2]: x_velocity（水平线速度），reward_usable: true  
- obs[3]: y_velocity（垂直线速度），reward_usable: true  
- obs[4]: body_angle（身体朝向角），reward_usable: true  
- obs[5]: angular_velocity（角速度），reward_usable: true  
- obs[6]: left_support_contact（左支撑接触标志，0.0 或 1.0），reward_usable: true  
- obs[7]: right_support_contact（右支撑接触标志，0.0 或 1.0），reward_usable: true  

所有维度均可用于奖励计算。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: 无引擎（no_engine）—— 不施加任何推力  
- action 1: 左方向引擎（left_orientation_engine）—— 点燃侧向姿态引擎，产生向左的力矩/力  
- action 2: 主引擎（main_engine）—— 点燃主体引擎，产生向上推力（可能也影响姿态）  
- action 3: 右方向引擎（right_orientation_engine）—— 点燃另一侧姿态引擎，产生向右的力矩/力  

（动作含义为对称侧向引擎与主推力，用于横向姿态调整和垂直减速。）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 可能为 body_not_awake_or_settled 中的“settled”子条件（身体稳定停在目标垫上），但无法从返回的终止信号中区分出是否成功，仅能从物理状态推断。
- failure-like termination: crash_or_body_contact（可能指与其他表面或非垫部分发生碰撞）、horizontal_position_outside_viewport（水平越界）
- ambiguous termination: body_not_awake_or_settled —— 既包含成功稳定，也可能包含因损坏/失能而停止运动。
- truncation: 依据 step source 无截断，仅以上三种终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: （空字典，无可用 info 字段）  
- forbidden_or_uncertain_info_fields: 任何未在 source 中声明的 info 键均不可用。尤其不能假设 info 中存在 "success"、"failure"、"termination_reason" 等字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测字典或数组，对应上述 8 维）
- action（当前步动作）
- next_obs（下一步观测，同样 8 维）
- info（空字典，不使用）
- training_progress（仅在 prompt 明确要求时允许，当前未提及，故禁用）

禁止使用：
- original_reward（官方奖励被屏蔽，不可用）
- 任何未声明的 info 字段
- 任何未声明的 obs 维度切片（但全部 8 维均已声明，可使用）

## 7. 可用于奖励函数的信号
- position: obs[0] x_position, obs[1] y_position（相对目标垫的水平和垂直偏移）  
- velocity: obs[2] x_velocity, obs[3] y_velocity  
- orientation: obs[4] body_angle, obs[5] angular_velocity  
- contact: obs[6] left_contact, obs[7] right_contact（指示支撑腿是否接触）  
- action/engine: 当前 action 值（0~3），可用于识别使用主引擎或姿态引擎  
- other: 无  

## 8. 不确定或不可用的信号
- 碰撞/越界标志：无法从 obs 中可靠推断是否发生 crash_or_body_contact 或 horizontal_outside_viewport（无明确阈值或直接信号），因此不能直接作为奖励条件。  
- 成功稳定标志：无显式 success 标志；可推断“接近目标且速度极小且双脚接触”为成功，但在终止前的 step 可能尚未达到此状态，依赖此信号绘制最终奖励有风险。  
- 环境步数/时间：不可用。  
- 燃料剩余量：不可用。  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_rigid_body_2d
  actuator_type: thrusters (one main, two lateral orientation)
  contact_structure: two_point_contact (left and right support legs)
primary_objectives:
  - reach_target_position_and_settle
secondary_objectives:
  - minimize_fuel_usage
  - minimize_arrival_time
main_failure_risks:
  - crashing_into_ground_or_obstacles
  - moving_outside_horizontal_bounds
  - excessive_oscillation_or_tipping
  - landing_too_hard (high velocity at contact)
  - failure_to_slow_down_and_overrun
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 驱动智能体向目标垫移动（减小 x,y 偏移）  
  why_required: 到达目标是任务的根本要求，缺少位置奖励无法完成任何接近行为  
  usable_signals: [x_position, y_position]  
  risks: 单纯基于距离的奖励可能导致高速冲向目标，忽略接触姿态与速度控制  

- role_id: soft_landing_settling  
  purpose: 在接近目标时抑制速度，鼓励双脚轻柔接触并稳定  
  why_required: 若没有软着陆引导，智能体可能以危险速度冲击垫面或无法停止摆动  
  usable_signals: [x_velocity, y_velocity, left_contact, right_contact, x_position, y_position]  
  risks: 过早强调软着陆可能抑制必要的探索和高速阶段的学习；需要条件化或权重自适应  

- role_id: orientation_stability  
  purpose: 维护身体角度接近 0（竖直），避免倾覆和不安全接触  
  why_required: 倾覆直接导致失败或不良接触，必须在全程加以约束  
  usable_signals: [body_angle, angular_velocity]  
  risks: 过度惩罚旋转可能使得智能体不取使用侧向引擎，影响横向调节  

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency  
  condition_to_use: 整个 episode 均可施加轻微动作惩罚，但主要起辅助作用；也可在接近目标后加大惩罚。  
  usable_signals: [action]  
  risks: 惩罚过重会阻碍使用引擎对抗重力，导致无法减速或悬停；需要保持小量级  

### 10.3 慎用/禁用职责 avoid_roles
- role_id: failure_penalty  
  reason: 无法从观测中可靠检测 crash、out_of_viewport 等明确失败条件，错误地施加惩罚会损坏学到的正常接近行为。终止状态也不提供显式信号。  
  forbidden_or_missing_signals: [failure_detection_signal]  

- role_id: time_bonus_or_step_penalty  
  reason: 环境未提供步数或时间戳，无法直接施加每步惩罚，也不应通过训练进度强行构造。  
  forbidden_or_missing_signals: [step_count, elapsed_time]  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | (none) | dense_state_signal (e.g., -distance), potential_shaping | 可结合位置差构造连续衰减奖励 |
| soft_landing_settling | x_velocity, y_velocity, left_contact, right_contact, x_position, y_position | (none) | quadratic_penalty (速度), contact_bonus, gating_by_spatial_threshold | 靠近目标时激活，利用接触与低速信号 |
| orientation_stability | body_angle, angular_velocity | (none) | quadratic_penalty (角度), angular_velocity_penalty | 全程施加，可与其他项叠加 |
| fuel_efficiency | action | (none) | action_penalty (按权重扣除) | 注意保持量级较小，避免抑制必要推力 |
| failure_penalty | — | failure_detection_signal (越界/碰撞) | — | 不在当前可用信号集合中，无法可靠实现 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过早坠毁或横向飞出 | 早期 episode length 短，x_position 方差极大或越界 |