# 匿名环境理解卡片

## 1. 任务目标
主体是一个2D飞行器/着陆器类型的轨迹优化问题。智能体从视口顶部中央附近以随机初始力出发，目标是以**最短时间**到达中央目标垫（target pad）并稳定停靠，同时**尽可能减少引擎推力**的使用。  
成功意味着智能体轻柔地接触目标垫、保持直立姿态、速度接近于零，并稳定下来。失败来源于坠毁、超出水平边界或无法稳定。  
注意：不能将“快速”和“低能耗”视为等权核心目标——核心是“到达并稳定停靠”，速度与能耗是附属优化目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定目标位置（目标垫）并停靠，动作空间与观测空间明确支持位置误差反馈，不存在多个权重相当、相互冲突且无法区分主次的目标，附属的速度/能耗要求不构成 multi-objective_task。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断，因未明确给出但通常为 float)
- obs[0]: x_position — 相对于目标垫的水平坐标（归一化/缩放未知，但语义为横向误差）。reward_usable: true
- obs[1]: y_position — 相对于目标垫高度的垂直坐标。reward_usable: true
- obs[2]: x_velocity — 水平线速度。reward_usable: true
- obs[3]: y_velocity — 垂直线速度。reward_usable: true
- obs[4]: body_angle — 机体方向角。reward_usable: true
- obs[5]: angular_velocity — 角速度。reward_usable: true
- obs[6]: left_support_contact — 左支撑脚接触标志（1.0 表示接触）。reward_usable: true
- obs[7]: right_support_contact — 右支撑脚接触标志（1.0 表示接触）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0 (no_engine): 不点火
- action 1 (left_orientation_engine): 左侧姿态引擎点火
- action 2 (main_engine): 主引擎（向下推力）点火
- action 3 (right_orientation_engine): 右侧姿态引擎点火

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  “body_not_awake_or_settled” 可能表示机体进入休眠/稳定的状态。当同时满足位置接近目标、速度很低、姿态垂直且存在支撑接触时，极可能对应成功着陆。  
- failure-like termination:  
  “crash_or_body_contact”（与地面/物体不安全碰撞）、  
  “horizontal_position_outside_viewport”（水平出界）属于明确的失败类终止。
- ambiguous termination:  
  “body_not_awake_or_settled” 在远离目标垫时也可能触发（如坠毁后静止），需结合其他观测信号才能判定成功与否。
- truncation: 未提及。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无 (info dict 为空 {})
- forbidden_or_uncertain_info_fields: 全部 info 字段均不存在且禁止依赖。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段（当前无）
- training_progress（仅当后续 prompt 明确要求时可用）

禁止使用：
- original_reward（官方奖励已遮蔽，不得复原）
- 任何未声明的 info 字段
- 任何未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: x_position, y_position — 可直接计算到目标的欧氏距离及变化量。
- velocity: x_velocity, y_velocity — 可用于评估接近平稳、能量或冲击。
- orientation: body_angle — 保持竖直（接近0）的约束信号。
- contact: left_support_contact, right_support_contact — 用于判断是否安全接触垫子。
- action/engine: 离散动作可转化为引擎使用代价。
- derived_possible（间接推断）:
  - 成功着陆推断：若 episode 因 body_not_awake_or_settled 终止，且在终止前的最后一步 next_obs 中 (|x_position| 小, |y_position| 小, 速度低, |body_angle| 小, 至少一个支撑接触) → 可视为成功事件，用于终端奖励。
  - 坠毁推断：若 episode 终止且 body_angle 巨变、y_position 突变、接触信号异常激活 → 可推导失败。
  - 注意：这些推断不能在 compute_reward 中直接访问 done 标志，但可以通过检查 next_obs 的状态组合来实现隐式终端奖励——step 执行后调用 compute_reward 时若 next_obs 恰好是终态，则可用该状态判定。

## 8. 不确定或不可用的信号
- 官方原始奖励（已遮蔽）
- info 中的任何字段（空字典）
- 任务进度百分比（training_progress 未声明可用）
- 全局地图或障碍物位置（观测未包含）
- 任何形式的“奖励塑形参考值”或“基准轨迹”

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 类着陆器 (lander-like body with two support legs)
  actuator_type: 一个主发动机(向下推力) + 两个姿态发动机(左右转动力矩)
  contact_structure: 两个底部支撑脚(left/right support)
primary_objectives:
  - 到达目标垫中心并稳定停靠
secondary_objectives:
  - 尽快完成(时间隐含最优性)
  - 使用尽量少的引擎推力(节能)
main_failure_risks:
  - 过度姿态纠正导致坠毁/碰撞
  - 主引擎推力过大导致高速撞击或弹跳
  - 水平方向漂移出视野
  - 接触地面时姿态过度倾斜无法恢复
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: progress_to_target
  purpose: 鼓励智能体每一步接近目标垫的欧氏距离。
  why_required: 导航到达任务的核心进展度量，避免悬停或绕圈。
  usable_signals: [x_position, y_position] (obs 0,1 与 next_obs 0,1)
  risks: 纯距离 reward 可能导致“快速冲过目标”而无法稳定，需配合 soft‑landing 约束。

- role_id: soft_landing_incentive
  purpose: 在接近目标垫时奖励垂直姿态、低速、左右支撑同时接触的状态，引导最终安全停靠。
  why_required: 纯距离 reward 不足以保证稳定着陆；若无该角色，策略可能选择高速硬着陆或只短暂接触即弹跳。
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 若权重过高可能导致 agent 在垫子边缘保持