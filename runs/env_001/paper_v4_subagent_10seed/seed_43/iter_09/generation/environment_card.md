# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 刚体着陆/停靠任务。agent 从视口上方中央附近出发，带有一个随机初始力。主目标是尽快飞到中央的目标着陆垫，并在垫上稳定停靠（安全接触）。次要目标是在完成主目标的前提下，尽量缩短飞行时间和减少发动机推力使用。agent 需要学会接近目标、降低速度、保持姿态稳定、并以双腿接触垫面实现软着陆。
不该混淆的目标：不能把存活时间当作主要奖励信号（任务不是为了活得久，而是尽快到达并停稳）；不能把能耗降低到影响到达任务的程度，能耗只是附属优化。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达指定的目标位置（中央着陆垫）并实现稳定接触，属于导航到达类任务。附属有时长和能耗优化，但不是多目标冲突中的多个同等重要目标，因此不选 multi_objective_task。动力学子类型进一步细分为 goal_approach_and_soft_contact（接近目标 + 低速稳定接触）。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: [8]
- dtype: float32 (以实际环境为准，通常为 float)
- obs[0]: x_position，含义：水平坐标（相对目标着陆垫中心），reward_usable: true
- obs[1]: y_position，含义：垂直坐标（相对垫面高度），reward_usable: true
- obs[2]: x_velocity，含义：水平线速度，reward_usable: true
- obs[3]: y_velocity，含义：垂直线速度，reward_usable: true
- obs[4]: body_angle，含义：机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity，含义：角速度，reward_usable: true
- obs[6]: left_support_contact，含义：左支撑腿接触标志（0.0 或 1.0），reward_usable: true
- obs[7]: right_support_contact，含义：右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不做任何推力，相当于滑行/自由落体
- action 1: left_orientation_engine，启动左侧姿态发动机（通常产生逆时针力矩）
- action 2: main_engine，启动主发动机（通常产生向上推力，可能同时影响姿态）
- action 3: right_orientation_engine，启动右侧姿态发动机（通常产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功标记。推测成功情况为：双腿接触着陆垫、机体接近垫中心、速度极低，此时可能触发 `body_not_awake_or_settled` 终止。
- failure-like termination: 
  - `crash_or_body_contact`: 除支撑腿之外的身体部分接触地面或障碍物（撞击/侧翻等）。
  - `horizontal_position_outside_viewport`: 水平飞出画面边界。
  - 因姿态或位置异常导致的无效停稳（例如单腿接触、翻倒后静止）也会触发 `body_not_awake_or_settled`，但不应视为成功。
- ambiguous termination: `body_not_awake_or_settled` 在成功软着陆和失败后静止时均可能触发，仅凭该事件无法区分。
- truncation: 无，只有终止 (terminated) 模式。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 空字典 `{}`（无任何 info 字段可用）
- forbidden_or_uncertain_info_fields: 所有未声明的字段均不可用；不存在 success、failure 等语义标志。

成功/失败只能通过 episode 结束时的观测信号间接推断：
- 推断成功：`left_support_contact == 1.0 and right_support_contact == 1.0`，`abs(x_position)` 很小，`abs(y_position)` 很小（接近垫面），线速度和角速度接近于零。
- 推断失败：上述条件不满足，或者观测到极端速度/位置值后终止。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前步观测）
- action（当前步动作）
- next_obs（下一步观测）
- info 中明确允许的字段（当前为空，不可用任何字段）
- training_progress 只有 prompt 明确允许时才用，此处未允许，禁止使用

禁止使用：
- original_reward 或任何官方奖励
- 未声明的 info 字段
- 未声明的 obs 切片或 any 环境私有状态

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position)，可计算到目标垫的距离（欧氏距离或加权距离），可计算相邻两步的距离变化。
- velocity: obs[2] (x_velocity), obs[3] (y_velocity)，可用于惩罚急停撞击，或在接近目标时鼓励减速。
- orientation: obs[4] (body_angle)，obs[5] (angular_velocity)，可鼓励保持竖直、减少旋转。
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact)，可给予双腿着陆奖励；单腿或不正常接触时不予奖励。
- action/engine: action 索引 1,2,3 表示使用引擎，0 为无推力，可做能耗惩罚；也可结合动力学判断推力强度。
- 间接成功信号 derived_possible: episode 终止时，通过位置、接触和速度组合推断是否成功软着陆，可用于终端奖励。

## 8. 不确定或不可用的信号
- 无直接的成功/failure 标志，需依赖 derived_possible 推断。
- 无剩余时间或步数信息（若需时间惩罚，需自己计时，但本任务未强制要求优化时长，不建议添加时间微分奖励以免过度复杂化）。
- 无风力、初始随机力等环境的隐藏状态。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: main_engine_and_two_orientation_engines
  contact_structure: two_support_contacts
primary_objectives:
  - reach the target landing pad (minimize horizontal/vertical distance)
  - make safe soft contact with both legs on the pad
  - stabilize body (zero/low velocity and angular velocity at touchdown)
secondary_objectives:
  - minimize time-to-land (implicitly through progress bonus)
  - minimize engine usage (fuel efficiency)
main_failure_risks:
  - overshooting or crashing outside viewport
  - landing with body tilt/one leg causing crash
  - excessive fuel consumption with no progress
  - premature engine cutoff leading to hard landing
  - over-controlling and oscillation
```

## 10. 奖励职责拆解 reward_role_decomposition

### 骨架选择推理小结
根据任务核心“离目标更近了吗”，主信号算子族选定为 **delta(distance)**（朝目标每一步的距离减少量），以避免纯 proximity 奖励造成的悬停陷阱。当距离很小、双腿接触时，配合接触成功奖励和稳定奖励；当偏离目标时，不因单纯存活而获得正分。

### 10.1 主职责 mandatory_roles
- role_id: approach_progress
  purpose: 鼓励每一步向目标着陆垫靠近。
  why_required: 核心目标为到达指定位置，必须提供密集的进度信号引导 agent 学习轨迹走向。
  usable_signals: [x_position, y_position] (计算