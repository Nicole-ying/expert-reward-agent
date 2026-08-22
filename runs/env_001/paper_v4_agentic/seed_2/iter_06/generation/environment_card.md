# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器（带有主引擎与两个方位引擎）从视口顶部中心附近出发，尽可能快地飞抵并稳定在中央目标平台上。主要目标是到达目标位置并保持安全、低速、姿态水平的着陆；次要目标是在过程中尽量少使用引擎推力（节约能耗）。不应将快速到达、姿态稳定或省燃料视作与主目标等同的多个独立目标——它们都是为主着陆服务的附属要求。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定目标位置（中央目标平台），能耗最小化和速度要求是附属优化项，不存在多个权重相当且冲突的核心目标，因此归入导航目标到达类。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（推测为标准连续值，也可能是 float32）
- 各维含义：
  - obs[0]: x_position，水平坐标（相对目标垫横向偏移），reward_usable: true
  - obs[1]: y_position，垂直坐标（相对垫面高度），reward_usable: true
  - obs[2]: x_velocity，水平线速度，reward_usable: true
  - obs[3]: y_velocity，垂直线速度，reward_usable: true
  - obs[4]: body_angle，机体方向角（可能以竖直为 0），reward_usable: true
  - obs[5]: angular_velocity，角速度，reward_usable: true
  - obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true
  - obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine —— 不做任何推力输出
  - action 1: left_orientation_engine —— 点燃左侧方位引擎（产生逆时针力矩）
  - action 2: main_engine —— 点燃主引擎（产生向前或向上的推力，具体方向需结合机体角度）
  - action 3: right_orientation_engine —— 点燃右侧方位引擎（产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - `body_not_awake_or_settled`：机体进入静止或稳定状态，很可能是在目标平台上成功着陆后触发。此条件在观测未提供直接标志，但表现为所有速度、角速度归零且接触标志为真的稳态。
- failure-like termination:
  - `crash_or_body_contact`：机体非支撑部分碰撞地面或障碍物，导致损毁。
  - `horizontal_position_outside_viewport`：水平坐标超出允许范围（飞离视口）。
- ambiguous termination:
  - 无。
- truncation:
  - 根据代码，`terminated` 在三种条件之一触发时设为 True，无其他截断，`truncated=False` 恒成立，因此不存在超时截断（除非底层实现有限定最大步数，但未在 spec 中体现，视为无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

注：成功与失败的推断只能通过观测信号间接进行。例如，当 `left_support_contact` 与 `right_support_contact` 均为 true，且 `x_position`、`y_position` 接近 0，速度、角速度近乎 0 时，可推定为成功着陆终止（derived_possible）。撞毁可能伴随接触信号突变或速度极大，但无可靠单步信号，因此不推荐直接用于奖励，可转为避免边界和冲击的策略。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（前一步观测）
- action（当前步执行的动作）
- next_obs（执行动作后、终止前的观测）
- info（但当前环境 info 始终为空，实际不可用）
- training_progress 仅当 prompt 明确允许时才使用（当前未明确允许，慎用）

禁止使用：
- original_reward（被屏蔽的官方奖励）
- 任何未在 observation_space 中声明的 obs 切片
- 任何 info 字典字段（因其为空）

## 7. 可用于奖励函数的信号
- position：`next_obs[0]` (x)，`next_obs[1]` (y) — 相对目标垫的位置
- velocity：`next_obs[2]` (vx)，`next_obs[3]` (vy)
- orientation：`next_obs[4]` (angle)，`next_obs[5]` (angular_velocity)
- contact：`next_obs[6]` (left_contact)，`next_obs[7]` (right_contact)
- action/engine：当前动作 `action`（0~3）可用于推断引擎使用
- derived_possible：可通过连续观测检测出界（|x| 过大 → failure precursor）、冲击（速度突变结合接触变化）、接近成功着陆（低速度 + 双接触 + 角度小 + 位置近零）等推断，用于构建条件奖励或惩罚

## 8. 不确定或不可用的信号
- 明确的终止原因字段（如 `'crash'`、`'landed'`）不可用
- 燃料消耗量或推力大小（动作仅表示引擎类型，未给出推力值）
- 身体其他部分碰撞信息（仅有两条支撑腿的接触标志）
- 目标是否已达到的布尔标志
- 视口边界的具体数值（需从采样或经验中推导）
- 任何步数或时间剩余信息（无 `truncation` 或 `steps_remaining`）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 飞行器/着陆器，具有左右两条支撑腿
  actuator_type: 三个独立离散引擎（主引擎、左方位引擎、右方位引擎）+ 无操作
  contact_structure: 两条支撑腿分别提供左、右接触布尔信号，其他身体部分接触可能导致 crash
primary_objectives:
  - 到达并停留在目标平台上（位置、速度、姿态三点同时满足）
secondary_objectives:
  - 最小化引擎使用（总动作中非零动作次数）
  - 尽快完成着陆（隐含时间惩罚）
main_failure_risks:
  - 水平飞出视口
  - 撞毁（非支撑部位触地）
  - 过度震荡或在目标上空不停盘旋，无法进入稳定状态
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 驱动飞行器向目标位置移动，基于到目标垫的欧氏距离或单向距离。
  why_required: 这是导航任务的核心，没有它无法学会向目标靠近。
  usable_signals: [next_obs[0], next_obs[1]]
  risks: 纯距离奖励可能导致飞行