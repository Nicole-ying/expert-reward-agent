# Env_001 环境理解卡片

## 1. 任务目标
控制一个二维飞行器，从初始位置出发，尽快到达并稳定降落在视口中央的目标平台上，同时尽量减少引擎推力消耗。飞行器初始带有随机微小作用力，需要学会接近目标、减速、保持稳定姿态、并安全着陆（实现两侧支撑脚同时接触，且速度接近于零）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据典型环境设定，第6、7维为0/1二值，但整体浮点）
- obs[0]: x_position — 机体相对于目标平台中心的水平坐标（m）
- obs[1]: y_position — 机体相对于目标平台高度的垂直坐标（m）
- obs[2]: x_velocity — 机体水平线速度（m/s）
- obs[3]: y_velocity — 机体垂直线速度（m/s）
- obs[4]: body_angle — 机体倾斜角度（rad）
- obs[5]: angular_velocity — 机体角速度（rad/s）
- obs[6]: left_support_contact — 左侧支撑脚接触标志（1.0 接触, 0.0 未接触）
- obs[7]: right_support_contact — 右侧支撑脚接触标志（1.0 接触, 0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不点火，仅靠惯性飞行
- action 1: left_orientation_engine — 点燃左侧姿态调整引擎（产生使机体逆时针旋转的力矩）
- action 2: main_engine — 点燃主引擎（产生向上的推力）
- action 3: right_orientation_engine — 点燃右侧姿态调整引擎（产生使机体顺时针旋转的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 如果同时满足位置接近目标、速度极小、姿态平稳且两侧支撑脚接触，则很可能代表成功着陆
- failure-like termination: crash_or_body_contact — 机体与地面或其他非目标部分发生碰撞；horizontal_position_outside_viewport — 水平位置超出视野边界
- ambiguous termination: body_not_awake_or_settled — 也可能只是漂移到不可控状态进入沉睡，未必成功，需结合 position/contact/velocity 判断
- truncation: 信息中未提供 episode 长度限制，但一般环境会有默认 step 上限

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 该环境 step 返回 info 为空字典 {}，因此只能使用 obs、next_obs、action，不得使用 info
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；original_reward 不可用；任何未在 obs 或 next_obs 中明确定义的信号均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (numpy array, shape (8,))
- action (int, 取值 0/1/2/3)
- next_obs (numpy array, shape (8,))
- info 中的字段：无（info为空字典）
- training_progress：该 prompt 未明确允许，不应使用

禁止使用：
- original_reward（官方奖励被 mask）
- 任何 info 字段（不存在）
- 任何未在上述 obs 中声明的信号（如 true_success 等隐藏变量）

## 7. 可用于奖励函数的信号
- position: next_obs[0]（x 误差）、next_obs[1]（y 误差），可构造距离奖励/惩罚
- velocity: next_obs[2], next_obs[3] 可惩罚着陆时速度过快
- orientation: next_obs[4] 可惩罚非水平姿态
- angular_velocity: next_obs[5] 可惩罚旋转
- contact: next_obs[6], next_obs[7] 可鼓励双足同时接触
- action/engine: action 值可用于惩罚引擎使用（尤其 1,2,3 相对于 0 有额外消耗）
- 复合信号：可组合位置、速度、姿态、接触构建成功着陆判断（如 |x|<ε_x, y≈0, |v|<ε_v, |angle|<ε_angle, left_contact & right_contact 均为 1）

## 8. 不确定或不可用的信号
- 显式 success / failure 标记（info 中未提供）
- 目标绝对坐标（仅提供相对坐标，但已足够）
- 风力数值、引擎推力大小（被遮蔽）
- 任何由官方奖励衍生出的中间信号（不可用）