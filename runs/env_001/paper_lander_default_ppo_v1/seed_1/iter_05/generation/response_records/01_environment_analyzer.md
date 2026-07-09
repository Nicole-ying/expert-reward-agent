# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器，从初始随机位置/速度出发，尽快到达并稳定地降落在画面中央的目标垫上，同时尽量减少引擎推力的使用。在接近目标时需要减速、保持稳定的姿态并安全地触垫。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测，包含浮点数与 0.0/1.0 标志）
- obs[0]: horizontal position relative to target pad (x_position_relative_to_target)
- obs[1]: vertical position relative to pad height (y_position_relative_to_pad_height)
- obs[2]: horizontal linear velocity (x_velocity)
- obs[3]: vertical linear velocity (y_velocity)
- obs[4]: body orientation angle (body_angle)
- obs[5]: angular velocity (angular_velocity)
- obs[6]: left support contact flag (1.0 = contact, 0.0 = no contact)
- obs[7]: right support contact flag (1.0 = contact, 0.0 = no contact)

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no engine – 不启动任何引擎，滑行/惯性飞行
- action 1: left orientation engine – 启动左侧姿态调整引擎（产生转向力矩）
- action 2: main engine – 启动主引擎（产生与飞行器朝向相关的推力）
- action 3: right orientation engine – 启动右侧姿态调整引擎（产生相反方向的转向力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（机体静止/稳定）可能是成功降落在目标垫上的信号，但前提是位置已接近目标，否则也可能是其他原因导致的静止（如飘走后静止在远处）。该条件本身不直接等于成功。
- failure-like termination: 
  - `crash_or_body_contact`（坠毁或危险的身体接触）明显是失败状态。
  - `horizontal_position_outside_viewport`（水平位置超出画面边界）属于失败/失控。
- ambiguous termination: 
  - `body_not_awake_or_settled` 在没有位置信息辅助时无法明确区分成功或失败。
- truncation: 暂无明确的最大步数限制，`step` 源码中仅返回 `terminated`，没有 `truncated` 返回（默认 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典）
- explicit_failure_flag_available: false （info 为空字典）
- allowed_info_fields: 无（info = {}，不得使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 执行动作前的观察
- action: 执行的动作
- next_obs: 执行动作后的观察

禁止使用：
- original_reward（已被掩码，不得使用）
- info 中的任何字段（info 为空）
- training_progress（除非任务描述显式要求，此处不要求）

## 7. 可用于奖励函数的信号
- position: next_obs[0]（距目标垫的水平距离）、next_obs[1]（距目标垫的垂直高度）
- velocity: next_obs[2]（水平速度）、next_obs[3]（垂直速度）
- orientation: next_obs[4]（机体角度）、next_obs[5]（角速度）
- contact: next_obs[6]（左支撑触垫）、next_obs[7]（右支撑触垫）
- action/engine: action 的选择（0~3），可用来惩罚引擎使用

## 8. 不确定或不可用的信号
- 终止时的成功标志：不存在（info 未提供 success / failure 信号）
- 原始奖励：已被掩码，不可使用
- 终止原因字符串：不可用
- 任何环境内部的时间步数、燃料消耗等未暴露的信息：不可用
