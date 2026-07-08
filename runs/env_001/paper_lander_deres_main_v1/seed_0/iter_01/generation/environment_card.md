# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近以随机初始速度出发，目标是尽可能快地到达并平稳降落在视口中央的指定着陆平台上，同时尽量少用引擎推力。智能体需要学会靠近目标点、降低速度、保持稳定姿态并实现安全接触（着陆腿与平台接触且最终静止）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（连续值）
- obs[0]: x_position — 机体相对于目标平台中心的水平坐标
- obs[1]: y_position — 机体相对于平台高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿是否接触（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右支撑腿是否接触（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无引擎（不做任何推进）
- action 1: 左姿态引擎（触发左侧姿态调整引擎）
- action 2: 主引擎（触发主推进引擎）
- action 3: 右姿态引擎（触发右侧姿态调整引擎）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - `body_not_awake_or_settled`：表示机体已经停止运动（低速、可能双脚着地且稳定），这通常是成功着陆的自然结果。

- failure-like termination:
  - `crash_or_body_contact`：机体与地面或障碍发生非允许的碰撞（如舱体直接撞击地面、侧翻等）。
  - `horizontal_position_outside_viewport`：机体水平位置超出边界，失去控制。

- ambiguous termination:
  - 无。

- truncation:
  - 环境未定义 episode 长度上限（通过其他方式截断），但通常会有一个最大步数限制作为安全截断，此处未明确给出。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无可用字段，step 返回 info = {}）
- forbidden_or_uncertain_info_fields: 任何可能存在的 info 字段均不可用（如 success, failure, termination_reason 等）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前步的 8 维观测
- action：当前动作（0~3）
- next_obs：下一时刻的 8 维观测
- info：环境返回的 info 字典（本环境恒为空 `{}`，因此仅能使用空字典）
- training_progress：本次不推荐使用（未在 prompt 中明确允许）

禁止使用：
- original_reward（官方原始奖励）
- official_reward 或任何已屏蔽的回报
- 未在观测空间中声明的 obs 切片
- 未在 info 中出现的任何字段

## 7. 可用于奖励函数的信号
基于观测空间，可直接使用的信号包括：
- position: obs[0] 横向偏差、obs[1] 垂直偏差
- velocity: obs[2] 水平速度、obs[3] 垂直速度
- orientation: obs[4] 机体倾角、obs[5] 角速度
- contact: obs[6] 左腿接触标志、obs[7] 右腿接触标志
- action/engine: 动作 id（0-3）可用于惩罚引擎使用或鼓励特定策略
- 组合衍生信号：如是否双脚同时接地、速度是否接近零、位置是否接近 0（目标点）

## 8. 不确定或不可用的信号
- 官方显式的 success / failure 布尔标志（info 中不存在）
- 目标平台坐标或是否在平台上方的绝对判断（需从位置和接触标志间接推断）
- original_reward 或 offline 奖励值（已屏蔽，严禁使用）
- 任何未在上述观测中出现的力学量（如推力大小、燃料消耗量等）