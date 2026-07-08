# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中心附近被随机初始力弹出，需要尽量快地到达并稳定停靠在中央目标垫上，同时尽可能节省引擎推力。智能体应学会：接近目标、减小速度、保持姿态稳定、安全接触目标垫。

## 2. 任务类型选择
- selected_route_id: navigation_goal_reaching
- confidence: high
- reason: 虽然任务要求兼顾快速与节能，但根本目标是到达并稳定停靠在目标垫（goal reaching）。路由规则明确指出“到达某个目标状态或目标区域，通常属于 navigation_goal_reaching，即使还要求安全、稳定或高效”。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（默认）
- 各维含义：
  - obs[0] (x_position): 相对于目标垫的水平坐标
  - obs[1] (y_position): 相对于垫子高度的垂直坐标
  - obs[2] (x_velocity): 水平线速度
  - obs[3] (y_velocity): 垂直线速度
  - obs[4] (body_angle): 机体方向角
  - obs[5] (angular_velocity): 角速度
  - obs[6] (left_support_contact): 左支撑腿接触标志（0.0 或 1.0）
  - obs[7] (right_support_contact): 右支撑腿接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- 动作含义：
  - action 0: no_engine — 不点火，无推力
  - action 1: left_orientation_engine — 点燃左侧姿态引擎
  - action 2: main_engine — 点燃主引擎（下方推力）
  - action 3: right_orientation_engine — 点燃右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（当飞行器速度极低且不再移动时终止，若此时已接触目标垫则可视为成功着陆）
- failure-like termination: crash_or_body_contact（机体坠毁或非正常接触地面），horizontal_position_outside_viewport（飞出水平边界）
- ambiguous termination: body_not_awake_or_settled 本身可能发生在未接触目标垫的任何位置，需结合观测判断
- truncation: 无（返回的 truncated 永远为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 字典为空，无任何字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，不能假设包含 “success”、“failure” 或 “termination_reason”

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察）
- action（当前执行的动作）
- next_obs（下一时刻观察）
- info 中明确允许的字段（本环境 info 为空，故实际不可用）

禁止使用：
- original_reward / official_reward
- training_progress（当前 prompt 未明确允许使用时必须禁止）
- 任何未声明的 info 字段
- 任何未在任务说明中出现的 obs 切片或隐藏内部状态

## 7. 可用于奖励函数的信号
- x_position
  - source: obs[0] 或 next_obs[0]
  - meaning: 水平坐标相对于目标垫
  - availability: available
- y_position
  - source: obs[1] 或 next_obs[1]
  - meaning: 垂直坐标相对于垫子高度
  - availability: available
- x_velocity
  - source: obs[2] 或 next_obs[2]
  - meaning: 水平线速度
  - availability: available
- y_velocity
  - source: obs[3] 或 next_obs[3]
  - meaning: 垂直线速度
  - availability: available
- body_angle
  - source: obs[4] 或 next_obs[4]
  - meaning: 机体方向角
  - availability: available
- angular_velocity
  - source: obs[5] 或 next_obs[5]
  - meaning: 角速度
  - availability: available
- left_support_contact
  - source: obs[6] 或 next_obs[6]
  - meaning: 左支撑腿是否接触表面（0/1）
  - availability: available
- right_support_contact
  - source: obs[7] 或 next_obs[7]
  - meaning: 右支撑腿是否接触表面（0/1）
  - availability: available
- action
  - source: action
  - meaning: 当前执行的引擎选择（可用于惩罚推力使用）
  - availability: available

## 8. 不确定或不可用的信号
- original_reward：被屏蔽，禁止使用
- info 内任何字段：info 字典为空，无可用信号
- 是否成功着陆的显式标志：不存在，必须从位置、速度、接触等多维信号自行推断
- 是否坠毁/越界的显式标志：不存在，终止条件混杂，需结合 next_obs 判断