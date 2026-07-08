# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 着陆器任务。一个类似飞行器的主体从视口顶部中央出发，受到一个随机初始力。
核心目标：尽可能**快速**且**稳定**地降落在中央目标平台上，同时**尽量减少发动机推力**（燃料消耗）。  
智能体需要学会接近目标、减速、保持姿态平衡，并安全接触平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (默认，接触标志以 1.0/0.0 表示)
- obs[0]: x_position — 飞行器相对目标平台的水平坐标（中心为 0）
- obs[1]: y_position — 飞行器相对平台高度的垂直坐标（平台高度处为 0）
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体倾斜角度
- obs[5]: angular_velocity — 机体角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（接触平台为 1.0，否则 0.0）
- obs[7]: right_support_contact — 右支撑腿接触标志（接触平台为 1.0，否则 0.0）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无推力
- action 1: left_orientation_engine — 点燃左侧姿态发动机（产生使机体向左旋转的力矩）
- action 2: main_engine — 点燃主发动机（产生向上的推力）
- action 3: right_orientation_engine — 点燃右侧姿态发动机（产生使机体向右旋转的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 当机体不再活跃或达到稳定状态时终止。如果此时机体已位于目标平台上、姿态平稳、双腿接触，可视为成功着陆。
- failure-like termination: crash_or_body_contact — 机体主体碰撞地面或接触平台之外的表面（非支撑腿的正常接触）。通常意味着坠毁或严重撞击。
- failure-like termination: horizontal_position_outside_viewport — 机体水平飞出视野，无法返回。
- ambiguous termination: 无明确标记的中间状态。终止时只有观测数据，无额外成功/失败标签。
- truncation: 本环境没有明确的时间截断（step 返回的第四个值是 False，即 truncated 恒为 False），但多数实际包装中会附加时间上限。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，info 为空字典，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；不能假设 info["success"] 或 info["failure"] 存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs – 当前观测（8维数组）
- action – 当前动作（0~3 整数）
- next_obs – 下一时刻观测（8维数组）
- info – 空字典，不允许从中读取任何字段
- training_progress – 训练进度（0.0~1.0），仅在 prompt 明确允许时使用

禁止使用：
- original_reward – 原始奖励被屏蔽，严禁直接使用
- official_reward – 同 original_reward
- info 中的任何未声明字段（info 目前为空）
- obs 中未在“可用于奖励函数的信号”里声明的切片

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1]) — 用于计算到目标平台的距离、靠近程度
- velocity: x_velocity (obs[2]), y_velocity (obs[3]) — 用于衡量运动速度、鼓励减速
- orientation: body_angle (obs[4]), angular_velocity (obs[5]) — 用于评估姿态稳定性和旋转惩罚
- contact: left_support_contact (obs[6]), right_support_contact (obs[7]) — 双腿是否接触平台，用于判断着陆状态
- action/engine: action 本身 — 可基于用了哪些引擎（特别是主发动机）给予燃料消耗惩罚

## 8. 不确定或不可用的信号
- 无显式的成功/失败标志（info 为空，无法直接获取 success、failure、termination_reason）
- 目标的绝对距离需要从 x_position、y_position 自行计算（无直接距离传感器）
- 燃料剩余、时间步、风力等环境内部状态不可见
- 原始奖励 original_reward 被屏蔽，不可用