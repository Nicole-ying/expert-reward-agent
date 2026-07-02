# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个二维着陆器类型的轨迹优化任务。智能体起始于视口顶部中央附近，并带有随机初始作用力。目标是**以尽可能小的引擎推力**，**尽快**到达并稳定悬停或停靠在视口中央的目标平台上。智能体需要学习接近目标、降低速度、保持稳定姿态，并与平台安全接触，同时避免碰撞、翻出视口或身体过度不稳定。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求“到达并停靠在中央目标平台”，本质是一个带有物理约束的导航型目标到达问题。虽然具有离散动作和连续状态，但核心是在有重力、接触和姿态要求的条件下到达指定位置并稳定，符合 goal‑reaching 任务特征。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 通常为 float32
- 各维度含义（按索引）：
  - obs[0]: x_position – 相对于目标平台的水平坐标
  - obs[1]: y_position – 相对于平台高度的垂直坐标
  - obs[2]: x_velocity – 水平线速度
  - obs[3]: y_velocity – 垂直线速度
  - obs[4]: body_angle – 机体角度
  - obs[5]: angular_velocity – 角速度
  - obs[6]: left_support_contact – 左侧支撑腿接触标志（1.0 表示接触，0.0 表示未接触）
  - obs[7]: right_support_contact – 右侧支撑腿接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- 动作总数: 4
- 每个 action 的含义：
  - action 0: no_engine – 不点火（什么都不做）
  - action 1: left_orientation_engine – 点燃左侧姿态发动机
  - action 2: main_engine – 点燃主发动机
  - action 3: right_orientation_engine – 点燃右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当智能体安全接触并稳定于目标平台时（例如左右腿均接触且速度很小），会触发 `body_not_awake_or_settled` 终止。**但这同时也可能是失败终止（如身体失去稳定而“睡眠”），因此不能独立作为成功标志。**
- failure-like termination: `crash_or_body_contact`（主体部分意外碰撞）和 `horizontal_position_outside_viewport`（飞离水平范围）都是明显的失败终止。
- ambiguous termination: `body_not_awake_or_settled` 本身含义模糊（既可能是成功着陆后的稳定，也可能是意外坠落后的静止），不能直接用于奖励或成败判断。
- truncation: step 源码中未设置最大步数，返回的 truncated 始终为 False。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 返回空字典，不能使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段，包括假设的 “success”、“failure”、“termination_reason” 等均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs – 当前观察（形状 (8,)）
- action – 当前动作（整数 0~3）
- next_obs – 下一时刻观察（形状 (8,)）
- info – 仅限已明确声明的字段（目前为空，因此不得读取任何 info 内容）
- training_progress – 仅在 prompt 明确允许时才能使用，本环境任务描述未提及，故**不可用**

禁止使用：
- original_reward（官方奖励被屏蔽，严禁参考或重建）
- official_reward 的任何变体
- 未在以上声明中列出的 info 字段
- 未在“允许使用”中出现的 obs 切片或衍生量

## 7. 可用于奖励函数的信号
- position: obs[0] (水平相对位置)、obs[1] (垂直相对位置)
- velocity: obs[2] (水平速度)、obs[3] (垂直速度)
- orientation: obs[4] (机体角度)
- angular velocity: obs[5] (角速度)
- contact: obs[6] (左腿接触)、obs[7] (右腿接触)
- action/engine: action（0~3）可用于衡量推力使用或引擎点燃情况

以上信号均可自由组合，用以设计描述接近目标、速度减小、姿态平稳、安全接触和节能等方面的奖励。

## 8. 不确定或不可用的信号
- 明确的“成功”或“失败”标志：info 为空，终止条件都是原语 (crash, out of bounds, not awake)，无显式的 episode 成功/失败标签。
- original_reward：被屏蔽，不可使用，也不可推断其形式。
- 任何基于动作“推力大小”的连续量：动作是离散的，只能判断引擎是否点燃，无法获取推力数值。
- 平台精确坐标或环境物理参数：观察只提供相对于目标 pad 的差分值，无法获得绝对位置。