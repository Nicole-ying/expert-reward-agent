# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个2D类飞行器轨迹优化任务。主体初始位于视口顶部中央附近，并受到随机初始力的作用。智能体的目标是**尽快飞到并稳定停留在中央的目标着陆垫上**，同时尽量少用引擎推力。学习过程中需要实现：逐渐接近目标垫、降低速度、保持稳定的姿态（角度）并安全地完成接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box (连续值)
- shape: (8,)
- dtype: float32 (推断)
- obs[0]: x_position — 相对于目标着陆垫中心的水平坐标
- obs[1]: y_position — 相对于目标着陆垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 主体朝向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: right_support_contact — 右侧支撑接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无任何推力
- action 1: left_orientation_engine — 启动左侧姿态发动机（产生一个方向的角力/力矩）
- action 2: main_engine — 启动主发动机（产生向上的推力，推测）
- action 3: right_orientation_engine — 启动右侧姿态发动机（与左侧相反方向的角力/力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当主体在目标垫上稳定下来（body_not_awake_or_settled）时，可能被视为成功。该条件检查主体是否已经停止运动并处于睡眠状态，结合位置接近目标垫、速度极低等因素可以判断为成功着陆。
- failure-like termination: 
  - crash_or_body_contact（坠毁或身体触地等异常接触）
  - horizontal_position_outside_viewport（水平坐标超出视口边界）
- ambiguous termination: body_not_awake_or_settled 也可能发生在非目标位置（如坠毁后不再运动），此类终止不一定是成功。
- truncation: 本环境暂未看到基于步数的截断，但通常可能有默认的时间上限（如 1000 步），此类截断不属于成功/失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （info 字典为空，无可用的字段）
- forbidden_or_uncertain_info_fields: 任何试图从 info 中读取 success、failure、termination_reason 等字段的操作均不允许，因为这些字段不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs（当前观察）
- action（当前执行的动作）
- next_obs（下一时刻观察）
- info 中明确允许的字段：无（info 为空）
- training_progress：本环境任务描述中未明确允许利用该参数，应避免依赖

禁止使用：
- original_reward（已屏蔽的官方奖励）
- 任何未声明的 info 字段
- 任何未在观察空间中声明的 obs/next_obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0] (x), obs[1] (y) —— 可用于衡量距离目标垫越近越好
- velocity: obs[2] (vx), obs[3] (vy) —— 可用于鼓励终端低速，或在接近目标时减速
- orientation: obs[4] (angle) —— 可用于鼓励保持竖直或特定安全姿态
- angular velocity: obs[5] —— 可惩罚快速旋转
- contact: obs[6], obs[7] —— 两腿是否接触地面，可用于检测着陆及双腿是否平稳
- action/engine: 从动作编号可以判断是否使用了主引擎或姿态发动机，从而度量能量消耗

## 8. 不确定或不可用的信号
- 任何来自 info 的真实 success/failure 标志不可用（info 为空）
- 官方奖励值：已屏蔽，不可重构
- 不可观测的外部风或扰动信息
- 其他未在观察空间中列出的内部物理状态（如燃料量，假设不存在）
