# 匿名环境理解卡片

## 1. 任务目标
控制一个2D飞行器从顶部出发，尽快且尽可能少地用引擎推力降落到中央目标垫上，并稳定停靠。主体要求是到达并安全着陆，附属要求是推进效率和姿态平稳。不要把纯粹的时间最短或燃料最少当成独立主目标，它们只是附属优化。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达指定目标垫并安全停靠，附属要求（快、省推力）不构成与主目标权重相同的多目标冲突，属于典型的导航到达问题。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float（推测）
- obs[0]: x_position, 相对目标垫的水平坐标，reward_usable: true
- obs[1]: y_position, 相对目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity, 水平线速度，reward_usable: true
- obs[3]: y_velocity, 垂直线速度，reward_usable: true
- obs[4]: body_angle, 机体倾斜角，reward_usable: true
- obs[5]: angular_velocity, 角速度，reward_usable: true
- obs[6]: left_support_contact, 左支撑腿接触标志（0或1），reward_usable: true
- obs[7]: right_support_contact, 右支撑腿接触标志（0或1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无推力
- action 1: left_orientation_engine — 点燃左姿态引擎，产生顺时针转动效果（具体方向取决于坐标系）
- action 2: main_engine — 点燃主引擎，通常产生向上推力以减速或提供升力
- action 3: right_orientation_engine — 点燃右姿态引擎，产生逆时针转动效果

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 极可能表示机体已稳定停靠并进入休眠，结合观测中位置接近原点、速度极小、至少一个支撑接触时，可判定为成功着陆。
- failure-like termination: crash_or_body_contact（可能与障碍物或地面猛烈碰撞）、horizontal_position_outside_viewport（水平出界）
- ambiguous termination: 如果 body_not_awake_or_settled 发生时位置偏离目标垫或姿态异常，则为失败（如侧翻冻住）。需通过观测信号区分。
- truncation: 无明确最大步数截断说明，但可能存在时间上限；该截断不属于任务成功或失败。

### 5.2 success/failure