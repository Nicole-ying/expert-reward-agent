# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主体是一个 2D 飞行器/着陆器，从顶部中央附近随机施加初始力开始，必须尽可能快地运动到屏幕中央的目标垫上，并稳定、安全地停泊（软着陆）；同时要尽量减少发动机使用。智能体的核心挑战在于：接近目标、减速、保持姿态水平、双腿同时轻柔接触垫面，并避免坠毁、出界或长时间悬停。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 环境提供明确的目标位置（目标垫相对坐标），需要控制 agent 到达并停驻在该目标上；附属的节能与快速要求只是效率优化，不是多目标冲突，因此归入导航目标到达类型。
dynamics_subtype: goal_approach_and_soft_contact

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position – 相对目标垫的水平坐标，可奖励用
- obs[1]: y_position – 相对目标垫高度的垂直坐标，可奖励用
- obs[2]: x_velocity – 水平线速度，可奖励用
- obs[3]: y_velocity – 垂直线速度，可奖励用
- obs[4]: body_angle – 机体朝向角，可奖励用
- obs[5]: angular_velocity – 角速度，可奖励用
- obs[6]: left_support_contact – 左支撑腿是否接触目标垫（1/0），可奖励用
- obs[7]: right_support_contact – 右支撑腿是否接触目标垫（1/0），可奖励用

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine – 无推力
- action 1: left_orientation_engine – 点燃左方向发动机（产生姿态/侧向控制）
- action 2: main_engine – 点燃主发动机（提供向上推力）
- action 3: right_orientation_engine – 点燃右方向发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: 机体已稳定/不活跃（body_not_awake_or_settled），可能对应成功软着陆（双腿接触垫面、速度与角度极小）
- failure‑like termination: 坠毁/机体与地面或障碍物接触（crash_or_body_contact）；水平位置超出视口（horizontal_position_outside_viewport）
- ambiguous termination: 所有终止条件在源码中未区分成功/失败，需从观测信号间接推断
- truncation: 无显式说明

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}）
- forbidden_or_uncertain_info_fields: 所有未在观察空间中
