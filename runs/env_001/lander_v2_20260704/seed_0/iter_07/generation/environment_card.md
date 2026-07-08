# Env_001 环境理解卡片

## 1. 任务目标
控制一个类似飞行器的刚体，从视口顶部中央附近以随机初始力开始运动，尽快到达并稳定降落在场地中央的目标着陆垫上，同时尽可能减少引擎推力消耗。智能体需要学会平滑接近目标、减小速度、保持姿态稳定，并实现两条着陆腿与垫面的安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（根据一般标准推断，源未显式声明）
- obs[0]: x_position — 相对于目标垫平面的水平坐标
- obs[1]: y_position — 相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 竖直线速度
- obs[4]: body_angle — 刚体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（0.0 或 1.0）
- obs[7]: right_support_contact — 右支撑腿接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无引擎推力（do nothing）
- action 1: 点燃左姿态调整引擎（产生一个方向的力矩）
- action 2: 点燃主引擎（产生推力）
- action 3: 点燃右姿态调整引擎（产生相反方向的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  身体稳定并静止（body_not_awake_or_settled），两条支撑腿均接触着陆垫（left_support_contact == 1.0 且 right_support_contact == 1.0），且水平与垂直位置接近目标。
- failure-like termination:
  1) 身体与地面发生不当碰撞（crash_or_body_contact），例如船体非支撑部分触地；
  2) 水平位置超出视口范围（horizontal_position_outside_viewport）；
  3) 身体在未成功着陆的情况下失去活动或稳定（body_not_awake_or_settled，但没有双支撑接触或位置远离目标）。
- ambiguous termination:
  身体进入稳定静止但只有一个支撑腿接触或完全无接触，位置接近目标——可能属于不完美着陆或“搁浅”，需根据实际任务要求判断成功与否。
- truncation:
  从提供的源码看，没有环境层级的截断机制，episode 会一直运行直到触发以上终止条件。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 标志）
- explicit_failure_flag_available: false （无 failure 标志）
- allowed_info_fields: 无（info 固定为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

说明：判断成功或失败只能根据 next_obs 中的位置、速度、支撑接触标志以及 terminated 信号间接推断，不能依赖任何 info 字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（8 维数组，含义见观察空间定义）
- action（执行的离散动作，0~3）
- next_obs（8 维数组，transition 后的状态）
- info 中明确允许的字段（无）

禁止使用：
- original_reward （即被屏蔽的官方奖励）
- official_reward 或任何预置奖励
- 未声明的 info 字段
- 未声明的 obs / next_obs 切片含义（但 obs/next_obs 全部 8 维均已知）
- training_progress（本环境未明确允许使用，故不得使用）

## 7. 可用于奖励函数的信号
- position: next_obs[0]（水平相对位置）, next_obs[1]（垂直相对位置）
- velocity: next_obs[2]（水平速度）, next_obs[3]（垂直速度）
- orientation: next_obs[4]（角度）, next_obs[5]（角速度）
- contact: next_obs[6]（左支撑腿接触）, next_obs[7]（右支撑腿接触）
- action/engine: 动作选择（action，0~3），可用来惩罚引擎使用

## 8. 不确定或不可用的信号
- 官方奖励值（original_reward）——已屏蔽，不可用且不应重建
- 任何 info 字段（如 success, failure, termination_reason 等）——不存在
- 训练的 progress 参数——未授权使用
- 环境内部物理信息（如引擎推力大小、风力等）——未暴露在观察中，不可用