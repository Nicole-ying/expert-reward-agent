# 匿名环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。  
主体从视口顶部中央附近出发，并受到随机初始力。  
目标是**尽可能快地飞到并稳定降落在中央目标垫上**，同时**消耗尽可能少的引擎推力**。  
智能体需要学会：接近目标、减速、保持姿态稳定、实现安全触地。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box（连续）
- shape: [8]
- dtype: float64 （默认，部分字段为0/1浮点）
- obs[0]: x_position —— 相对目标垫的水平坐标
- obs[1]: y_position —— 相对目标垫高度的垂直坐标
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑腿接触标志（1.0 接触; 0.0 未接触）
- obs[7]: right_support_contact —— 右支撑腿接触标志（1.0 接触; 0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点燃任何引擎
- action 1: left_orientation_engine —— 点燃左侧姿态引擎
- action 2: main_engine —— 点燃主引擎（产生推力）
- action 3: right_orientation_engine —— 点燃右侧姿态引擎（与左相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 机体不再运动且稳定，可能对应成功着陆
- failure-like termination: crash_or_body_contact —— 机体重重撞击或非支撑部位触地（并非正常支撑腿触地）
- failure-like termination: horizontal_position_outside_viewport —— 飞出水平边界
- ambiguous termination: 无
- truncation: 源码中未展示步数限制，但通常环境存在episode截断，这里无信息，不依赖

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典）
- explicit_failure_flag_available: false
- allowed_info_fields: （无）
- forbidden_or_uncertain_info_fields: info.* （任何 info 字段都未声明，不可安全使用）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs （当前观测）
- action （执行的动作）
- next_obs （执行动作后的观测）
- info 中明确允许的字段：无
- training_progress：任务描述未提及可用，默认禁止

禁止使用：
- original_reward / official_reward
- info 中的任何字段
- 未声明的 obs 切片（但 obs 维度已完整解释，所有维度都可用）

## 7. 可用于奖励函数的信号
- **位置**：obs[0] x_position, obs[1] y_position（相对目标垫的水平和垂直距离）
- **速度**：obs[2] x_velocity, obs[3] y_velocity（下降速度等）
- **姿态与角速度**：obs[4] body_angle, obs[5] angular_velocity
- **接触**：obs[6] left_support_contact, obs[7] right_support_contact（是否两条支撑腿着地）
- **动作/引擎**：当前动作类型（是否使用主引擎或姿态引擎）

## 8. 不确定或不可用的信号
- info 中任何字段（ex: success, failure, landing_flag）—— 源码显示 info 为空，不存在
- original_reward —— 被屏蔽，不可用
- 真实任务名（LunarLander 等）—— 已被匿名化，不可用于奖励设计
- 环境内部物理量（如推力大小、风力）—— step 源码未给出，不确定不能依赖