# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
该环境是一个二维飞行器着陆任务。  
智能体控制飞行器从顶部中央区域出发，受初始随机力影响，需在最短时间内到达并稳定在中央目标垫上。  
过程中必须尽量降低推进剂消耗（减少引擎使用），同时学习精确导航、减速、保持姿态并安全触垫。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务明确要求到达中心目标垫并稳定，具有明确的导航目标位置（坐标原点），符合目标导向的导航类任务特征。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（接触标志以 1.0/0.0 形式出现，推断为浮点型）
- obs[0]: x_position —— 飞行器相对于目标垫中心的水平坐标
- obs[1]: y_position —— 飞行器相对于目标垫高度的垂直坐标
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体俯仰/倾斜角度
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左侧支撑腿/结构接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact —— 右侧支撑腿/结构接触标志（同上）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不开启任何引擎，惯性滑行
- action 1: left_orientation_engine —— 点燃左姿态引擎，产生旋转力矩（姿态调整）
- action 2: main_engine —— 点燃主引擎，提供垂直方向的推力（主要用于减速/上升）
- action 3: right_orientation_engine —— 点燃右姿态引擎，产生反向旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 飞行器停止运动且稳定（通常意味着已在目标垫上安全着陆）
- failure-like termination: 
  - crash_or_body_contact —— 飞行器主体碰撞地面或发生硬接触（非正常着陆）
  - horizontal_position_outside_viewport —— 水平位置超出可视边界
- ambiguous termination: 无
- truncation: 未在任务说明中明确给出，但环境中通常存在最大时间步上限（truncation），此处不依赖其信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  （info 字典为空，无 `success` 字段）
- explicit_failure_flag_available: false  
  （同理，无 `failure` 或 `termination_reason`）
- allowed_info_fields: {} （info 为空，不应使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 内容均不可用，禁止依赖任何隐含的终止原因。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（8 维数组）
- action（整数 0-3）
- next_obs（8 维数组）

禁止使用：
- original_reward（已被屏蔽，不可复原）
- official_reward 或任何形式的真实 reward
- info（空字典，无任何字段可用）
- training_progress（仅当环境 prompt 明确允许时才可使用，本环境未允许）

## 7. 可用于奖励函数的信号
以下信号可直接从 obs 或 next_obs 中提取，可用于设计奖励：
- position： obs[0]（x 位置）、obs[1]（y 位置），均以目标垫为参考系
- velocity： obs[2]（x 速度）、obs[3]（y 速度）
- orientation： obs[4]（机体角度）、obs[5]（角速度）
- contact： obs[6]（左接触）、obs[7]（右接触）
- action/engine： action 0-3，可用于判断是否使用引擎，施加燃料消耗惩罚

## 8. 不确定或不可用的信号
- original_reward：不可用
- info 中的任何信号（例如 success、failure、termination_reason）：info 为空，不可用
- 明确的“已到达目标”标志：环境没有提供直接的 reach flag，需要通过位置/速度/接触组合自行判断（例如 next_obs[1]≈0 且接触标志为真、速度很小可视为成功着陆）
- 外部施加的风或随机力：仅初始存在，后续 step 中未给出观测，不可直接使用（但可通过速度变化间接体现）
