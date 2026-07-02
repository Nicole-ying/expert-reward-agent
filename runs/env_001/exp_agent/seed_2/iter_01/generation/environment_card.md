# Env_001 环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从一个随机初始状态尽快飞到中央的目标着陆垫，并稳定停留在其上。  
目标包括三个子要求：  
- 尽可能短的时间到达目标上方并降落；  
- 着陆时速度与姿态尽量平稳（接触安全）；  
- 整个飞行过程中尽量少用引擎推力（节省能量）。  

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是使飞行器到达并停靠在空间中的一个目标位置（着陆垫），属于典型的导航目标到达任务；能量消耗是次要优化指标，不改变任务本质。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: 推断为 float64 或 float32（连续值，接触标志为 0.0/1.0）  
- 维度含义：  
  - obs[0] (x_position): 飞行器相对于目标着陆垫中心的水平坐标  
  - obs[1] (y_position): 飞行器相对于目标着陆垫高度的垂直坐标  
  - obs[2] (x_velocity): 飞行器水平线速度  
  - obs[3] (y_velocity): 飞行器垂直线速度  
  - obs[4] (body_angle): 飞行器机体角度（orientation）  
  - obs[5] (angular_velocity): 飞行器角速度  
  - obs[6] (left_support_contact): 左支撑腿接触标志（1.0 接触，0.0 未接触）  
  - obs[7] (right_support_contact): 右支撑腿接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：  
  - action 0: no_engine —— 不点火，依靠惯性滑行  
  - action 1: left_orientation_engine —— 点燃左侧姿态调整引擎（产生转动）  
  - action 2: main_engine —— 点燃主引擎（产生向上的推力）  
  - action 3: right_orientation_engine —— 点燃右侧姿态调整引擎（产生反向转动）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  - body_not_awake_or_settled（飞行器静止/稳定，可能表示成功着陆并稳定在垫子上）  
- failure-like termination:  
  - crash_or_body_contact（飞行器与地面或其他物体异常碰撞）  
  - horizontal_position_outside_viewport（水平方向飞出有效区域）  
- ambiguous termination:  
  - body_not_awake_or_settled 本身不携带位置信息，需结合 obs[0] 和 obs[1] 是否接近目标来判断真正的成功。  
- truncation:  
  - 始终为 False（无时间截断）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: [] （info 字典为空，无可信字段）  
- forbidden_or_uncertain_info_fields: 全部 info 字段均禁止使用（不存在任何可信标志）

## 6. reward 函数接口契约
函数签名必须为：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察向量）
- action（当前执行的动作）
- next_obs（下一时刻观察向量）
- info 中明确允许的字段（当前允许列表为空）
- training_progress（仅在 prompt 明确允许时使用）

禁止使用：
- original_reward（来自环境的原始奖励）
- official_reward（任何形式的环境内部奖励）
- 未声明的 info 字段（即所有 info 键）
- 未声明的 obs 切片（不得滥用未说明意义的维度）

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position)，可得到距离目标垫的距离  
- velocity: obs[2] (x_velocity), obs[3] (y_velocity)，可用于惩罚硬着陆或大速度  
- orientation: obs[4] (body_angle)，可鼓励保持在直立姿态  
- angular_velocity: obs[5] (angular_velocity)，可惩罚剧烈旋转  
- contact: obs[6] (left_contact), obs[7] (right_contact)，可检测着陆腿是否均已安全接触  
- action/engine: 动作本身就是燃料消耗信号，可惩罚非零推力的动作以鼓励节能

## 8. 不确定或不可用的信号
- original_reward / official_reward —— 强制屏蔽，不可用  
- info 内任何字段（如 success 标志） —— 不可用（info 为空）  
- 目标垫是否被准确接触（需要结合位置与接触信号自行判断）  
- 是否发生 crash 等事件的具体标志（只能通过观察变化和终止原因推断）  
- 外部风扰等环境未知因素 —— 不可观测