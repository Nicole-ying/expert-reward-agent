# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿铰接体，使其在保持直立和身体状态健康的前提下，持续向前跳跃前进。主要目标是产生稳定的前向位移，同时避免摔倒或进入无效状态（高度过低/过高、躯干角度过大、状态值非有限）。策略应追求持续的前向移动，而不仅仅是不摔倒。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box (连续向量)
- shape: (11,)
- dtype: float32 或 float64（连续值）
- obs[0]: torso_height —— 躯干（主体）的垂直位置
- obs[1]: torso_angle —— 躯干的朝向角度
- obs[2]: upper_joint_angle —— 大腿关节角度
- obs[3]: lower_joint_angle —— 小腿关节角度
- obs[4]: foot_joint_angle —— 足部关节角度
- obs[5]: forward_velocity —— 躯干水平方向速度（前向速度）
- obs[6]: vertical_velocity —— 躯干垂直方向速度
- obs[7]: torso_angular_velocity —— 躯干角速度
- obs[8]: upper_joint_speed —— 大腿关节角速度
- obs[9]: lower_joint_speed —— 小腿关节角速度
- obs[10]: foot_joint_speed —— 足部关节角速度

## 4. 动作空间 action_space
- type: Box (连续向量)
- shape: (3,)
- 范围: 每个动作维度 [-1.0, 1.0]
- action 0: upper_joint_torque —— 施加在大腿关节上的扭矩
- action 1: lower_joint_torque —— 施加在小腿关节上的扭矩
- action 2: foot_joint_torque —— 施加在足部关节上的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功信号；任务没有“目标到达”的终点，前向移动是可无限持续的，因此不存在成功终止。
- failure-like termination:
  - body_height_outside_healthy_range （躯干高度超出健康范围）
  - torso_angle_outside_healthy_range （躯干角度超出健康范围）
  - state_value_outside_finite_healthy_range （状态值中出现非有限值，如NaN或无穷大）
  以上三种条件触发 `terminated=True`，均视为失败。
- ambiguous termination: 无。
- truncation: 达到 `time_limit_reached` 时触发 `truncated=True`，不算失败，仅截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false（只能通过 `terminated` 标志间接推断，但 `info` 为空字典，不包含失败原因字段）
- allowed_info_fields: [] （无可用 info 字段）
- forbidden_or_uncertain_info_fields:
  - reward_forward
  - reward_ctrl
  - reward_survive
  - x_position
  - y_position
  - z_distance_from_origin
  等官方奖励项和世界位置信息均被禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观察向量（形状 (11,)）
- action: 当前执行的动作向量（形状 (3,)）
- next_obs: 下一时刻观察向量（形状 (11,)）
- info: 允许使用的字段为空（即 `info` 不可用）
- training_progress: 只有在 prompt 明确说明时才使用（本次说明未提及，故不应依赖）

禁止使用：
- original_reward（官方奖励已被屏蔽）
- official_reward
- 任何未声明的 info 字段（包括那些被禁止的字段）
- 未在观察空间中明确定义的 obs 切片（例如不能直接访问世界坐标 x 位置，因为不在观察中）

## 7. 可用于奖励函数的信号
- position:
  - torso_height (obs[0])：用于判断重心是否在健康范围，可构造生存/高度奖励
  - torso_angle (obs[1])：用于鼓励直立
  - 关节角度 obs[2], obs[3], obs[4]：可构造姿态或效率项
- velocity:
  - forward_velocity (obs[5])：核心信号，直接衡量前向移动速度，可用于正奖励
  - vertical_velocity (obs[6])：可避免过大的上下跳动
  - 关节速度 obs[8], obs[9], obs[10]：可构造平滑性或能耗惩罚
- orientation:
  - torso_angle, torso_angular_velocity (obs[1], obs[7])：保持直立
- contact: 无直接接触信号，但可通过高度、速度间接推断。
- action/engine:
  - 动作向量 (action[0], action[1], action[2])：可惩罚大扭矩动作（例如平方和）

## 8. 不确定或不可用的信号
- 世界坐标 x（前向位移）：被禁止，不可用。
- 官方奖励分量（reward_forward, reward_ctrl, reward_survive）：被屏蔽，不可用。
- 接触力、接地状态：未提供。
- info 字典中任何字段：均为空，不可用。
- 历史累积信息（因 reward 函数无状态，不能跨步累积速度或距离）。
