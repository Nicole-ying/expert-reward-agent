# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿关节体，使其向前连续跳跃前进。  
要求尽可能保持身体直立，同时高效使用关节扭矩。  
一旦身体高度、躯干倾角或任何状态值超出健康范围，回合立即终止。  
主要目标是持续向前移动，而非仅仅保持平衡不倒。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (11,)
- dtype: float（推断）
- obs[0]: torso_height —— 躯干（主体）的垂直高度
- obs[1]: torso_angle —— 躯干的倾斜角
- obs[2]: upper_joint_angle —— 大腿关节角度
- obs[3]: lower_joint_angle —— 小腿关节角度
- obs[4]: foot_joint_angle —— 足部关节角度
- obs[5]: forward_velocity —— 躯干水平速度（前进方向）
- obs[6]: vertical_velocity —— 躯干垂直速度
- obs[7]: torso_angular_velocity —— 躯干角速度
- obs[8]: upper_joint_speed —— 大腿关节角速度
- obs[9]: lower_joint_speed —— 小腿关节角速度
- obs[10]: foot_joint_speed —— 足部关节角速度

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (3,)
- 范围: 每个关节扭矩 ∈ [-1.0, 1.0]
- action 0: upper_joint_torque —— 施加在大腿关节的扭矩
- action 1: lower_joint_torque —— 施加在小腿关节的扭矩
- action 2: foot_joint_torque —— 施加在足部关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止；任务没有设定“到达终点”之类的条件，因此不存在成功终止信号。
- failure-like termination:  
  - 身体高度超出健康范围（跌倒或飞起）  
  - 躯干倾角超出健康范围（过度倾斜）  
  - 任意状态值变为非有限值（NaN / inf）
- ambiguous termination: 无
- truncation: 时间步到达上限（truncated）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空）
- forbidden_or_uncertain_info_fields:  
  - reward_forward  
  - reward_ctrl  
  - reward_survive  
  - x_position  
  - y_position  
  - z_distance_from_origin  
  （任何未声明的 info 字段均禁止使用）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，11维）
- action（3维连续动作）
- next_obs（下一时刻观测，11维）
- info 中仅可包含空字典（无可用字段）

禁止使用：
- original_reward（官方奖励被屏蔽）
- official_reward 或任何 masked reward
- info 中任何字段（即使环境内部有，对外也视作不存在）
- training_progress（本 prompt 未允许使用）

## 7. 可用于奖励函数的信号
- position:  
  - torso_height (obs[0])  
  - torso_angle (obs[1])  
  - 各关节角度 (obs[2]~obs[4])
- velocity:  
  - forward_velocity (obs[5]) —— 可直接用于鼓励前进  
  - vertical_velocity (obs[6])  
  - torso_angular_velocity (obs[7])  
  - 各关节角速度 (obs[8]~obs[10])
- orientation: torso_angle, 各关节角度
- contact: 无直接观测
- action/engine: 动作扭矩本身 (action[0]~action[2])，可度量控制代价

## 8. 不确定或不可用的信号
- 世界坐标位置（x, y, z）被排除在观测之外，不可用
- 接触或足底受力信息未提供
- 任何原本可能出现在 info 中的奖励分量（如前进奖励、控制惩罚、生存奖励）均不可用
- 没有成功标志，没有直接指示“向前移动了多少米”的绝对位移信号（仅有瞬时 forward_velocity）