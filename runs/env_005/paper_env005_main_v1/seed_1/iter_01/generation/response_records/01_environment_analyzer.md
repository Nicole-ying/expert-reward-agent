# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个 3D 四足机器人（8 个扭矩控制关节）在平坦地面上稳定地向前行走或奔跑。机器人必须保持身体高度在 0.2~1.0 的范围内；超出该范围（摔倒或蹦飞）或出现数值异常将立即终止回合。核心目标是实现持续、稳定的前进运动，而非仅仅保持站立。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (105,)
- dtype: float32
- obs[0]: 主体垂直高度 (body_z)
- obs[1]: 主体姿态四元数第 1 分量 (body_orientation)
- obs[2]: 主体姿态四元数第 2 分量
- obs[3]: 主体姿态四元数第 3 分量
- obs[4]: 主体姿态四元数第 4 分量
- obs[5]: 关节 1 角度 (hip_1)
- obs[6]: 关节 2 角度 (ankle_1)
- obs[7]: 关节 3 角度 (hip_2)
- obs[8]: 关节 4 角度 (ankle_2)
- obs[9]: 关节 5 角度 (hip_3)
- obs[10]: 关节 6 角度 (ankle_3)
- obs[11]: 关节 7 角度 (hip_4)
- obs[12]: 关节 8 角度 (ankle_4)
- obs[13]: 主体世界 x 方向速度（前进速度）
- obs[14]: 主体世界 y 方向速度（横向速度）
- obs[15]: 主体垂直速度 (body_z_velocity)
- obs[16]: 主体滚转角速度 (roll)
- obs[17]: 主体俯仰角速度 (pitch)
- obs[18]: 主体偏航角速度 (yaw)
- obs[19]: 关节 1 角速度
- obs[20]: 关节 2 角速度
- obs[21]: 关节 3 角速度
- obs[22]: 关节 4 角速度
- obs[23]: 关节 5 角速度
- obs[24]: 关节 6 角速度
- obs[25]: 关节 7 角速度
- obs[26]: 关节 8 角速度
- obs[27:33]: 躯干接触力/力矩 (6D)
- obs[33:39]: 左前腿上段接触力/力矩
- obs[39:45]: 左前腿下段接触力/力矩
- obs[45:51]: 左前足接触力/力矩（强地面接触信号）
- obs[51:57]: 右前腿上段接触力/力矩
- obs[57:63]: 右前腿下段接触力/力矩
- obs[63:69]: 右前足接触力/力矩（强地面接触信号）
- obs[69:75]: 左后腿上段接触力/力矩
- obs[75:81]: 左后腿下段接触力/力矩
- obs[81:87]: 左后足接触力/力矩（强地面接触信号）
- obs[87:93]: 右后腿上段接触力/力矩
- obs[93:99]: 右后腿下段接触力/力矩
- obs[99:105]: 右后足接触力/力矩（强地面接触信号）

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- dtype: float32
- bounds: [-1.0, 1.0] 每个维度
- action 0: 髋关节 1 力矩 (hip_1_torque)
- action 1: 踝关节 1 力矩 (ankle_1_torque)
- action 2: 髋关节 2 力矩 (hip_2_torque)
- action 3: 踝关节 2 力矩 (ankle_2_torque)
- action 4: 髋关节 3 力矩 (hip_3_torque)
- action 5: 踝关节 3 力矩 (ankle_3_torque)
- action 6: 髋关节 4 力矩 (hip_4_torque)
- action 7: 踝关节 4 力矩 (ankle_4_torque)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无（本环境没有明确的成功到达目标，只有通过截断结束一个 episode）
- failure-like termination:  
  a) 身体高度超出健康范围 (body_height_outside_healthy_range)，即 body_z ≤ 0.2 或 body_z ≥ 1.0  
  b) 任何状态值出现 NaN 或无穷大 (state_value_outside_finite_range)
- ambiguous termination: 无
- truncation: 到达预设的时间步上限（time_limit_reached），视为正常结束

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，没有任何允许字段）
- forbidden_or_uncertain_info_fields: 所有与官方奖励相关的字段（如 reward_forward, reward_ctrl, reward_contact, reward_survive）以及 x_position, y_position, distance_from_origin, x_velocity, y_velocity 等均被禁止，不可使用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (上一状态)
- action (当前动作)
- next_obs (下一状态)
- info 中明确允许的字段（当前为空，故 info 实质上不可用）

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段（例如 reward_forward, reward_ctrl 等）
- training_progress（prompt 未明确允许，不得使用）
- 终止标志（terminated/truncated）或任何隐含的 episode 结束信号

## 7. 可用于奖励函数的信号
- position: 主体高度 (obs[0])，主体姿态四元数 (obs[1:5])，各关节角度 (obs[5:13])
- velocity: 前进速度 (obs[13])，横向速度 (obs[14])，垂直速度 (obs[15])，滚转/俯仰/偏航角速度 (obs[16:18])，各关节角速度 (obs[19:27])
- orientation: 四元数姿态，可用于惩罚大幅度翻滚或俯仰
- contact: 各足端接触力 (obs[45:51], obs[63:69], obs[81:87], obs[99:105])，可推断足部着地状态与地面反作用力
- action/engine: 关节力矩动作本身 (action[0:8])，可用于惩罚过大的控制量

## 8. 不确定或不可用的信号
- 任何 info 字段（info 为空且所有官方奖励字段均被禁止）
- 绝对位置（x_position, y_position, distance_from_origin）
- 官方奖励组件（reward_forward, reward_ctrl, reward_contact, reward_survive）
- 成功/失败标记（环境中不存在显式的 success 或 failure 标志）
- 训练进度（training_progress 未声明允许使用）
- 原始奖励值（original_reward 被强制遮蔽）
