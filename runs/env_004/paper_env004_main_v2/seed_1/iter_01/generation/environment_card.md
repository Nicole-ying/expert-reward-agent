# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿铰接体，通过三个关节的连续力矩驱动，使其持续向前跳跃移动。策略必须在保持身体直立、关节力矩高效的前提下，尽可能产生正向的水平位移。跌倒、身体高度异常、躯干倾角过大或状态数值溢出都会导致回合立即结束。奖励设计应鼓励“持续向前推进”，而非仅仅“保持不倒”。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: [11]
- dtype: float32 (假设，通常为float)
- obs[0]: torso_height —— 主体质心垂直高度
- obs[1]: torso_angle —— 躯干相对于垂直方向的倾角
- obs[2]: upper_joint_angle —— 大腿关节角度
- obs[3]: lower_joint_angle —— 小腿关节角度
- obs[4]: foot_joint_angle —— 足部关节角度
- obs[5]: forward_velocity —— 主体水平速度（向前为正）
- obs[6]: vertical_velocity —— 主体垂直速度
- obs[7]: torso_angular_velocity —— 躯干角速度
- obs[8]: upper_joint_speed —— 大腿关节角速度
- obs[9]: lower_joint_speed —— 小腿关节角速度
- obs[10]: foot_joint_speed —— 足部关节角速度

## 4. 动作空间 action_space
- type: Box，连续动作，范围 [-1.0, 1.0]
- action 0: upper_joint_torque —— 大腿关节力矩
- action 1: lower_joint_torque —— 小腿关节力矩
- action 2: foot_joint_torque —— 足部关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止条件；任务目标是持续前进，并不存在“到达终点”的概念，因此没有成功终止。
- failure-like termination: body_height_outside_healthy_range（身体高度超标，可能跌倒或跳太高）、torso_angle_outside_healthy_range（躯干倾角过大，即将倒下）、state_value_outside_finite_healthy_range（观测值出现 NaN 或 inf）均属于失败终止。
- ambiguous termination: 无。
- truncation: time_limit_reached，回合因达到最大步数而截断，既非成功也非失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: **空**（任何 info 字段均不得使用）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_survive, x_position, y_position, z_distance_from_origin 等（所有列出的字段均禁止使用，且 info 实际返回 {}）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测向量
- action：当前动作向量
- next_obs：下一时刻观测向量
- training_progress：**仅当 prompt 明确声明时**才允许使用（本环境未声明，应禁止使用）

禁止使用：
- original_reward（官方奖励已被屏蔽）
- info 中的任何字段（allowed_info_fields 为空，且所有有意义字段均在 forbidden_info_fields 中）
- 任何未在观测空间中声明的信号（如世界坐标 x、y、z_distance）
- 奖励函数必须是无状态的，不能跨 step 累加信息

## 7. 可用于奖励函数的信号
- position: torso_height（obs[0]）—— 可用于惩罚过高或过低，保持健康高度区间
- velocity: forward_velocity（obs[5]）、vertical_velocity（obs[6]）、torso_angular_velocity（obs[7]）、upper_joint_speed（obs[8]）、lower_joint_speed（obs[9]）、foot_joint_speed（obs[10]）
- orientation: torso_angle（obs[1]）—— 可惩罚偏离直立的角度
- contact: 无直接接触传感器（仅能通过高度和倾角间接推断脚与地面的接触情况）
- action/engine: action[0], action[1], action[2] —— 可惩罚过大的力矩或过快的动作变化

## 8. 不确定或不可用的信号
- 绝对世界位置（x, y, z_distance）不可用，禁止通过 info 或观测获取。
- 任何来自 info 的奖励分量（reward_forward, reward_ctrl, reward_survive 等）均不可用。
- 没有明确的 success/failure flag，无法通过标志位直接判断成功或失败。
- 无接触力、地面反作用力等物理交互信号。
- 回合持续时间步数不可直接用于奖励（函数无状态）；但可通过 training_progress 提供全局进度（本环境未允许使用）。