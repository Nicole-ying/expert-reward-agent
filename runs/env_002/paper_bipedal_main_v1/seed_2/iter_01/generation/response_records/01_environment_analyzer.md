# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 双足机器人运动任务。机器人需要在崎岖地形上稳定行走，尽可能走得远、走得快，同时尽量节省能量。机器人拥有髋关节和膝关节，需要协调四条关节（双腿各一髋一膝）产生连续的步态。如果身体摔倒，回合立即终止；如果抵达地形终点，回合也终止。任务希望机器人学会快速、高效的前进步态，避免摔倒。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (24,)
- dtype: 推断为 float32 或 float64
- obs[0]: hull_angle —— 主躯干相对于直立方向的角度
- obs[1]: hull_angular_velocity —— 主躯干角速度
- obs[2]: horizontal_velocity —— 水平（前进/后退）线速度
- obs[3]: vertical_velocity —— 垂直方向线速度
- obs[4]: hip1_angle —— 腿1髋关节角度
- obs[5]: hip1_speed —— 腿1髋关节角速度
- obs[6]: knee1_angle —— 腿1膝关节角度
- obs[7]: knee1_speed —— 腿1膝关节角速度
- obs[8]: leg1_contact —— 腿1触地标志（1.0 触地，0.0 离地）
- obs[9]: hip2_angle —— 腿2髋关节角度
- obs[10]: hip2_speed —— 腿2髋关节角速度
- obs[11]: knee2_angle —— 腿2膝关节角度
- obs[12]: knee2_speed —— 腿2膝关节角速度
- obs[13]: leg2_contact —— 腿2触地标志（1.0 触地，0.0 离地）
- obs[14..23]: lidar_0..lidar_9 —— 10 个激光雷达距离测量值，用于感知前方地形

## 4. 动作空间 action_space
- type: Box（连续动作）
- shape: (4,)
- 取值范围: 每维均在 [-1.0, 1.0]（关节扭矩）
- action 0: hip_torque_leg1 —— 腿1髋关节扭矩
- action 1: knee_torque_leg1 —— 腿1膝关节扭矩
- action 2: hip_torque_leg2 —— 腿2髋关节扭矩
- action 3: knee_torque_leg2 —— 腿2膝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（抵达地形终点，通常算成功完成任务）
- failure-like termination: body_fallen_over（身体摔倒，失败）
- ambiguous termination: 无
- truncation: 无（step 返回的 terminated 标志直接表示结束，没有另行截断）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，没有 success 字段）
- explicit_failure_flag_available: false（info 为空，没有 failure 字段）
- allowed_info_fields: 无（info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs —— 当前观测（24维向量，含义如第3节所述）
- action —— 当前动作（4维连续扭矩）
- next_obs —— 下一时刻观测（24维向量）
- info 中明确允许的字段：本次 info 为空，因此不能依赖任何 info 字段
- training_progress —— 仅当 prompt 明确允许时才使用（此处不允许）

禁止使用：
- original_reward（官方奖励已被掩码，不可用）
- official_reward 或任何从原始环境获得的奖励信号
- 未声明的 info 字段（所有字段均未声明，故禁止一切 info 访问）
- 未声明的 obs 切片（例如不能凭空使用 obs 中未解释的高维分量）

## 7. 可用于奖励函数的信号
- position（通过观测可间接获得位移增量）：
  - 可利用连续帧的 horizontal_velocity（obs[2] 与 next_obs[2]）计算前进距离增量
  - 但注意没有直接的位置坐标
- velocity：
  - obs[2] horizontal_velocity（前进速度）
  - obs[3] vertical_velocity（垂直速度，可用于惩罚跳跃或坠落）
- orientation：
  - obs[0] hull_angle（躯干倾角，接近 0 表示直立）
  - obs[1] hull_angular_velocity（倾摆角速度，可用于惩罚快速翻滚）
- contact：
  - obs[8] leg1_contact（触地标志）
  - obs[13] leg2_contact（触地标志）
- action/engine：
  - 动作扭矩（action[0..3]）可用于衡量能耗（例如平方和）

## 8. 不确定或不可用的信号
- 明确的成功/失败标志（info 中没有相关字段）
- 原始官方奖励值（已被掩码，不可用）
- 地形终点位置或剩余距离（未在观测中直接给出）
- 真实的全局位置坐标（观测中只有速度，没有积分后的绝对位置）
- 是否到达终点的标记（只能从 terminated 结合其他观测推断，不能直接获取）
