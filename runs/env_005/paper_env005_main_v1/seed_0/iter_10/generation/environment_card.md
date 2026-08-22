# 匿名环境理解卡片

## 1. 任务目标
控制一个 3D 四足机器人（8 个扭矩驱动关节）向前行走或奔跑，同时保持身体直立。机器人的身体高度必须维持在一个健康的范围内（0.2 ~ 1.0），超出范围即终止。核心目标是维持稳定的前进运动，而不是仅仅保持平衡不动。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box（连续）
- shape: [105]
- dtype: 推测为 float32（连续物理量）
- obs[0]: 主躯干的垂直位置（body_z）
- obs[1]: 机体姿态四元数分量 1
- obs[2]: 机体姿态四元数分量 2
- obs[3]: 机体姿态四元数分量 3
- obs[4]: 机体姿态四元数分量 4
- obs[5]: 第 1 个关节角（hip_1）
- obs[6]: 第 2 个关节角（ankle_1）
- obs[7]: 第 3 个关节角（hip_2）
- obs[8]: 第 4 个关节角（ankle_2）
- obs[9]: 第 5 个关节角（hip_3）
- obs[10]: 第 6 个关节角（ankle_3）
- obs[11]: 第 7 个关节角（hip_4）
- obs[12]: 第 8 个关节角（ankle_4）
- obs[13]: 主躯干世界坐标系 x 方向速度（前进速度）
- obs[14]: 主躯干世界坐标系 y 方向速度（侧向速度）
- obs[15]: 主躯干世界坐标系 z 方向速度（垂直速度）
- obs[16]: 机体滚转角速度（roll）
- obs[17]: 机体俯仰角速度（pitch）
- obs[18]: 机体偏航角速度（yaw）
- obs[19]: 第 1 个关节角速度
- obs[20]: 第 2 个关节角速度
- obs[21]: 第 3 个关节角速度
- obs[22]: 第 4 个关节角速度
- obs[23]: 第 5 个关节角速度
- obs[24]: 第 6 个关节角速度
- obs[25]: 第 7 个关节角速度
- obs[26]: 第 8 个关节角速度
- obs[27:33]: 躯干（torso）接触力/力矩（fx, fy, fz, tx, ty, tz）
- obs[33:39]: 前左上腿接触力/力矩
- obs[39:45]: 前左下腿接触力/力矩
- obs[45:51]: 前左脚掌接触力/力矩（地面接触信号强）
- obs[51:57]: 前右上腿接触力/力矩
- obs[57:63]: 前右下腿接触力/力矩
- obs[63:69]: 前右脚掌接触力/力矩
- obs[69:75]: 后左上腿接触力/力矩
- obs[75:81]: 后左下腿接触力/力矩
- obs[81:87]: 后左脚掌接触力/力矩
- obs[87:93]: 后右上腿接触力/力矩
- obs[93:99]: 后右下腿接触力/力矩
- obs[99:105]: 后右脚掌接触力/力矩

## 4. 动作空间 action_space
- type: Box，连续，形状 [8]，每个维度范围 [-1.0, 1.0]
- action 0: 第 1 个髋关节扭矩（hip_1）
- action 1: 第 1 个踝关节扭矩（ankle_1）
- action 2: 第 2 个髋关节扭矩（hip_2）
- action 3: 第 2 个踝关节扭矩（ankle_2）
- action 4: 第 3 个髋关节扭矩（hip_3）
- action 5: 第 3 个踝关节扭矩（ankle_3）
- action 6: 第 4 个髋关节扭矩（hip_4）
- action 7: 第 4 个踝关节扭矩（ankle_4）

所有动作直接输出为关节扭矩，无需缩放即可施加到仿真器中。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功信号；可以认为 episode 因时间限制而截断（truncated）且未发生失败终止即为成功，即机器人存活并保持运动直到时间结束。
- failure-like termination: 身体高度超出健康范围（z < 0.2 或 z > 1.0）或状态中出现 NaN/inf 值。
- ambiguous termination: 无。
- truncation: 达到预设的时间步上限。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}）
- forbidden_or_uncertain_info_fields: 全部 info 字段均不可用，包括任何可能的官方奖励项坐标等（已强制屏蔽）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）
- action（当前动作）
- next_obs（下一时刻观测）
- info 中明确允许的字段（此处为空，不可用）
- training_progress（仅当接口说明明确允许时可用，本环境无此许可，不可用）

禁止使用：
- original_reward（官方奖励已遮掩）
- 任何未声明的 info 字段（此处均未声明）
- 任何未在上述索引中说明的 obs 切片

## 7. 可用于奖励函数的信号
- position: 躯干垂直位置 (obs[0])，用于鼓励保持在健康范围中位。
- velocity: 前进速度 (obs[13] 的 body_x_velocity)，侧向速度 (obs[14])，垂直速度 (obs[15])，角速度 (obs[16:19])，各关节速度 (obs[19:27])。
- orientation: 躯干姿态四元数 (obs[1:5])，可判断身体倾斜程度。
- contact: 脚掌接触力 (obs[45:51] 前左脚、obs[63:69] 前右脚、obs[81:87] 后左脚、obs[99:105] 后右脚)，可用于判断支撑相与摆动相，鼓励合理步态。
- action/engine: 动作本身（action）可用于动作平滑性或能耗调节；关节位置 (obs[5:13]) 可用于观察姿态。

## 8. 不确定或不可用的信号
- 世界坐标 x/y 位置：未提供。
- 速度目标或期望前进速度：未提供。
- 能量/功耗直接测量：未提供（但可通过关节扭矩与速度间接估算，能耗本身不是硬指标）。
- 机器人实际世界朝向的绝对偏航：可以从四元数和关节状态推断，但 forward 方向是固定的世界 x 轴（obs[13] 直接给）。