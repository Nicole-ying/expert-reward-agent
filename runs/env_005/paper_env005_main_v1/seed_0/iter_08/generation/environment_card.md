# 匿名环境理解卡片

## 1. 任务目标
这是一个连续控制的四足机器人行走任务。机器人拥有 8 个扭矩控制的关节（4 条腿，每条腿包含髋关节和踝关节），必须保持稳定站立并向前行进。本体高度必须维持在 0.2~1.0 的健康范围内，超出该范围（跌倒或弹飞）会立即终止回合。策略的核心目标是实现持续、稳定的前向运动，而非仅保持直立不动。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (105,)
- dtype: 假定为 float32（连续值）
- obs[0]:  本体 z 坐标（垂直高度）
- obs[1]:  本体方向四元数分量 1
- obs[2]:  本体方向四元数分量 2
- obs[3]:  本体方向四元数分量 3
- obs[4]:  本体方向四元数分量 4
- obs[5]:  关节 1 角度（第1髋关节）
- obs[6]:  关节 2 角度（第1踝关节）
- obs[7]:  关节 3 角度（第2髋关节）
- obs[8]:  关节 4 角度（第2踝关节）
- obs[9]:  关节 5 角度（第3髋关节）
- obs[10]: 关节 6 角度（第3踝关节）
- obs[11]: 关节 7 角度（第4髋关节）
- obs[12]: 关节 8 角度（第4踝关节）
- obs[13]: 本体前进速度（世界坐标系 x 方向线速度）
- obs[14]: 本体侧向速度（世界坐标系 y 方向线速度）
- obs[15]: 本体垂直速度
- obs[16]: 本体滚转角速度
- obs[17]: 本体俯仰角速度
- obs[18]: 本体偏航角速度
- obs[19]: 关节 1 角速度
- obs[20]: 关节 2 角速度
- obs[21]: 关节 3 角速度
- obs[22]: 关节 4 角速度
- obs[23]: 关节 5 角速度
- obs[24]: 关节 6 角速度
- obs[25]: 关节 7 角速度
- obs[26]: 关节 8 角速度
- obs[27:33]:   躯干接触力/力矩（6D：[Fx, Fy, Fz, Tx, Ty, Tz]，世界坐标系）
- obs[33:39]:   前左大腿接触力/力矩
- obs[39:45]:   前左小腿接触力/力矩
- obs[45:51]:   前左脚接触力/力矩
- obs[51:57]:   前右大腿接触力/力矩
- obs[57:63]:   前右小腿接触力/力矩
- obs[63:69]:   前右脚接触力/力矩
- obs[69:75]:   后左大腿接触力/力矩
- obs[75:81]:   后左小腿接触力/力矩
- obs[81:87]:   后左脚接触力/力矩
- obs[87:93]:   后右大腿接触力/力矩
- obs[93:99]:   后右小腿接触力/力矩
- obs[99:105]:  后右脚接触力/力矩

说明：接触力块共 78 维，对应 13 个身体部位（躯干 + 四条腿的上段/下段/脚，共 1 + 4×3 = 13），每个部位 6 维力与力矩。

## 4. 动作空间 action_space
- type: Box（连续动作）
- shape: (8,)
- 范围：每个动作为 [-1.0, 1.0] 的扭矩值
- action 0: hip_1_torque  （第1条腿髋关节扭矩）
- action 1: ankle_1_torque（第1条腿踝关节扭矩）
- action 2: hip_2_torque   （第2条腿髋关节扭矩）
- action 3: ankle_2_torque（第2条腿踝关节扭矩）
- action 4: hip_3_torque   （第3条腿髋关节扭矩）
- action 5: ankle_3_torque（第3条腿踝关节扭矩）
- action 6: hip_4_torque   （第4条腿髋关节扭矩）
- action 7: ankle_4_torque（第4条腿踝关节扭矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确成功信号。任务期望通过稳定行走最大化存活步数，并非到达某个目标点。
- failure-like termination: 身体高度超出 (0.2, 1.0) 范围（摔倒或弹飞）；状态值出现 NaN 或 Inf。
- ambiguous termination: 无（上述均属于失败终止）。
- truncation: 达到预设的最大步数时截断（时间限制）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（无任何允许的 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均被禁止使用，包括 reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin, x_velocity, y_velocity 等。info 在 step 中返回空字典，禁止依赖任何 info 内容。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前步的完整观察向量（105维）
- action：当前步采用的动作向量（8维）
- next_obs：执行动作后的下一观察向量（105维）
- info 中明确允许的字段：无（info 为空字典）
- training_progress：仅当 prompt 明确允许时可用，当前环境未授权使用

禁止使用：
- original_reward（已被掩码，不可用）
- official_reward 或任何官方奖励项
- 未声明的 info 字段（所有 info 字段均被禁止）
- 未声明的 obs 切片假设

## 7. 可用于奖励函数的信号
- position: 本体高度 obs[0]（用于保持健康范围）；身体垂直位移可间接反映跌倒倾向。
- velocity: 本体前进速度 obs[13]（主要推进目标）；侧向速度 obs[14]（希望抑制）；垂直速度 obs[15]（希望抑制过大波动）；本体角速度 obs[16-18] 可用来惩罚翻滚或俯仰。
- orientation: 四元数 obs[1-4] 可用于保持身体水平（例如惩罚偏离直立）。
- contact: 四条腿的脚部接触力（obs[45:51], obs[63:69], obs[81:87], obs[99:105]）可用于检测足部触地情况，鼓励对称或规避空踩。
- action/engine: 动作向量可用来施加能耗惩罚（如 -‖action‖²），鼓励平滑、低扭矩控制。

## 8. 不确定或不可用的信号
- info 中的所有字段（官方奖励项、位置信息等）均不可用。environment 明确禁止它们用于奖励设计。
- 无显式成功/失败标志，不能使用基于 info 的终止原因。
- 无世界坐标 x 绝对位置（但前进速度可直接观测，可通过累积速度增量间接反映“走得远”，但注意奖励函数是单步、无状态限制，不能累积跨步量，不过仍可基于当前速度进行奖励）。