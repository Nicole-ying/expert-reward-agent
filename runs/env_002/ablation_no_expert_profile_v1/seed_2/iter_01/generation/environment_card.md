# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个二维双足身体在不平坦地形上尽可能远、尽可能快地向前行走。  
附属目标：最小化能量消耗，保持身体稳定（防止摔倒）以延长运行时间。  
不混淆的目标：没有明确的终点坐标或抓取任务，仅追求前进距离与速度。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心是持续前进通过地形，附属有速度、能耗和平衡要求，符合 locomotion_continuous_control 定义。不涉及指定目标点或生存唯一目标。

动力学子类型 dynamics_subtype: planar_bipedal_gait （平面双足步态前进）

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float64（推断）  

各维度含义：  
- obs[0]: hull_angle – 主体相对于竖直方向的倾角，reward_usable: true  
- obs[1]: hull_angular_velocity – 主体角速度，reward_usable: true  
- obs[2]: horizontal_velocity – 前进/后退线速度，reward_usable: true  
- obs[3]: vertical_velocity – 上下线速度，reward_usable: true  
- obs[4]: hip1_angle – 腿1髋关节角度，reward_usable: true（可用于姿态约束）  
- obs[5]: hip1_speed – 腿1髋关节角速度，reward_usable: true  
- obs[6]: knee1_angle – 腿1膝关节角度，reward_usable: true  
- obs[7]: knee1_speed – 腿1膝关节角速度，reward_usable: true  
- obs[8]: leg1_contact – 腿1触地标志（1.0接触，0.0不接触），reward_usable: true（可用于鼓励交替支撑）  
- obs[9]: hip2_angle – 腿2髋关节角度，reward_usable: true  
- obs[10]: hip2_speed – 腿2髋关节角速度，reward_usable: true  
- obs[11]: knee2_angle – 腿2膝关节角度，reward_usable: true  
- obs[12]: knee2_speed – 腿2膝关节角速度，reward_usable: true  
- obs[13]: leg2_contact – 腿2触地标志，reward_usable: true  
- obs[14]~obs[23]: lidar_0..9 – 10个激光雷达距离测量值（前方地形轮廓），reward_usable: true，但典型奖励中较少依赖全部，可用于避障/垂直地形的感知。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- bounds: [-1.0, 1.0] per dimension，连续控制  
- action_dim 0: hip_torque_leg1 – 作用在腿1髋关节的扭矩  
- action_dim 1: knee_torque_leg1 – 作用在腿1膝关节的扭矩  
- action_dim 2: hip_torque_leg2 – 作用在腿2髋关节的扭矩  
- action_dim 3: knee_torque_leg2 – 作用在腿2膝关节的扭矩  

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（走到地形终点，可视为成功）  
- failure-like termination: body_fallen_over（身体摔倒）  
- ambiguous termination: 无  
- truncation: 未明确提及（可能由环境默认步数限制触发，但未在描述中给出）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中无字段指示）  
- explicit_failure_flag_available: false（info 中无字段指示）  
- allowed_info_fields: 无（step 返回 info 为空字典）  
- forbidden_or_uncertain_info_fields: 除空字典外的任何键

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）  
- action（采取的动作）  
- next_obs（下一观测）  
- info（当前为空，无可用字段）  
- training_progress：除非环境卡片明确允许，否则不使用（此处未允许）

禁止使用：
- original_reward  
- 任何未被明确允许的 info 或 obs 切片（如假设的终止原因、子目标等）

## 7. 可用于奖励函数的信号
- position（位置类信号）：无直接全局位置，但有 lidar 提供前方地形轮廓，可间接反映位置进展。  
- velocity（速度类）：  
  - horizontal_velocity (obs[2]) – 用于鼓励前进速度  
  - vertical_velocity (obs[3]) – 可用于抑制剧烈上下震荡  
  - hull_angular_velocity (obs[1]) – 可用于惩罚快速翻滚  
- orientation（姿态类）：  
  - hull_angle (obs[0]) – 鼓励身体保持竖直  
- contact（接触类）：  
  - leg1_contact (obs[8]), leg2_contact (obs[13]) – 可用于交替支撑/步态节律的奖励  
  - 关节角度角速度：hip1/2, knee1/2 的角度和速度可约束极端姿态或高频率震荡  
- action/engine（动作/能耗类）：  
  - action 的平方或范数 – 用于惩罚能量消耗，实现“最小化能量消耗”附属目标  
- other：  
  - lidar 距离数据可用于检测是否陷入凹坑或过大坡度，但非必需

## 8. 不确定或不可用的信号
- 明确的成功/失败标识：无，无法在奖励中直接使用 termination 原因。  
- 地形终点距离或绝对位置：不可用。  
- 能量消耗的真实物理测量：不可用，需通过动作范数近似。  
- 身体各部位世界坐标：不可用（观测仅给出局部关节状态和有限距离 lidar）。