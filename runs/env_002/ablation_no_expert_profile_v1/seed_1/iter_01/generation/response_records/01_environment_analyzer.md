# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个平面双足步行任务。智能体需要在不平坦地形上协调两条腿的髋、膝四个关节，尽可能远且快地向前行走，同时尽量节约能量消耗。摔倒会立刻结束这一回合，到达地形终点也结束回合。主目标是持续前进（最大化移动距离/速度），附属目标是降低能量消耗。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
confidence: high
reason: 核心目标是控制双足身体在连续地面上持续前进通过地形，附属有能耗约束，符合 locomotion_continuous_control 定义。动作是连续关节扭矩，观测包含身体姿态、关节、接触和前方地形距离，属于典型的平面双足步态控制。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推断自连续值域）
- obs[0]: hull_angle，躯干相对直立方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，躯干角速度，reward_usable: true
- obs[2]: horizontal_velocity，前进方向线速度，reward_usable: true
- obs[3]: vertical_velocity，上下方向线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1触地标志（1.0=触地，0.0=离地），reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2触地标志（1.0=触地，0.0=离地），reward_usable: true
- obs[14]: lidar_0，前方第一个激光雷达距离测量值，reward_usable: true
- obs[15]: lidar_1，前方第二个激光雷达距离测量值，reward_usable: true
- obs[16]: lidar_2，前方第三个激光雷达距离测量值，reward_usable: true
- obs[17]: lidar_3，前方第四个激光雷达距离测量值，reward_usable: true
- obs[18]: lidar_4，前方第五个激光雷达距离测量值，reward_usable: true
- obs[19]: lidar_5，前方第六个激光雷达距离测量值，reward_usable: true
- obs[20]: lidar_6，前方第七个激光雷达距离测量值，reward_usable: true
- obs[21]: lidar_7，前方第八个激光雷达距离测量值，reward_usable: true
- obs[22]: lidar_8，前方第九个激光雷达距离测量值，reward_usable: true
- obs[23]: lidar_9，前方第十个激光雷达距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [4]
- bounds: [-1.0, 1.0] 每个关节
- action[0]: hip_torque_leg1，施加到腿1髋关节的扭矩
- action[1]: knee_torque_leg1，施加到腿1膝关节的扭矩
- action[2]: hip_torque_leg2，施加到腿2髋关节的扭矩
- action[3]: knee_torque_leg2，施加到腿2膝关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形终点，可视为成功完成前进目标）
- failure-like termination: body_fallen_over（身体摔倒，任务失败）
- ambiguous termination: 无
- truncation: 环境描述未提及时间/步数截断，但常见实现可能存在最大步数截断，当前未提供任何信息

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 返回空字典，无 success 标记）
- explicit_failure_flag_available: false（同上，无 failure 标记）
- allowed_info_fields: 无（info 为空字典，但根据任务描述，允许使用的 info 字段未预先定义，当前无法使用任何 info 信号）
- forbidden_or_uncertain_info_fields: 所有 info 字段（当前环境返回 {}，因此任何假设的 info 字段均禁止使用）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前时刻的观察数组（可以切片使用第3节列出的各维度）
- action：当前时刻采取的动作数组
- next_obs：下一时刻的观察数组
- info：当前环境的 info 字典（已知为空，故不能使用任何键）
- training_progress：仅当明确提示允许使用时才允许；当前任务未提示，不建议使用

禁止使用：
- original_reward（已被遮蔽的官方奖励信号）
- 未在观察空间中声明的 obs 切片
- 未在允许列表中的 info 字段
- 任何假设的终止原因变量（如 done、terminated 等，因为它们不在函数参数中，info 中也没有）

## 7. 可用于奖励函数的信号
- position: 无绝对位置信息；可通过速度积分隐式度量前进距离（例如利用 horizontal_velocity）。
- velocity: horizontal_velocity（前进速度），vertical_velocity（上下速度），hull_angular_velocity（躯干转动速度），各关节速度 hip1_speed, knee1_speed, hip2_speed, knee2_speed。
- orientation: hull_angle（躯干倾斜角度），可用于保持直立。
- contact: leg1_contact, leg2_contact（触地标志，可用于奖励合理的步态切换或惩罚不必要的离地）。
- action/engine: action[0..3] 各关节扭矩，可用于计算能耗（例如扭矩平方和）。
- other: lidar_0..9（前方地形距离），可用于指导安全高度变化或地形感知，但主目标前进不应直接依赖地形距离，可作为辅助约束。

## 8. 不确定或不可用的信号
- 绝对位置/前进距离：观察空间无 x 坐标或位移累积值，无法直接获取智能体沿地形已行进的距离。
- 终点到达信号：reached_end_of_terrain 终止条件在观察中无对应标志；info 为空，无法在 compute_reward 中判断当前 step 是否触发了终点到达。
- 摔倒信号：body_fallen_over 终止条件同样在观察中无直接标志，无法在奖励函数中直接使用。
- 地形相关信息：除 lidar 外无其他高度图或摩擦系数等，无法获取更多地形属性。
- 步态阶段/步行周期：观察无明确步态相位，仅能通过腿接触和关节角度间接推断。
- 任何 info 键：由于 info 为 {}，无法使用 success、failure、completion 等常见环境标记。
