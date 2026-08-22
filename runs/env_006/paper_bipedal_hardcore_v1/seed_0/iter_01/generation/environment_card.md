# 匿名环境理解卡片

## 1. 任务目标
本环境中的双足机器人需要在不规则的粗糙地形上尽可能向前行进，同时保持能量效率。地形包含阶梯状、树桩、坑洞等变化，因此机体必须利用前方 10 个 LIDAR 测距信号来预判地形，动态调整步态。主要目标是：稳定行走、远离摔倒、尽量走远；次要目标是：最小化不必要的关节扭矩（能耗）。该任务的核心是崎岖地形上的持续运动控制，而非单纯到达指定坐标点。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是持续前进通过复杂地形，前进速度和稳定性是首要评估维度，附带能耗优化但不构成同等权重的冲突目标，故排除 multi_objective_task。导航中并没有指定固定目标位置，只是“越远越好”，因此不是 navigation_goal_reaching。没有抓取或驾驶安全约束，也不是稀疏探索环境。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（默认推断）  
- obs 各维含义：

| index | 名称                     | 含义                                   | reward_usable |
|-------|--------------------------|----------------------------------------|---------------|
| 0     | hull_angle               | 身体基座倾角                           | true          |
| 1     | hull_angular_velocity    | 身体基座角速度                         | true          |
| 2     | horizontal_speed         | 质心水平速度                           | true          |
| 3     | vertical_speed           | 质心垂直速度                           | true          |
| 4     | joint_0_angle (hip_1)    | 髋关节 1 角度                          | true          |
| 5     | joint_0_speed (hip_1)    | 髋关节 1 角速度                        | true          |
| 6     | joint_1_angle (knee_1)   | 膝关节 1 角度                          | true          |
| 7     | joint_1_speed (knee_1)   | 膝关节 1 角速度                        | true          |
| 8     | joint_2_angle (hip_2)    | 髋关节 2 角度                          | true          |
| 9     | joint_2_speed (hip_2)    | 髋关节 2 角速度                        | true          |
| 10    | joint_3_angle (knee_2)   | 膝关节 2 角度                          | true          |
| 11    | joint_3_speed (knee_2)   | 膝关节 2 角速度                        | true          |
| 12    | leg_1_ground_contact     | 腿 1 是否接地（0 或 1）                | true          |
| 13    | leg_2_ground_contact     | 腿 2 是否接地（0 或 1）                | true          |
| 14    | lidar_1                  | 第一根 LIDAR 测距值（前方地形高度）    | true          |
| 15    | lidar_2                  | 第二根 LIDAR 测距值                    | true          |
| 16    | lidar_3                  | 第三根 LIDAR 测距值                    | true          |
| 17    | lidar_4                  | 第四根 LIDAR 测距值                    | true          |
| 18    | lidar_5                  | 第五根 LIDAR 测距值                    | true          |
| 19    | lidar_6                  | 第六根 LIDAR 测距值                    | true          |
| 20    | lidar_7                  | 第七根 LIDAR 测距值                    | true          |
| 21    | lidar_8                  | 第八根 LIDAR 测距值                    | true          |
| 22    | lidar_9                  | 第九根 LIDAR 测距值                    | true          |
| 23    | lidar_10                 | 第十根 LIDAR 测距值                    | true          |

注：接地信号为 0/1 标量，间接反映了支撑相，可用于步态激励或摔倒检测。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- bounds: [-1.0, 1.0]  
- 各维含义：
  - action_dim 0: hip_1_torque，第一髋关节力矩
  - action_dim 1: knee_1_torque，第一膝关节力矩
  - action_dim 2: hip_2_torque，第二髋关节力矩
  - action_dim 3: knee_2_torque，第二膝关节力矩

所有动作均为连续值，力矩限幅在 [-1, 1] 内。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形末端，视为成功）
- failure-like termination: body_fallen_over（机体摔倒）
- ambiguous termination: 无
- truncation: 无时间上限截断（隐含 episode 可能在短步数内因摔倒而终止，但未明确提供 truncation 信号）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无法直接读取 success 标志）
- explicit_failure_flag_available: false （同上，无直接 failure 字段）
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；original_reward 被强制屏蔽

终止条件的判断只能通过观测间接进行：
- 摔倒：可依据 hull_angle 绝对值超过某经验阈值（如 >0.5 rad）且可能伴随 vertical_speed 突变或 leg 接触异常；标记为 derived_possible。
- 到达终点：在 episode 结束时若 terminated=True 且未检测到摔倒，可推测为成功。但 compute_reward 中无法直接获取 terminated 标志，只能通过最后一步的 next_obs 状态推测，存在误判风险。

因此，成功/失败的信号是弱可用的，理想情况下应避免依赖终点信号，而是专注于持续前进和生存的激励。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观察（24 维）
- action：当前动作（4 维）
- next_obs：下一步观察（24 维）
- info：被约束为空，无可用字段
- training_progress：当前 prompt 未明确允许，但接口保留；默认不使用

禁止使用：
- original_reward
- official_reward
- 任何未在 obs 或 action 中声明的信号
- 任何 info 字段

## 7. 可用于奖励函数的信号
- position: 无直接位置（但 horizontal_speed 可积分得到水平位移增量）；垂直位移可从 vertical_speed 累积或间接通过高度变化推断（但无绝对高度观测）。
- velocity: horizontal_speed (obs[2])，vertical_speed (obs[3])，各关节角速度 (obs[5,7,9,11])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg_1_ground_contact (obs[12])，leg_2_ground_contact (obs[13])，二值信号，用于检测支撑相或摔倒（例如连续若干步双脚未接地即可能摔倒）。
- action/engine: action 四维力矩（hip_1, knee_1, hip_2, knee_2），可直接用于扭矩惩罚。
- other: LIDAR 读数 (obs[14:24])，提供地形预览，可用于鼓励预判性步态调整，但不易直接转化为标量奖赏，通常用于辅助特征而非独立 reward 项；也可用于检测极端地形。

## 8. 不确定或不可用的信号
- 绝对水平位置：不可用，不能直接计算到目标的距离或前进里程。
- 成功/失败标记：info 为空，无法直接读取；只能通过观测模式 pattern 间接猜测。
- 地形难度/类别：无标签，仅能通过 LIDAR 序列推断。
- 能量消耗直接测量：未提供力矩积分或电机功率，只能用动作平方近似。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: two-legged planar walker
  actuator_type: four torque-controlled revolute joints (two hips, two knees)
  contact_structure: binary foot contact sensors per leg
primary_objectives:
  - walk forward stably on uneven terrain
  - avoid falling (keep hull_angle within safe range)
secondary_objectives:
  - minimize unnecessary joint torque (energy efficiency)
  - maintain anticipatory gait using LIDAR information
main_failure_risks:
  - large hull tilt leading to fall on stairs or pits
  - low forward speed or stagnation on rough patches
  - excessive torques causing jerky motion or self-collision
  - ignoring LIDAR, leading to unprepared contact with obstacles
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 提供稠密的速度型奖赏，鼓励机器人持续向前移动。
  why_required: 主要任务必须前进，否则任何其他目标无意义。
  usable_signals: next_obs[2] (horizontal_speed)
  risks: 如果速度项过大可能导致代理只顾高速冲撞而失去稳定；需限制或组合姿态约束。

- role_id: postural_stability
  purpose: 惩罚身体倾斜和剧烈旋转，防止摔倒。
  why_required: 存活是完成行走的前提，摔倒将导致 episode 终止。
  usable_signals: next_obs[0] (hull_angle), next_obs[1] (hull_angular_velocity)
  risks: 权重过大会阻碍探索，可能使代理不敢迈步。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当任务明确要求“最小化不必要的关节扭矩”时才启用；可在训练中后期逐步加强。
  usable_signals: action (四维力矩), 也可用 joint_speed 的平方项。
  risks: 过强的能量惩罚会抑制行走的自然步态，造成几乎静止的行为。

- role_id: successful_termination_bonus (derived_possible)
  condition_to_use: 仅当通过观测可较高置信度推断出 reached_end_of_terrain 才添加。例如：在 episode 的最后 step，hull_angle 小且 horizontal_speed 低且 vertical_speed 接近零，且腿接触正常。若实现需额外判断逻辑。
  usable_signals: next_obs[0], next_obs[2], next_obs[3], next_obs[12], next_obs[13]
  risks: 推断不可靠，容易将中间静止误认为成功；不建议作为主要奖励来源，只能在高级调优中尝试稀疏奖励。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_failure_penalty
  reason: 无法直接获取失败标志；用 hull_angle 阈值实现的话相当于把姿态惩罚加倍，没必要单独设立角色。
  forbidden_or_missing_signals: info 为空，无 direct_failure_flag

- role_id: lidar_terrain_anticipation
  reason: LIDAR 信号虽可用，但没有直接的对齐方式将其转化为单个标量奖赏。建议让 RL 自主学习 LIDAR 与动作的关系，而不显式建模成奖励项，否则可能