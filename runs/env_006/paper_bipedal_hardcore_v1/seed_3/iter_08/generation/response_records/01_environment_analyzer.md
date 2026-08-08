# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主目标是控制一个双足代理在布满梯子、树桩、坑洼等不规则障碍的崎岖地形上尽可能稳定地向前行走。 次要目标是避免摔倒（终止 episode）并在前进过程中尽量减少不必要的关节力矩消耗。 代理通过 LIDAR 感知前方地形高度，需要学会利用这些读数提前调整步态，以应对比平面复杂得多的地面状况。 本任务不是精确到达某个坐标点，而是持续前进，不应被误认为目标导航或稀疏探索任务。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是持续在崎岖地形上向前移动，没有明确的目标位置坐标；附属目标（省力矩、防摔）服务于主目标，属于典型的 locomtion 任务族。

动力学子类型选择：planar_bipedal_gait  
理由：双足结构，通过髋、膝关节力矩控制步态，在二维平面上前进，符合 planar_bipedal_gait 定义。地形障碍引入适应需求，但未改变基本动力学类型。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32 (推断，典型为 float32)
- 字段含义（index → name, meaning, reward_usable）：
  | index | name                    | meaning                                        | reward_usable |
  |-------|-------------------------|------------------------------------------------|---------------|
  | 0     | hull_angle              | 身体躯干的倾斜角(rad)                           | true          |
  | 1     | hull_angular_velocity   | 躯干角速度(rad/s)                               | true          |
  | 2     | horizontal_speed        | 质心水平方向速度(m/s)                           | true          |
  | 3     | vertical_speed          | 质心垂直速度(m/s)                               | true          |
  | 4     | joint_0_angle           | 髋关节 1 角度(rad)                              | optional      |
  | 5     | joint_0_speed           | 髋关节 1 角速度(rad/s)                          | optional      |
  | 6     | joint_1_angle           | 膝关节 1 角度(rad)                              | optional      |
  | 7     | joint_1_speed           | 膝关节 1 角速度(rad/s)                          | optional      |
  | 8     | joint_2_angle           | 髋关节 2 角度(rad)                              | optional      |
  | 9     | joint_2_speed           | 髋关节 2 角速度(rad/s)                          | optional      |
  | 10    | joint_3_angle           | 膝关节 2 角度(rad)                              | optional      |
  | 11    | joint_3_speed           | 膝关节 2 角速度(rad/s)                          | optional      |
  | 12    | leg_1_ground_contact    | 腿 1 接触地面标志 (连续或判断后二值化)          | true          |
  | 13    | leg_2_ground_contact    | 腿 2 接触地面标志                               | true          |
  | 14-23| lidar_1 … lidar_10      | 前方地形高度测距读数（10 个扫描点）              | true          |

说明：lidar 测量的是前方不同距离处地面的高度或距离，可用于预判障碍物。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- 动作维度含义：
  - action[0]: hip_1_torque，施加到第一个髋关节的力矩，作用于腿 1 髋部
  - action[1]: knee_1_torque
