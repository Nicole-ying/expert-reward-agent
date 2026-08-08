# 匿名环境理解卡片

## 1. 任务目标
双足机器人需要在布满障碍（阶梯、树桩、坑洼等）的粗糙地形上持续向前行走，尽可能走得远且高效。  
主要目标是**稳定前进**并避免摔倒；次要目标包括**最小化关节力矩消耗**和**最终抵达地形末端**。  
机器人可利用前方的激光雷达（LiDAR）感知地形，提前调整步态。  
到达地形末端会正常结束，摔倒则提前失败。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason:  
- 核心任务是让双足机器人持续向前通过具有物理挑战的地形，没有指定的目标点位置，到达末端只是环境终止条件而非必须达成的任务核心。  
- 附属目标（节能、抵达末端）是优化方向，但并非替代前进这个底层持续控制目标。  
- 不选multi_objective_task，因为多个目标虽然被提及，但前进与平衡是主要需求，节能和到达末端是附加期望，不存在权重相当且冲突的核心多目标。  
- 动力学子类型（dynamics_subtype）：planar_bipedal_gait（平面双足步态，地形高度变化但不属于全 3D 操纵）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [24]  
- dtype: 根据 float32 推断  
- 各维度含义及 reward 可用性：

| 索引 | 名称                      | 含义                           | reward_usable |
|------|---------------------------|--------------------------------|---------------|
| 0    | hull_angle                | 躯干倾斜角                     | true          |
| 1    | hull_angular_velocity     | 躯干角速度                     | true          |
| 2    | horizontal_speed           | 质心水平速度                   | true          |
| 3    | vertical_speed             | 质心垂直速度                   | true          |
| 4    | joint_0_angle （hip_1）    | 髋关节1角度                    | true          |
| 5    | joint_0_speed             | 髋关节1角速度                  | true          |
| 6    | joint_1_angle （knee_1）   | 膝关节1角度                    | true          |
| 7    | joint_1_speed             | 膝关节1角速度                  | true          |
| 8    | joint_2_angle （hip_2）    | 髋关节2角度                    | true          |
| 9    | joint_2_speed             | 髋关节2角速度                  | true          |
| 10   | joint_3_angle （knee_2）   | 膝关节2角度                    | true          |
| 11   | joint_3_speed             | 膝关节2角速度                  | true          |
| 12   | leg_1_ground_contact      | 左腿触地指示（二值）           | true          |
| 13   | leg_2_ground_contact      | 右腿触地指示（二值）           | true          |
| 14–23| lidar_1…lidar_10          | 前方10个LiDAR测距（地形高度）   | 谨慎使用      |

- 注意：LiDAR原始数值是距离测量值，可用于隐式学习地形应对，但不建议直接作为稠密奖励信号，因为其语义与前进或平衡无直接线性关系。

## 4. 动作空间 action_space
- type: Box  
- shape: [4]  
- 范围: [-1.0, 1.0]  
- 每个动作维度含义：
  - action[0]: hip_1_torque – 第一个髋关节力矩  
  - action[1]: knee_1_torque – 第一个膝关节力矩  
  - action[2]: hip_2_torque – 第二个髋关节力矩  
  - action[3]: knee_2_torque – 第二个膝关节力矩  
- 连续力矩控制，无离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain – 抵达地形末端，环境正常结束。
- failure-like termination: body_fallen_over – 身体倾倒（典型失败）。
- ambiguous termination: 无。
- truncation: 无明显时间截断（原文未提及 max steps，但可能存在于环境中，视为 ambiguous）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （info 字段为空，不允许使用）
- explicit_failure_flag_available: **false**
- allowed_info_fields: []  （info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（因为不允许使用任何 info 内容）
- 终止原因只能通过以下方式**推断**（derived_possible）：
  - **摔倒推断**：终止时 `next_obs[0]`（hull_angle）很可能超过阈值（如>0.8 rad），或者两腿触地指示同时为0且躯干姿态异常。可利用 `next_obs` 在 reward 中检测。
  - **到达终点推断**：终止时 `next_obs` 的 hull_angle 较小且无异常，但无法从观测直接区分；因为无位置信息，可通过 episode 忽然结束且未触发摔倒检测来判断。奖励函数设计中可仅通过前进速度奖励覆盖此目标，避免依赖显式到达奖励。

## 6. reward 函数接口契约
函数签名（不可修改参数列表）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
- **允许使用**：
  - `obs` 全部 24 维观测
  - `action` 全部 4 维动作
  - `next_