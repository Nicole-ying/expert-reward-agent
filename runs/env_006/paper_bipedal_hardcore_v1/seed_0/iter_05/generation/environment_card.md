# 匿名环境理解卡片

## 1. 任务目标
双足机器人在布满阶梯、树桩、坑洞等不规则障碍物的崎岖地面上持续前进，走的越远且越高效越好。机器人配备 LIDAR 测距仪，可提前感知前方地形高度变化。每条腿的髋关节和膝关节均由独立的力矩控制器驱动。任务是让机器人学会在不规则地面上稳定行走，利用 LIDAR 信息提前调整步态以应对前方障碍，同时避免摔倒并尽量减少不必要的关节力矩消耗。当机器人摔倒或到达地形尽头时，回合结束。

核心目标：持续、稳定地通过崎岖地面继续前进（或到达尽头），次要目标：避免摔倒、降低关节力矩消耗。不应将“到达终点”视作唯一导航目标，而是作为行进完成的自然结果。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务核心是在复杂地形上连续移动，主目标是保持稳定并尽量走远，没有明确的坐标导航目标；附属能耗和姿态要求不影响主目标性质。最匹配 locomotion_continuous_control（持续前进通过地形）。

动力学子类型 dynamics_subtype: planar_bipedal_gait  
解释：双足、髋膝四关节力矩驱动，平面步态前进，尽管地形粗糙，但动力学本质仍是平面双足行走。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推断）
- 字段详解（index 从 0 开始）：

| Index | 名称 | 含义 | reward_usable |
|-------|------|------|---------------|
| 0 | hull_angle | 身体姿态角（俯仰） | true |
| 1 | hull_angular_velocity | 身体姿态角速度 | true |
| 2 | horizontal_speed | 质心水平速度（前进方向） | true |
| 3 | vertical_speed | 质心垂直速度 | true |
| 4 | hip_1_angle | 髋关节 1 角度 | true |
| 5 | hip_1_speed | 髋关节 1 角速度 | true |
| 6 | knee_1_angle | 膝关节 1 角度 | true |
| 7 | knee_1_speed | 膝关节 1 角速度 | true |
| 8 | hip_2_angle | 髋关节 2 角度 | true |
| 9 | hip_2_speed | 髋关节 2 角速度 | true |
| 10 | knee_2_angle | 膝关节 2 角度 | true |
| 11 | knee_2_speed | 膝关节 2 角速度 | true |
| 12 | leg_1_ground_contact | 腿 1 触地指示（0/1） | true |
| 13 | leg_2_ground_contact | 腿 2 触地指示（0/1） | true |
| 14‑23 | lidar_1 … lidar_10 | 前方 10 束激光测距读数（地形高度） | true（高级用法） |

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [4]
- action bounds: [-1.0, 1.0]
- 各维度含义：

| Action dim | 名称 | 含义 |
|------------|------|------|
| 0 | hip_1_torque | 施加于第一个髋关节的力矩 |
| 1 | knee_1_torque | 施加于第一个膝关节的力矩 |
| 2 | hip_2_torque | 施加于第二个髋关节的力矩 |
| 3 | knee_2_torque | 施加于第二个膝关节的力矩 |

控制类型：continuous（力矩控制，值被截断在 [-1,1] 内）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination：reached_end_of_terrain（到达地形尽头）
- failure-like termination：body_fallen_over（身体摔倒）
- ambiguous termination：无
- truncation：不含截断终止（可能仅在达到最大时间步时截断，但未说明）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（info 为空）

终止原因可通过最后一步的观测信号间接推断，属于 derived_possible：
- 摔倒：可通过 hull_angle 绝对值超过阈值（如 > 0.7 rad）、或 vertical_speed 突然大幅向下等判断。由于终止仅发生于身体摔倒，所以 terminated 且满足摔倒观测条件时一定是摔倒。
- 到达终点：episode 在正常的行进中提前结束，且观测中无摔倒迹象（hull_angle 较小、速度正常），可合理推断为 reached_end_of_terrain。

在奖励函数设计中，可以在终态时利用推导信号分配成功奖励或失败惩罚，但不可依赖显式 info 标签。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，24维）
- action（当前动作，4维）
- next_obs（下一观测，24维）
- info 中明确允许的字段（当前无允许字段，info 始终为空，故 info 不可用）
- training_progress 仅在 prompt 明确允许时使用（此处未提及，建议禁用）

禁止使用：
- original_reward（官方奖励被屏蔽）
- official_reward（同源）
- 任何未声明的 info 字段
- 任何超出 obs[0:24] 范围的数据

## 7. 可用于奖励函数的信号
以下信号可直接作为奖励或惩罚的输入：

- **姿态/稳定信号**：hull_angle（理想为 0），hull_angular_velocity（用作平滑项）
- **前进速度**：horizontal_speed（正值表示前进，可做前进奖励核心）
- **垂直运动**：vertical_speed（接近 0 的稳定状态可给予奖励）
- **关节状态**：各关节角与角速度（可用于步态规范、能耗惩罚的基础）
- **触地信号**：leg_1_ground_contact、leg_2_ground_contact（监测正确支撑模式，避免双足同时离地等）
- **动作量**：action（torque）的 L2 范数或绝对值（作为能量消耗的代理）
- **LIDAR**：lidar_1 … lidar_10（可间接用来鼓励根据地形提前调整姿态，但设计复杂，属于进阶信号）
- **摔倒推断**（derived_possible）：终止时的 hull_angle 或 vertical_speed 异常，可构建摔倒惩罚
- **到达终点推断**（derived_possible）：终止时正常姿态，给予到达奖励

## 8. 不确定或不可用的信号
- 绝对世界坐标（x,y）：环境未提供
- 与终点的距离或进度百分比：无
- 显式 done_reason 或 success flag：info 为空，不可用
- 地形高度图或路径信息：除 10 束 LIDAR 外无全局信息
- 能量总消耗计量：只能通过累积力矩间接计算
- 任何官方奖励函数组件：被屏蔽

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: 单刚体躯干 + 两条腿
  actuator_type: 力矩驱动髋膝（4个独立电机）
  contact_structure: 两腿交替触地（提供二值接触信号）
primary_objectives:
  - 在崎岖地面上尽可能远地前进（或到达尽头）
  - 避免摔倒
secondary_objectives:
  - 最小化关节力矩（能耗）
  - 维持平稳姿态（减少垂直振荡）
main_failure_risks:
  - 身体倾角过大导致摔倒
  - 步态失调，前进缓慢或原地打转
  - 关节过度用力，能量浪费
  - 未能利用 LIDAR 信息导致在障碍物处失稳
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: progress_reward**
  purpose: 鼓励机器人获得正向水平位移，推进时间越长越好
  why_required: 这是任务核心，没有前进奖励策略不会产生持续行走行为
  usable_signals: [horizontal_speed（obs[2]），可以通过时间差分 next_obs[2] - obs[2] 获得加速度，但 best 是直接累积或速度加权；也可以配合触地事件细化每步前进量]
  risks: 过度奖励水平速度可能鼓励机器人用危险姿态猛冲；需要限定速度上限或结合姿态约束

- **role_id: fall_penalty**
  purpose: 惩罚摔倒行为，迫使机器人学会稳定步态
  why_required: 摔倒直接导致 episode 结束，极大降低收益；明确惩罚是保证安全性的关键
  usable_signals: [终止时观测的 hull_angle（obs[0]）、vertical_speed（obs[3]），derived_possible 摔倒推断（角度超出阈值+垂直速度下拉）]
  risks: 惩罚设计不当可能使机器人过于谨慎而不敢迈步；最好只在 episode 结束时施加一次大的负奖励，而非每步都惩罚姿态

### 10.2 条件职责 conditional_roles
- **role_id: energy_efficiency_penalty**
  purpose: 降低关节