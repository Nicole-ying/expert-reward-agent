# 匿名环境理解卡片

## 1. 任务目标
训练一个双足机器人通过布满梯子、树桩、坑洼和不平地形的复杂地面，目标是尽可能远且高效地前进，最终到达地形另一端。不允许摔倒，同时希望最小化不必要的关节力矩以实现节能。核心是稳健前进，附属目标是抑制摔倒和降低功耗。注意：任务描述中“到达尽头、避免摔倒、最小化力矩”三者都可取，但主次关系明确——前进到达尽头是最高目标，生存（不摔倒）是必要条件，力矩最小化属于锦上添花的次生需求。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是驱动双腿持续前进通过非结构化地形，没有固定位点导航需求，也没有多目标冲突；摔倒仅作为失败终止条件，不属于平衡生存任务族。附属的能量优化不影响主族分类。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32
- obs[0]: hull_angle – 身体倾角，reward_usable: true（可用于检测摔倒或大扰动）
- obs[1]: hull_angular_velocity – 身体角速度，reward_usable: true（惩罚急剧旋转）
- obs[2]: horizontal_speed – 水平速度（前进方向），reward_usable: true（直接作为前进主奖励）
- obs[3]: vertical_speed – 垂直速度，reward_usable: true（惩罚异常跳动或坠落）
- obs[4]: joint_0_angle (髋关节1角度)，reward_usable: true（用于姿态约束）
- obs[5]: joint_0_speed (髋关节1角速度)，reward_usable: true（平滑项）
- obs[6]: joint_1_angle (膝关节1角度)，reward_usable: true
- obs[7]: joint_1_speed (膝关节1角速度)，reward_usable: true
- obs[8]: joint_2_angle (髋关节2角度)，reward_usable: true
- obs[9]: joint_2_speed (髋关节2角速度)，reward_usable: true
- obs[10]: joint_3_angle (膝关节2角度)，reward_usable: true
- obs[11]: joint_3_speed (膝关节2角速度)，reward_usable: true
- obs[12]: leg_1_ground_contact (0/1)，reward_usable: true（用于步态模式识别）
- obs[13]: leg_2_ground_contact (0/1)，reward_usable: true
- obs[14~23]: lidar_1~lidar_10 – 前方地形激光测距值，reward_usable: true（可通过差分检测障碍冲击或预测危险，但不建议直接用作奖励信号）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- action_dim 0: hip_1_torque – 髋关节1力矩
- action_dim 1: knee_1_torque – 膝关节1力矩
- action_dim 2: hip_2_torque – 髋关节2力矩
- action_dim 3: knee_2_torque – 膝关节2力矩
四个关节均独立力矩控制，连续动作空间。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 到达地形尽头（reached_end_of_terrain），导致episode终止。
- failure-like termination: 身体摔倒（body_fallen_over），导致episode终止。
- ambiguous termination: 无。所有终止情况必为上述之一。
- truncation: step 返回 truncated=False，不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（未在info中提供）
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 始终为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，不允许在奖励函数中依赖 info。

尽管info无可信标签，但可通过观测信号间接推断终止原因：  
- 摔倒推断：hull_angle 超过阈值（如 >1.0 rad）、身体垂直速度突变向下、或两个腿部接触信号同时长时间为0（失去立足）等组合信号可作为 derived_possible 摔倒信号。  
- 到达终点推断：agent 水平速度持续非零，episode 突然终止且无明显摔倒迹象（hull_angle 正常，垂直速度平稳），此逻辑可用于判断成功，但只能在 episode 结束时进行，奖励函数可在观察到终止时用 next_obs 判断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs (shape [24])
- action (shape [4])
- next_obs (shape [24])
- info 中无可用字段，因 info 为空，实际禁止使用 info
- training_progress (float 0~1)：仅当 prompt 明确允许时才使用，此处可保留为可选参数但建议默认不用

禁止使用：
- original_reward（即官方奖励，已被屏蔽）
- official_reward 及其他未声明的环境返回量
- 未声明的 info 字段（info 为空）
- 未声明的 obs 切片索引（所有字段均已声明，无法外推位置、绝对坐标等）

## 7. 可用于奖励函数的信号
以下信号可直接或间接用于奖励设计：
- 前进速度：obs[2] (horizontal_speed) 可在每一步提供连续正向激励。
- 身体姿态/稳定：obs[0] (hull_angle) 可惩罚大倾角；obs[1] (hull_angular_velocity) 可惩罚快速旋转。
- 垂直方向异常：obs[3] (vertical_speed) 可惩罚异常跳动（绝对值过大）。
- 关节平滑与能量：action 本身（力矩）可用于二次惩罚（\|action\|²），也可对相邻步的动作差施加惩罚。
- 接触信号：obs[12], obs[13] 可用于生成优雅离地、着地模式，或提供 foot-air-time 奖励（derived_possible）。
- 雷达测距：obs[14:23] 可用于检测极度近距离（即将碰撞）提供的警示信号，但不建议直接用作奖励，可作为惩罚条件。
- 终止推断信号：从 next_obs 中提取 hull_angle、vertical_speed、contact 的组合，以识别摔倒或成功（derived_possible），限用于 episode 结束时的特殊奖励/惩罚。

## 8