# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个连续控制运动任务。  
一个3D四足机器人，拥有四条腿和8个扭矩控制关节，需要在保持身体高度在健康范围（0.2~1.0）的前提下，持续稳定地向前行走或奔跑。  
主目标是**产生稳定向前的运动，并维持身体高度与直立姿态**；次目标是**在稳定性的基础上尽可能提升前向速度**。不应将任务简化为单纯的速度最大化，也不应将“仅保持直立”当作成功（机器人必须在健康姿态下持续前进）。

## 2. 任务类型选择
**selected_route_id**: `locomotion_continuous_control`  
**confidence**: high  
**reason**: 任务核心是“通过持续运动向前穿越地形”，没有指定的目标位置，也不存在稀疏探索或抓取操作。仅有稳定性作为前提，速度作为附属优化，但仍属于典型的前向运动控制族。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (27,)
- **dtype**: float32（假设连续值）
- **obs各维度含义与可用性**：

| 索引 | 名称 | 含义 | reward_usable | 备注 |
|---|---|---|---|---|
| 0 | body_z | 本体垂直高度 | true | 用于健康/存活门控 |
| 1 | quat_w | 本体四元数 w | true | 与 2-4 联合计算倾斜 |
| 2 | quat_x | 本体四元数 x | true | body_up_z = 1 - 2*(x² + y²) |
| 3 | quat_y | 本体四元数 y | true | |
| 4 | quat_z | 本体四元数 z | true | |
| 5 | joint_1_angle | 髋关节1角度 | true | 可配合 velocity 做动作质量评估 |
| 6 | joint_2_angle | 踝关节1角度 | true | |
| 7 | joint_3_angle | 髋关节2角度 | true | |
| 8 | joint_4_angle | 踝关节2角度 | true | |
| 9 | joint_5_angle | 髋关节3角度 | true | |
| 10 | joint_6_angle | 踝关节3角度 | true | |
| 11 | joint_7_angle | 髋关节4角度 | true | |
| 12 | joint_8_angle | 踝关节4角度 | true | |
| 13 | body_x_velocity | 世界系 x 方向线速度（前向） | true | 主试仪用信号 |
| 14 | body_y_velocity | 侧向线速度 | true | 可用于惩罚侧漂 |
| 15 | body_z_velocity | 垂直线速度 | true | 可选垂直运动惩罚 |
| 16 | body_roll_velocity | 绕 x 轴角速度 | true | 辅助稳定性评估 |
| 17 | body_pitch_velocity | 绕 y 轴角速度 | true | 同上 |
| 18 | body_yaw_velocity | 绕 z 轴角速度 | true | 如不需要转向可忽略 |
| 19‑26 | joint_*_velocity | 8个关节角速度 | true | 可用于加速度/能耗惩罚 |

## 4. 动作空间 action_space
- **type**: Box
- **shape**: (8,)
- **每个维度**：连续扭矩，范围 [-1.0, 1.0]
- **动作含义**：

| 动作维度 | 名称 | 含义 |
|------------|------|----------|
| 0 | hip_1_torque | 第一髋关节扭矩 |
| 1 | ankle_1_torque | 第一踝关节扭矩 |
| 2 | hip_2_torque | 第二髋关节扭矩 |
| 3 | ankle_2_torque | 第二踝关节扭矩 |
| 4 | hip_3_torque | 第三髋关节扭矩 |
| 5 | ankle_3_torque | 第三踝关节扭矩 |
| 6 | hip_4_torque | 第四髋关节扭矩 |
| 7 | ankle_4_torque | 第四踝关节扭矩 |

## 5. step 与终止条件分析

### 5.1 终止模式
- **failure‑like termination**:
  1. `body_height_outside_healthy_range`：本体高度 z ≤ 0.2 或 ≥ 1.0，对应**摔倒/倒塌**或**跳跃过高/翻倒**。
  2. `state_value_outside_finite_range`：任意状态值为 NaN 或 inf，属于**数值崩溃**。
- **ambiguous termination**: 无（以上均为明显失败性终止）。
- **truncation**: `time_limit_reached`，表示**完成整个 episode**，可视为成功存活并运动至时间上限。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: []（info 字典为空）  
- **forbidden_or_uncertain_info_fields**: 所有 info 字段均被禁止，尤其禁止使用官方奖励项、x/y 位置、距离等。  

*但失败终止可通过对 `next_obs[0]` 的瞬时值推断：若 `body_z` 不在 (0.2, 1.0) 内，可认为摔倒/翻倒，记为 derived_possible 信号。*

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

**允许使用**：
- `obs`
