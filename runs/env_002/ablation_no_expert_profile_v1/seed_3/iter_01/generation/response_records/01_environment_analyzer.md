# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一项平面双足行走任务。主体目标：控制两条腿的髋关节和膝关节，在起伏地形上尽可能远、尽可能快地向前行走，同时尽量降低能量消耗。次要目标：维持主躯干直立（防止倾倒），并利用关节协调产生稳定的步伐。不应混淆的目标：仅追求速度而忽略能耗，或仅追求平衡但不前进。

## 2. 任务类型选择
**selected_route_id:** locomotion_continuous_control  
**confidence:** high  
**reason:** 核心任务是持续向前移动并通过不平坦地形，速度与能耗均为附属优化要求，并非多个权重相当且冲突的核心目标。此外，主躯干平衡是生存前提，但不是独立的核心目标。因此典型的 locomotion_continuous_control 类别，对应动力学形态为平面双足步态。

**dynamics_subtype:** planar_bipedal_gait

## 3. 观察空间 observation_space
- **type:** Box
- **shape:** [24]
- **dtype:** float (根据描述推断为 float32)
- **obs 索引含义与奖励可用性：**

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | hull_angle | 主躯干相对于竖直方向的倾斜角 | true |
| 1 | hull_angular_velocity | 主躯干绕质心的角速度 | true |
| 2 | horizontal_velocity | 前进/后退方向的线速度 | true |
| 3 | vertical_velocity | 上下方向的线速度 | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1触地标志 (1.0 触地, 0.0 未触地) | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
| 10 | hip2_speed | 腿2髋关节角速度 | true |
| 11 | knee2_angle | 腿2膝关节角度 | true |
| 12 | knee2_speed | 腿2膝关节角速度 | true |
| 13 | leg2_contact | 腿2触地标志 | true |
| 14..23 | lidar_0..9 | 前方地形激光雷达距离测量值（10个） | true (可用于地形感知，但不可直接作为奖励) |

所有观察分量均可被奖励函数使用，但请注意，某些分量（如 LIDAR）不应直接作为优化目标，因为它们反映地形信息，可用于辅助约束或指引。

## 4. 动作空间 action_space
- **type:** Box (continuous)
- **shape:** [4]
- **动作解释：**

| 维度 | 名称 | 含义 |
|------|------|------|
| 0 | hip_torque_leg1 | 施加于腿1髋关节的扭矩，范围[-1, 1] |
| 1 | knee_torque_leg1 | 施加于腿1膝关节的扭矩，范围[-1, 1] |
| 2 | hip_torque_leg2 | 施加于腿2髋关节的扭矩，范围[-1, 1] |
| 3 | knee_torque_leg2 | 施加于腿2膝关节的扭矩，范围[-1, 1] |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:** `reached_end_of_terrain` （到达地形终点，视为成功完成行走任务）
- **failure-like termination:** `body_fallen_over` （身体倾倒，视为失败）
- **ambiguous termination:** 无
- **truncation:** 无额外说明，但通常可能通过最大时间步限制，此处未提及即不关心

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false （没有明显的 `info["success"]` 或类似真值，但可通过 `reached_end_of_terrain` 终止条件间接推断成功）
- **explicit_failure_flag_available:** false （同样没有 `info["failure"]`，仅通过 `body_fallen_over` 终止条件间接推断）
- **allowed_info_fields:** 无。从代码片段看，`step` 返回 `info = {}` ，因此不允许使用任何 info 字段。
- **forbidden_or_uncertain_info_fields:** 任何 info 字段均被视为不存在，禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
**允许使用：**
- `obs` (完整的24维观察向量)
- `action` (4维扭矩动作向量)
- `next_obs` (下一时刻的24维观察向量，但需注意在奖励计算中通常不直接提供，只能通过函数签名获得，而实际接口可能不会传入；若提供了则可使用)
- `info` 中明确允许的字段：无（根据终止分析，info 为空，故任何字段均不可用）
- `training_progress` 仅在 prompt 明确允许时使用，这里未明确允许，默认为不用。
- **禁止使用：**
- `original_reward` （即被隐藏的官方奖励，严格禁止使用）
- 任何未声明的 `info` 字段或从观察中推断但未列出的变量
- 任何形式的官方奖励重构或近似

## 7. 可用于奖励函数的信号
可根据观测向量直接获得或间接计算的可用信号：
- **position / displacement:**
  - 无直接的全局位置，但可利用水平速度积分，或通过 LIDAR 推断大致前进距离。实际实现中不可直接获得绝对位移，除非环境暴露（此处没有）。故只能依赖速度等动态量。
- **velocity:**
  - forward velocity: `obs[2]` (horizontal_velocity)，可直接用于鼓励前进。
  - vertical velocity: `obs[3]`，可用于惩罚不必要的弹跳或坠落。
- **orientation:**
  - hull_angle: `obs[0]`，用于保持直立。
  - hull_angular_velocity: `obs[1]`，用于姿态稳定性。
- **contact:**
  - leg1_contact: `obs[8]`
  - leg2_contact: `obs[13]`
    可用于检测步态支撑情况，鼓励轮流触地或避免双足同时离地。
- **action/energy:**
  - 四维扭矩的动作向量 `action`，可通过平方和等衡量能量消耗，从而惩罚高能耗动作。
- **other:**
  - 所有关节角度和角速度可用于姿态平滑或限制关节活动范围惩罚。
  - LIDAR 读数（obs[14..23]）可用于感知前方地形，但仅作为辅助信息，不应直接充当奖励项。

## 8. 不确定或不可用的信号
- **全局位置/距离：** 观测中无(x,y)坐标，无法获得已行走的实际距离，因此不能直接奖励位移进步，只能通过速度 surrogate 实现。
- **能量/功率真实值：** 虽有扭矩指令，但无电机功率、电流等信息，只能近似用扭矩平方和表示能耗。
- **地形坡度/高度：** 只提供激光雷达距离，未直接给出高度图或坡度，可能需要进一步处理（如差分）来推断地形复杂度，这属于不确定信号，不宜纳入基础奖励。
- **成功/失败显式标志：** info 为空，所以没有直接的成功/失败真值作为奖励系数，必须通过终止条件推断，但终止标志在 `compute_reward` 函数中不可获取，因此不能用于奖励塑造。
- **步态周期/相位：** 观察中无时间信息或步态相位变量，无法直接给出步态规律奖励。
- **foot clearance / swing height：** 缺乏脚部高度或离地间隙测量，不能直接奖励抬脚高度等。
