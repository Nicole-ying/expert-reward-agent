# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个二维双足步行任务。
智能体需要控制两条腿的髋关节和膝关节，在崎岖不平的地形上**尽可能远、尽可能快地稳定行走**，同时尽量**减少能量消耗**。  
身体一旦**摔倒**，回合立即终止；若**到达地形终点**，也视为终止。  
核心要求是产生稳健的双足步态，保持前进。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推测连续值）
- obs[0]: hull_angle — 身体相对于竖直方向的角度 (rad)
- obs[1]: hull_angular_velocity — 身体角速度 (rad/s)
- obs[2]: horizontal_velocity — 水平线速度（前进方向） (m/s)
- obs[3]: vertical_velocity — 竖直线速度 (m/s)
- obs[4]: hip1_angle — 腿1髋关节角度 (rad)
- obs[5]: hip1_speed — 腿1髋关节角速度 (rad/s)
- obs[6]: knee1_angle — 腿1膝关节角度 (rad)
- obs[7]: knee1_speed — 腿1膝关节角速度 (rad/s)
- obs[8]: leg1_contact — 腿1触地标志 (1.0 接触, 0.0 不接触)
- obs[9]: hip2_angle — 腿2髋关节角度 (rad)
- obs[10]: hip2_speed — 腿2髋关节角速度 (rad/s)
- obs[11]: knee2_angle — 腿2膝关节角度 (rad)
- obs[12]: knee2_speed — 腿2膝关节角速度 (rad/s)
- obs[13]: leg2_contact — 腿2触地标志 (1.0 接触, 0.0 不接触)
- obs[14..23]: lidar_0..lidar_9 — 10 个激光测距值，测量前方地形距离 (m)

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: [4]
- bounds: [-1.0, 1.0] 每个关节
- action 0: hip_torque_leg1 — 腿1髋关节力矩
- action 1: knee_torque_leg1 — 腿1膝关节力矩
- action 2: hip_torque_leg2 — 腿2髋关节力矩
- action 3: knee_torque_leg2 — 腿2膝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain`（到达地形终点，可视为成功完成路线）
- failure-like termination: `body_fallen_over`（身体摔倒，失败）
- ambiguous termination: 无
- truncation: 未提及超时截断，推测由 `terminated` 直接控制，无单独 `truncation`

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何其他 info 字段（包括从未声明的 success、failure、termination_reason 等）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前观察（24维数组）
- `action`：当前动作（4维数组）
- `next_obs`：下一观察（24维数组）
- `info` 中明确允许的字段：无（info 为空，不能使用任何键）
- `training_progress`：**不允许使用**（任务描述未提及）

禁止使用：
- `original_reward`（官方奖励被屏蔽）
- 任何其他未声明的 `info` 字段
- 未在观察空间中声明的 obs 切片维度

## 7. 可用于奖励函数的信号
从 `obs` 或 `next_obs` 中可以提取以下信息构造自定义奖励：
- **水平速度**：`obs[2]` / `next_obs[2]`，用于鼓励快速前进。
- **身体倾斜（稳定性）**：`obs[0]`（角度）、`obs[1]`（角速度），用于处罚偏离直立。
- **垂直速度**：`obs[3]`，可能用于处罚过大的上下跳动。
- **能量消耗**：通过 `action` 的平方和或绝对值之和近似力矩能耗，鼓励动作平滑与节能。
- **接触模式**：`obs[8]`、`obs[13]` 触地标志，可用于塑造步态（例如处罚双脚同时离地或同时着地）。
- **关节状态**：`obs[4:8]` 和 `obs[9:13]` 的角度、角速度，用于限制关节极限或鼓励特定姿态。
- **前进距离**：通过对水平速度积分或在自定义代码中累加（需跨时间步累加，不能仅从 obs 直接得到绝对位置，但奖励设计时不能用“官方位置”之外的信息）。

## 8. 不确定或不可用的信号
- **绝对位置坐标**：观察空间内无 x/y 坐标，无法直接获取已行走距离或剩余距离。
- **地形终点距离**：激光雷达可能隐含前方信息，但没有明确的“剩余距离”字段。
- **成功/失败标志**：`info` 中无任何字段，不能基于终止原因奖励。
- **能耗真实值**：环境中没有提供能量消耗的测量值，只能用动作大小作为代理。
- **地形类型或难度**：完全匿名，未给出任何地形参数。
- **训练进度**：`training_progress` 不可用，因为任务描述未允许使用它。



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- locomotion_continuous_control：任务目标是稳定前进通过地形。重点观察 fast_then_fail / hover_or_stand_still / over_conservative_policy。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### forward_progress_reward
- 角色: 前进方向引导
- 数学形态: lambda_fwd * forward_velocity
- 需要信号: forward velocity component
- 使用说明: 奖励沿前进方向的速度。适合 locomotion 类任务。
- 风险: 快速前进但容易摔倒。
- 后续迭代: 若 fast_then_fail，配合稳定性约束。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### alive_bonus
- 角色: 存活激励
- 数学形态: lambda_alive * I[not_done]
- 需要信号: done flag
- 使用说明: 每步给予小额存活奖励，鼓励 agent 不提前终止。适合 locomotion/balance 类任务。
- 风险: hover_or_stand_still（原地不动来获取存活奖励）。
- 后续迭代: 若 agent 不动，减小权重或配合前向奖励。

### action_smoothness_penalty
- 角色: 动作平滑约束
- 数学形态: -lambda_smooth * |action - previous_action|
- 需要信号: previous action or action history
- 使用说明: 惩罚动作的剧烈变化。适合连续控制任务。
- 风险: 离散动作空间不可用（无历史信息）。
- 后续迭代: 若动作抖动，增大权重。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。