# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个两足机器人在不平坦地形上向前行走，尽量走得远（距离）、走得快（速度），同时最小化能耗。机器人需要通过协调双腿的髋关节和膝关节力矩，产生稳定的双足步态。一旦身体倾倒，回合即结束。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (24,)
- dtype: float32（通常）
- obs[0]: hull_angle – 躯干相对竖直方向的倾角
- obs[1]: hull_angular_velocity – 躯干角速度
- obs[2]: horizontal_velocity – 质心水平（前向）速度
- obs[3]: vertical_velocity – 质心垂直速度
- obs[4]: hip1_angle – 第一腿髋关节角度
- obs[5]: hip1_speed – 第一腿髋关节角速度
- obs[6]: knee1_angle – 第一腿膝关节角度
- obs[7]: knee1_speed – 第一腿膝关节角速度
- obs[8]: leg1_contact – 第一腿是否触地（1.0 触地，0.0 不触地）
- obs[9]: hip2_angle – 第二腿髋关节角度
- obs[10]: hip2_speed – 第二腿髋关节角速度
- obs[11]: knee2_angle – 第二腿膝关节角度
- obs[12]: knee2_speed – 第二腿膝关节角速度
- obs[13]: leg2_contact – 第二腿是否触地（1.0 触地，0.0 不触地）
- obs[14] ~ obs[23]: lidar_0 ~ lidar_9 – 10个激光雷达测距值，测量前方地形距离

## 4. 动作空间 action_space
- type: Box (连续)
- 动作数量: 4 维
- 每维范围: [-1.0, 1.0]（连续力矩值）
- action 0: hip_torque_leg1 – 第一腿髋关节施加的力矩
- action 1: knee_torque_leg1 – 第一腿膝关节施加的力矩
- action 2: hip_torque_leg2 – 第二腿髋关节施加的力矩
- action 3: knee_torque_leg2 – 第二腿膝关节施加的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形终点，完成行走任务）
- failure-like termination: body_fallen_over（身体倾倒，行走失败）
- ambiguous termination: 无
- truncation: 未在给定代码中体现（可能存在最大步数截断，但未明确提供）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，terminated 仅由布尔值给出，无显式成功/失败标志字段）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；无法确定成功/失败原因细分

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前时刻的完整观察（24 维数组）
- action：当前时刻采取的动作（4 维数组）
- next_obs：下一时刻的观察（24 维数组）
- info：仅允许空字典（暂无内容可用）
- training_progress：仅当 prompt 明确允许时可用（此处未提及，默认不使用）

禁止使用：
- original_reward（官方奖励已掩盖，禁止直接或间接使用）
- official_reward
- 任何未在 observation_space 中声明的 obs 切片
- 任何未明确允许的 info 字段

## 7. 可用于奖励函数的信号
- 躯干倾角：obs[0] (hull_angle) – 可用于衡量平衡，偏离直立越大可能越差
- 躯干角速度：obs[1] (hull_angular_velocity) – 联合倾角判断稳定性
- 水平速度：obs[2] (horizontal_velocity) – 前进速度，鼓励快速向前
- 垂直速度：obs[3] (vertical_velocity) – 可用于惩罚剧烈上下跳动
- 关节角度/角速度：obs[4]~obs[7] 和 obs[9]~obs[12] – 可用于能量惩罚（例如动作平方和、关节力矩做功近似）
- 触地标志：obs[8], obs[13] – 可用于判断步态交替
- 动作本身：action (4 维) – 可直接计算控制能量（如 sum(action^2)）
- 下一时刻的水平速度：next_obs[2] – 可用于衡量速度变化
- LIDAR 测距：obs[14]~obs[23] – 可用于鼓励避开近距离障碍、保持安全距离
- 终止相关推断：既无显式成功/失败标志，也不应直接依赖终止状态（因 reward 在终止前步计算）

## 8. 不确定或不可用的信号
- 显式成功/失败标志：info 为空，无法获取
- 官方奖励：original_reward 已被屏蔽，不可用
- 地形进度百分比或剩余距离：obs 中无该字段（除非通过 LIDAR 间接推断，但不可靠）
- 能量消耗的真实物理测量：只能通过动作或关节力矩近似，无法直接获取
- 接触力大小：仅提供二值触地标志，无接触力或地面反力



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