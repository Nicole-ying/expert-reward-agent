# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个两足机器人（两条腿，每条腿有髋关节和膝关节）在不平坦的 2D 地形上稳定行走。  
主要目标是尽可能远、尽可能快地向前移动；次要目标是尽量降低能量消耗（动作力度）。  
机器人需要协调双腿的关节力矩，生成稳定的步态。一旦身体倒下，回合立即结束。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box  
- shape: [24]  
- dtype: float32（假设，实际未见明确定义，通常连续观测为 float）  
- 各维度含义：  
  - obs[0]: hull_angle —— 身体（主躯干）相对于竖直方向的倾斜角  
  - obs[1]: hull_angular_velocity —— 身体角速度  
  - obs[2]: horizontal_velocity —— 前向/后向水平线速度  
  - obs[3]: vertical_velocity —— 上下线速度  
  - obs[4]: hip1_angle —— 腿 1 髋关节角度  
  - obs[5]: hip1_speed —— 腿 1 髋关节角速度  
  - obs[6]: knee1_angle —— 腿 1 膝关节角度  
  - obs[7]: knee1_speed —— 腿 1 膝关节角速度  
  - obs[8]: leg1_contact —— 腿 1 接地标志（1.0=接触地面，0.0=未接触）  
  - obs[9]: hip2_angle —— 腿 2 髋关节角度  
  - obs[10]: hip2_speed —— 腿 2 髋关节角速度  
  - obs[11]: knee2_angle —— 腿 2 膝关节角度  
  - obs[12]: knee2_speed —— 腿 2 膝关节角速度  
  - obs[13]: leg2_contact —— 腿 2 接地标志（1.0=接触地面，0.0=未接触）  
  - obs[14] ~ obs[23]: lidar_0 ~ lidar_9 —— 10 个前方地形激光雷达测距值（前方不同角度的距离）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 连续动作，每个分量范围 [-1.0, 1.0]
- action 0: hip_torque_leg1 —— 施加于腿 1 髋关节的力矩（扭矩）
- action 1: knee_torque_leg1 —— 施加于腿 1 膝关节的力矩
- action 2: hip_torque_leg2 —— 施加于腿 2 髋关节的力矩
- action 3: knee_torque_leg2 —— 施加于腿 2 膝关节的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain` —— 机器人到达地形的终点（任务目标完成）
- failure-like termination: `body_fallen_over` —— 身体倒下（任务失败）
- ambiguous termination: 无（从 step 源码看，terminated 只由这两个条件构成，不存在其他模糊情况）
- truncation: step 返回的第四个元素为 `False`，表示不存在时间截断，所有终止都是由环境状态触发的真实终止

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  （`info` 字典为空，没有显式成功标志；只能从终止时是否为 `reached_end_of_terrain` 推断）
- explicit_failure_flag_available: false  
  （同上，需要根据是否为 `body_fallen_over` 推断）
- allowed_info_fields: 无（`info` 为空，下游 reward 函数不允许依赖任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，也不可假设存在官方 success/failure 标记

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- `obs` —— 当前步的完整观测 (np.ndarray shape=(24,))
- `action` —— 当前步执行的动作 (np.ndarray shape=(4,))
- `next_obs` —— 下一步的完整观测
- `training_progress` —— 仅当 prompt 明确允许时才使用（本任务描述中没有提到，一般不应使用）

**禁止使用：**
- `original_reward` —— 官方奖励已遮蔽，不得直接或间接使用
- `official_reward` 或任何等效奖励信号
- `info` 中任何字段（当前为空，即使非空也未授权）
- 未在观测空间中明确解释的 `obs` 切片（所有 24 维均已解释，故可使用全部）
- 任何全局状态、环境内部变量

## 7. 可用于奖励函数的信号
从当前观测 `obs`、动作 `action` 和下一步观测 `next_obs` 中可以直接提取以下信号（需自行计算）：

- **前进速度**  
  - `obs[2]` / `next_obs[2]`：水平速度（正值前进，负值后退），可直接鼓励向前移动。
- **身体姿态稳定性**  
  - `obs[0]` / `next_obs[0]`：身体倾角，理想应接近 0（直立）；可惩罚大倾角。
  - `obs[1]`：身体角速度，鼓励小的角速度变化。
- **能量消耗（动作幅度）**  
  - `action` 的平方和或绝对值之和：反映当前控制力矩的大小，可用来惩罚过大动作以降低能耗。
- **关节状态与步态**  
  - `obs[4] ~ obs[7]`（腿 1 髋、膝角度/速度）、`obs[9] ~ obs[12]`（腿 2 对应值）：可设计周期性、摆动/支撑相位奖励。
  - `obs[8]` / `obs[13]`：接地标志，可用于检测步态相位（支撑相、摆动相），鼓励交替接地。
- **地形感知（激光雷达）**  
  - `obs[14:24]`：前方地形距离信息，可用于惩罚与障碍物过近的行为，但通常不作为主要前进奖励的直接来源。
- **终止时的结果推断**  
  - 虽然 `info` 为空，但可在训练循环中结合 `terminated` 标志与 `next_obs` 的状态判断：若水平速度接近零且身体倾角超过某阈值，可视为倒下；若 `terminated` 且身体直立、水平位置到达终点附近（需从环境外部获得位置信息，但此处观测中无位置，故无法直接获得终点信息），因此纯从观测难以区分成功终止与失败终止。设计奖励时需注意不要依赖终点识别。

## 8. 不确定或不可用的信号
- **水平位置 / 前进距离**：观测空间中没有 x 坐标或里程计，因此无法直接奖励“走过的距离”。只能通过水平速度间接鼓励前进。
- **能量消耗的官方度量**：未提供。
- **显式的成功/失败标志**：`info` 为空，不可用。
- **reached_end_of_terrain 的直接检测**：在单步观测中无法得知是否到达终点，只有终止时才能从训练循环得知终止原因，但在 `compute_reward` 函数内部 `terminated` 和 `truncated` 信号未作为参数传入，因此不能可靠使用。
- **身体绝对高度或是否倒下**：观测中有身体倾角，没有绝对高度，但可通过倾角和角速度推断倒下。
- **外部地形信息**：仅有前方 10 个激光点，没有完整地形高度图，不能获取身后或完成的地形信息。



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