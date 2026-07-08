# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
- 控制一个双足身体在不平坦的 2D 地形上向前行走。
- 核心要求：走得尽可能远、尽可能快，同时尽量降低能量消耗。
- 智能体需要协调两条腿的髋、膝关节，产生稳定的双足步态。
- 若身体摔倒，回合立即终止；若成功走到地形末端，回合也会终止。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box (连续浮点数)
- shape: (24,)
- dtype: float32
- 观测向量各索引含义如下：
  - obs[0]: hull_angle —— 身体（躯干）相对于竖直方向的角度
  - obs[1]: hull_angular_velocity —— 身体的角速度
  - obs[2]: horizontal_velocity —— 前向/后向线速度
  - obs[3]: vertical_velocity —— 上下线速度
  - obs[4]: hip1_angle —— 腿1髋关节角度
  - obs[5]: hip1_speed —— 腿1髋关节角速度
  - obs[6]: knee1_angle —— 腿1膝关节角度
  - obs[7]: knee1_speed —— 腿1膝关节角速度
  - obs[8]: leg1_contact —— 腿1是否接触地面（1.0 = 接触，0.0 = 未接触）
  - obs[9]: hip2_angle —— 腿2髋关节角度
  - obs[10]: hip2_speed —— 腿2髋关节角速度
  - obs[11]: knee2_angle —— 腿2膝关节角度
  - obs[12]: knee2_speed —— 腿2膝关节角速度
  - obs[13]: leg2_contact —— 腿2是否接触地面（1.0 = 接触，0.0 = 未接触）
  - obs[14..23]: lidar_0..9 —— 10 个激光雷达测距值，返回前方地形距离

## 4. 动作空间 action_space
- type: Box (连续控制量)，范围 [-1.0, 1.0] 各关节独立
- shape: (4,)
- 动作索引与含义：
  - action 0: hip_torque_leg1 —— 施加在腿1髋关节的力矩
  - action 1: knee_torque_leg1 —— 施加在腿1膝关节的力矩
  - action 2: hip_torque_leg2 —— 施加在腿2髋关节的力矩
  - action 3: knee_torque_leg2 —— 施加在腿2膝关节的力矩

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 到达地形末端（reached_end_of_terrain）—— 可能意味着成功走完全程，但没有显式的成功标志。
- failure-like termination: 身体摔倒（body_fallen_over）—— 典型失败终止。
- ambiguous termination: 无。
- truncation: 未提供任何时间步上限截断信息（返回的 truncated 均为 False，info 为空），任务可能默认依赖环境内部最大步数，但本文档无法确认。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无 success 字段）
- explicit_failure_flag_available: false （info 为空，无 failure 字段）
- allowed_info_fields: 无（info 始终为 {}，没有任何可用字段）
- forbidden_or_uncertain_info_fields:
  - info 的所有字段均不可用（因 info 为空）
  - 无法通过 info 获取 reached_end_of_terrain 或 body_fallen_over 的布尔标志（只能在 terminated 信号中推断，但 terminated 本身不区分成功/失败）
  - 不得假设任何额外的终止原因字符串（如 "termination_reason"）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` : 当前 step 的完整观测数组（24维）
- `action` : 当前 step 执行的动作数组（4维）
- `next_obs` : 下一 step 的观测数组（24维），可用于获得转移后的状态
- `info` : 当前 step 的 info 字典（此处恒为空 {}，不可使用任何字段）

禁止使用：
- `original_reward` —— 官方奖励被严格屏蔽，不得在自定义奖励中使用
- `official_reward` 或任何形式的原始奖励信号
- `info` 中任何未声明的字段（当前 info 为空，因此所有 info 字段均禁止）
- `training_progress` —— 当前任务描述未明确允许使用训练进度，故禁止
- 任何未在“可用于奖励函数的信号”中列出的 obs 切片

## 7. 可用于奖励函数的信号
直接从 `obs` 或 `next_obs` 中可提取的物理含义信号：

- **身体姿态/稳定性**：
  - `hull_angle`（obs[0]）：身体倾斜角度，可用于惩罚偏离直立
  - `hull_angular_velocity`（obs[1]）：身体转动速度

- **线速度**：
  - `horizontal_velocity`（obs[2]）：前进速度，是衡量“走得快”的核心信号
  - `vertical_velocity`（obs[3]）：上下跳动速度

- **关节状态**：
  - 髋关节角度/速度：obs[4], obs[5], obs[9], obs[10]
  - 膝关节角度/速度：obs[6], obs[7], obs[11], obs[12]

- **地面接触**：
  - `leg1_contact`, `leg2_contact`（obs[8], obs[13]）：可用来鼓励双脚交替支撑、惩罚拖地等

- **动作（关节力矩）**：
  - action[0..3] 的绝对值或平方值，可作为能耗惩罚的直接信号

- **激光雷达（地形感知）**：
  - lidar_0..9（obs[14..23]）：前方地形距离信息，可用于评估地形难度或前方是否有障碍

## 8. 不确定或不可用的信号
- **绝对位置/前进距离**：观测中不包含 agent 在世界坐标系下的 x 坐标，无法直接计算走过的累计距离。
- **到达终点标志**：`reached_end_of_terrain` 只在终止时触发，但没有被写入 info，无法在奖励函数中直接获取。
- **摔倒标志**：`body_fallen_over` 同样仅在 `terminated` 中体现，无直接信号。
- **能量消耗测量**：任务没有提供真实的能量消耗测量值，只能通过力矩近似。
- **地形粗糙度/摩擦参数**：未在观测中给出，不可直接使用。
- **时间戳/步数**：`training_progress` 未允许使用，步数信息不可直接作为奖励信号（除非从外部传入，但规范禁止）。



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