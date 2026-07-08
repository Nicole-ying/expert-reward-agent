# environment_card.md

# Env_003 环境理解卡片

## 1. 任务目标
在一个水平的直线轨道上，通过向移动底座施加固定大小的水平力（离散两个方向），控制底座移动，使底座顶端通过无动力关节连接的摆杆尽可能长时间地保持直立（即杆偏离竖直的角度不超过阈值），同时底座本身不能超出轨道边界。**本质是一个生存平衡任务：最大化存活步数（或时间）。**

## 2. 任务类型选择
selected_route_id: survival_balance_task
confidence: high
reason: 任务要求持续维持系统在不稳定平衡点附近，没有预设目标位置或导航点，终止仅由平衡状态破坏或轨道越界触发，奖励稀疏且与存活时长正相关，属于典型的生存/平衡任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (4,)
- dtype: float32
- 各维度含义（按索引）：
  - obs[0]: 底座水平位置 (base_position)，取值范围约 [-4.8, 4.8]，但有效安全范围实际为 [-2.4, 2.4]，超出则终止。
  - obs[1]: 底座水平速度 (base_velocity)，无界。
  - obs[2]: 杆相对于竖直方向的偏角 (pole_angle)，单位弧度，范围约 [-0.4189, 0.4189]；终止阈值为 [-0.20944, 0.20944]。
  - obs[3]: 杆角速度 (pole_angular_velocity)，无界。

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 2
- 动作含义：
  - action 0: 向轨道负方向施加固定大小的水平力 (push_negative_direction)
  - action 1: 向轨道正方向施加固定大小的水平力 (push_positive_direction)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无**。环境不定义“成功”概念，存活得越久越好。
- failure-like termination:
  - 底座位置绝对值超过 2.4（即底座掉出轨道有效区域）。
  - 杆偏转角绝对值超过 0.20943951 弧度（约 12°）。
  - 以上任一发生均视为**平衡失败**，episode 终止。
- ambiguous termination: 无。
- truncation: episode 达到 500 步后强制截断，属于时间截断，不代表任务失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: info 为空字典 `{}`，无任何额外字段。
- forbidden_or_uncertain_info_fields: 所有在 info 中未声明的字段（例如 success、failure、termination_reason、reward 等）均不存在，禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前步观察
- `action`：当前步执行的动作
- `next_obs`：下一步观察
- `info`：空字典，但接口允许传入（实际无有效字段）
- `training_progress`：**默认禁止**，除非任务 prompt 明确允许。本项目 prompt 未提及可以使用，故禁止使用。

禁止使用：
- `original_reward`：官方奖励被隐藏，不可用。
- 任何未在 info 或 obs 中声明的字段。
- 任何未在观察空间中明确列出的 obs 切片。

## 7. 可用于奖励函数的信号
- **底座位置**：`obs[0]` / `next_obs[0]`，可用于惩罚远离中心。
- **底座速度**：`obs[1]` / `next_obs[1]`，可用于平滑控制。
- **杆偏角**：`obs[2]` / `next_obs[2]`，核心平衡信号。
- **杆角速度**：`obs[3]` / `next_obs[3]`，可用于倾向稳定性。
- **动作**：`action`，可用于鼓励能量最小或对称控制。

## 8. 不确定或不可用的信号
- **官方原始奖励**：被显式隐藏（masked），不可用。
- **显式成功/失败标志**：不存在。
- **存活步数/时间**：未直接提供在观察或 info 中，无法直接获取。
- **底座所处区域精细状态**（如是否接近边界）：只能通过底座位置计算，但位置是可用信号。
- **任何 info 中的额外奖励或终止原因**：info 为空，不存在。



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。

## 1. 任务路由摘要
- survival_balance_task：按该任务类型选择信号，并先检查接口可用性。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### progress_delta_reward
- 角色: 密集学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 使用说明: 奖励每一步更接近目标。适合目标位置已知的导航/到达任务。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 使用说明: 连续负距离信号，引导 agent 靠近目标。与 progress_delta_reward 同时大权重使用会产生重复信号。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 使用说明: 基于势能差分的塑形信号。比 progress_delta 更抽象，当任务有明确的势能定义时可使用。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 使用说明: 多条件组合的完成近似。不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 使用说明: 当环境提供显式 success flag 时可用。若 explicit_success_flag_available=false，不可使用。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 使用说明: 惩罚每步耗时。先完成任务再加入，不建议 v1 使用。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 使用说明: 按条件切换奖励模式。v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 从上述任务路由推荐的骨架中选择，优先选择所需信号在环境接口中可用的骨架。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。