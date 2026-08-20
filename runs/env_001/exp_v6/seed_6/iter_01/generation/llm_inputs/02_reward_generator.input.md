# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器（类似于月球着陆器）轨迹优化问题。飞行器从视口顶部中央附近以随机初始受力开始运动。  
智能体需要**尽快降落到中央的目标平台上并稳定着陆**，同时**尽量减少发动机推力的使用**。  
期望行为：接近目标、降低速度、保持姿态稳定、安全接触目标平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是控制飞行器到达并稳定在目标垫上，属于典型的导航‑目标到达问题（带能耗约束）。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（假定默认）
- obs[0]: `x_position` — 飞行器相对于目标平台的水平坐标
- obs[1]: `y_position` — 飞行器相对于平台高度的垂直坐标
- obs[2]: `x_velocity` — 水平线速度
- obs[3]: `y_velocity` — 垂直线速度
- obs[4]: `body_angle` — 飞行器姿态角
- obs[5]: `angular_velocity` — 角速度
- obs[6]: `left_support_contact` — 左侧支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: `right_support_contact` — 右侧支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` — 无推力，仅靠惯性运动
- action 1: `left_orientation_engine` — 点燃左姿态调整发动机（改变角速度）
- action 2: `main_engine` — 点燃主发动机（产生竖直向上的推力）
- action 3: `right_orientation_engine` — 点燃右姿态调整发动机（相反方向改变角速度）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` — 飞行器不再活跃或已稳定（可能表示成功着陆并稳定在目标平台）
- failure-like termination:  
  `crash_or_body_contact` — 坠毁或与非目标表面的身体接触  
  `horizontal_position_outside_viewport` — 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 环境源码未出现时间截断，但实际 gym 封装可能有默认时间限制（未提供细节，故在 reward 设计中不应依赖）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 中无 `success` 字段）
- explicit_failure_flag_available: false （info 中无 `failure` 字段）
- allowed_info_fields: 无（info 为空字典 `{}`，不会提供可用的额外信号）
- forbidden_or_uncertain_info_fields: `success`, `failure`, `termination_reason` 等任何未在源码中出现的字段

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` — 执行动作前的观察（8 维数组）
- `action` — 执行的离散动作编号（0,1,2,3）
- `next_obs` — 动作执行后的观察（8 维数组）
- `info` 中明确允许的字段：无（info 恒为空，不可用）

禁止使用：
- `original_reward` / `official_reward` — 已被掩蔽，不可重建或直接使用
- `training_progress` — 任务描述未明确允许使用，禁止
- 任何未在“允许使用”中列出的 `info` 或 `obs` 派生量

## 7. 可用于奖励函数的信号
- **位置信息**：`next_obs[0]`（相对目标垫水平位置）、`next_obs[1]`（相对目标垫垂直位置）
- **速度信息**：`next_obs[2]`（水平速度）、`next_obs[3]`（垂直速度）
- **姿态信息**：`next_obs[4]`（姿态角）、`next_obs[5]`（角速度）
- **接触信息**：`next_obs[6]`（左支撑接触）、`next_obs[7]`（右支撑接触）
- **动作/引擎使用**：`action` 是否为 1,2,3（引擎激活）可构造能耗惩罚

## 8. 不确定或不可用的信号
- 官方原始奖励：被掩蔽，不可用
- `info` 中的任何字段：不可用（空字典）
- `training_progress`：未授权使用
- 明确的任务成功/失败标志：不存在于 `info` 或 `terminated` 中，只能通过终止条件的名称推断，不能直接用作奖励信号
- 外界风速或其他随机扰动：源码中被省略，不可知



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。

## 1. 任务路由摘要
- navigation_goal_reaching：用密集过程引导；无 success flag 时禁用终点成功核心项；重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要
### progress_delta_reward
- 角色: 主学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 本轮建议: 推荐作为 v1 主信号：奖励每一步更接近目标。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 本轮建议: 可作为小权重 anchor；不要和 progress_delta_reward 同时大权重堆叠。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 本轮建议: 如果任务要求稳定接近/着陆，v1 可以小权重加入。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速撞击或姿态失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 本轮建议: 可选小权重；不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 本轮建议: 若 explicit_success_flag_available=false，不作为 v1 核心项。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 本轮建议: 若 explicit_failure_flag_available=false，不作为 v1 核心项。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 本轮建议: 通常后续加入，不建议 v1 太早加入。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 本轮建议: 通常后续加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能接近并稳定后再优化燃料。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 本轮建议: v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 本轮建议: 不作为 v1 首选；比 progress_delta 更抽象。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget，而不是固定组件数量。
- 推荐 2~4 个组件：1 个主学习信号 + 0~2 个稳定/安全约束 + 0~1 个任务完成 proxy。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty 作为 v1 核心项。
- 如果速度/姿态信号明确可用，且任务需要稳定接近或着陆，可以加入轻量 stability_penalty。
- 如果使用 contact，只能作为 soft_landing_proxy 的一部分，必须和 near_target、low_speed、stable_angle 组合，不要直接把 contact 当 success。
- energy_penalty、time_penalty、gated_reward 默认后续迭代再加入。
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。