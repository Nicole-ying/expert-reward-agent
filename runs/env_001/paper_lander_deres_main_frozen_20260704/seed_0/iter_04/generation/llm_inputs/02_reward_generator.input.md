# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维飞行器/着陆器任务。智能体从视口顶部中央附近出发，带有随机初始力。核心目标是**尽快抵达中心目标垫并稳定停靠**，同时尽量**减少引擎推力消耗**。智能体需要学习接近目标、降低速度、保持稳定姿态，并与目标垫安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 浮点数（除接触标志可能是 float 外，但含义为二值）
- obs[0]: `x_position` — 相对于目标垫的水平坐标
- obs[1]: `y_position` — 相对于目标垫高度的高度差
- obs[2]: `x_velocity` — 水平线速度
- obs[3]: `y_velocity` — 垂直线速度
- obs[4]: `body_angle` — 机体朝向角度
- obs[5]: `angular_velocity` — 角速度
- obs[6]: `left_support_contact` — 左侧支撑/接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: `right_support_contact` — 右侧支撑/接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` — 不开启任何引擎，惯性滑行
- action 1: `left_orientation_engine` — 点燃左侧姿态调整引擎（产生旋转力矩）
- action 2: `main_engine` — 点燃主引擎（产生向上的推力，可能同时影响角速度）
- action 3: `right_orientation_engine` — 点燃右侧姿态调整引擎（产生相反旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 可能是主体静止/稳定，暗示成功着陆（但需要配合位置判断）；当主体不再移动且位置接近目标垫时，可视作成功
- failure-like termination: `crash_or_body_contact`（非目标垫的碰撞/坠毁）、`horizontal_position_outside_viewport`（水平飞出允许区域）
- ambiguous termination: 仅凭 `body_not_awake_or_settled` 无法直接判定成功或失败，需结合位置、速度、接触等判断；可能包含卡在障碍物上等非成功情况
- truncation: 代码返回 `terminated` 仅由上述条件控制，未提及显式步数截断；但仍可能存在最大步数限制（由环境外部封装，当前片段未体现）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （返回的空字典，无任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；尤其注意 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 均不存在，严禁假设其存在

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前步的观察数组（含全部8个元素）
- action: 当前步执行的动作编号（int）
- next_obs: 下一步的观察数组（含全部8个元素）
- info: 仅允许使用明确声明的字段（当前为空，故 info 不得用于逻辑判断）
- training_progress: 除非 prompt 明确允许，否则**禁止使用**（本环境卡片未明确允许，因此不可用）

禁止使用：
- original_reward（官方奖励或任何隐藏奖励信号）
- official_reward
- 未声明的 info 字段（包括但不限于 success, failure, termination_reason, distance_remaining 等）
- 未声明的 obs 切片（只能使用索引0-7的完整观察，禁止自行假定其他维度）

## 7. 可用于奖励函数的信号
从 obs 和 next_obs 中可直接提取：
- position: `x_position` (obs[0]), `y_position` (obs[1]) — 可用于计算到目标的距离
- velocity: `x_velocity` (obs[2]), `y_velocity` (obs[3]) — 可用于惩罚过快的速度或鼓励减速
- orientation: `body_angle` (obs[4]), `angular_velocity` (obs[5]) — 可用于鼓励保持稳定姿态
- contact: `left_support_contact` (obs[6]), `right_support_contact` (obs[7]) — 可用于判断是否接触目标垫（两侧同时接触可能表示平稳着陆）
- action/engine: 动作编号可用于惩罚引擎使用（动作0无惩罚，动作1/2/3有惩罚）

## 8. 不确定或不可用的信号
- 官方奖励 `original_reward`（已屏蔽）
- 明确的 success / failure 布尔标志（info 中不存在）
- 目标垫的精确坐标（仅能从相对位置推断）
- 真实环境名称、基准名称
- 剩余步数或最大步数（未在片段中暴露）
- 外部风力、引擎冲量等物理参数（被屏蔽）
- 任何通过 info 传递的辅助度量（如 fuel_consumption, distance 等）



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- navigation_goal_reaching：任务目标是接近/到达指定位置。重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

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
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。



# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -111.760

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| progress + soft_landing_bonus + stability_penalty | 2 | -111.760 | -115.610 | unsolved |

## Previous interventions

- iter 2 (score=-115.610, structure=progress + soft_landing_bonus + stability_penalty): selected_level**: Level 1——progress和stability_penalty同为逐步稠密信号，|penalty/progress|=0.858远超0.5经验触发器，且progress是唯一主引导信号。 | selected_intervention**: 仅降低stability_penalty三个系数各10倍：w_vel=0.001, w_angle=0.001, w_angvel=0.0005。其他组件不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
