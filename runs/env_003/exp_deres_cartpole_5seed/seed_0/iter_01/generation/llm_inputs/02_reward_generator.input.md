# environment_card.md

# Env_003 环境理解卡片

## 1. 任务目标
智能体通过向左或向右施加一个固定大小的水平力，控制一个沿轨道移动的底座。目标是在底座不超出轨道边界（`|base_position| ≤ 2.4`）的前提下，尽可能保持底座上方未驱动的杆子直立（`|pole_angle| ≤ 0.20943951` 弧度）。任务以存活时间（步数）为核心追求，每步不倒塌、不出界即为成功延续，一旦违反任一终止条件即失败，在 500 步截断时视为完成一次完整尝试。

## 2. 任务类型选择
selected_route_id: survival_balance_task  
confidence: high  
reason: 任务核心是“保持平衡 + 存活”，没有显式目标位置或目标达成信号，只有失败边界。失败条件明确（杆子倾斜过大或底座出界），成功等同于存活至截断。这完全符合 survival_balance_task 的特征。

## 3. 观察空间 observation_space
- type: Box (连续值)
- shape: (4,)
- dtype: float32
- obs[0] (index 0, base_position): 底座在轨道上的水平位置，理论范围 [-4.8, 4.8]，终止边界为 ±2.4。
- obs[1] (index 1, base_velocity): 底座的水平速度，无界。
- obs[2] (index 2, pole_angle): 杆子相对于竖直方向的角度（弧度），范围 [-0.418879, 0.418879]，实际终止边界为 ±0.20943951。
- obs[3] (index 3, pole_angular_velocity): 杆子的角速度，无界。

## 4. 动作空间 action_space
- type: Discrete(2)
- action 0: push_negative_direction —— 向轨道负方向施加固定水平力。
- action 1: push_positive_direction —— 向轨道正方向施加固定水平力。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无。没有“到达目标”等成功事件，唯一正向结果是存活到 500 步截断。
- failure-like termination: 底座位置绝对值超过 2.4（出界），或杆子角度绝对值超过 0.20943951（倒下），二者之一发生即终止，属于失败。
- ambiguous termination: 无。
- truncation: 达到 500 步时截断，此时底座和杆子均未触发失败条件，可视为一次完整的存活尝试。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `info` 固定为 `{}`，此环境中没有任何额外字段可用。
- forbidden_or_uncertain_info_fields: 所有自定义字段（如 `"success"`, `"failure"`, `"termination_reason"` 等）均不存在且禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前步的观察（shape (4,)）
- `action`：执行的动作（0 或 1）
- `next_obs`：下一步的观察（shape (4,)）
- `info`：仅限当前环境明确暴露的字段（当前为空字典）

严格禁止：
- `original_reward`：官方奖励已被隐藏，不得任何形式使用或还原。
- 任何未在 `allowed_info_fields` 中列出的 `info` 字段。
- `training_progress` 参数（本任务描述中未明确允许使用它）。

## 7. 可用于奖励函数的信号
以下信号全部来自 `obs` 和 `next_obs`，无需依赖 `info`：
- 底座位置：`obs[0]`, `next_obs[0]`
- 底座速度：`obs[1]`, `next_obs[1]`
- 杆子角度：`obs[2]`, `next_obs[2]`（越接近 0 越直立）
- 杆子角速度：`obs[3]`, `next_obs[3]`
- 动作：`action`（推力方向）

## 8. 不确定或不可用的信号
- 官方原始奖励：已遮蔽，禁止使用。
- 任何 `info` 内的自定义标志（如 `"success"`, `"failure"`, `"TimeLimit.truncated"` 等）：不存在且不可用。
- 额外的环境信息（如力矩、能量消耗等）：未提供，不可用。



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

### event_reward
- 角色: 事件目标奖励
- 数学形态: R_event * I[event]
- 需要信号: event flag
- 使用说明: 对特定事件给予奖励。适合 resource_gathering 等离散目标任务。
- 风险: event_reward_farming（反复触发事件）。
- 后续迭代: 若事件被 exploit，加 cooldown 或递减。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 从上述任务路由推荐的骨架中选择，优先选择所需信号在环境接口中可用的骨架。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。