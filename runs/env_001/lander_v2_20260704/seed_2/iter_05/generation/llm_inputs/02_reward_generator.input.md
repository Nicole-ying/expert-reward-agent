# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器/着陆器轨迹优化任务。主体从画面顶部中央附近出发，带有随机初始速度。  
核心任务：**尽快到达并稳定著陆在画面中央的目标着陆垫上**，同时**尽量减少主引擎推力消耗**。  
代理需学会向目标靠近、减速、保持正确姿态并安全接触垫面。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（其中 contact 标志为 0.0 或 1.0）
- obs[0]: x 坐标，相对于目标着陆垫中心的水平位置
- obs[1]: y 坐标，相对于着陆垫高度的垂直位置
- obs[2]: 水平线速度 vx
- obs[3]: 垂直线速度 vy
- obs[4]: 机身朝向角（弧度）
- obs[5]: 角速度
- obs[6]: 左支撑腿接触标志（1.0 表示接触，0.0 未接触）
- obs[7]: 右支撑腿接触标志（1.0 表示接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` – 不點火，仅靠重力/惯性漂移
- action 1: `left_orientation_engine` – 点燃左侧姿态調整引擎
- action 2: `main_engine` – 点燃主引擎（向下推力，消耗燃料）
- action 3: `right_orientation_engine` – 点燃右侧姿态調整引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- **成功型终止**：`body_not_awake_or_settled`（机体静止/镇定）—— 当它发生且左右支撑腿均接触垫面时，可视为成功着陆。但单独出现时也可能是中途静止或悬空失败，需结合接触信号确认。
- **失败型终止**：
  - `crash_or_body_contact`：机体除着陆腿以外的部分碰撞地面或障碍物，或发生破坏性碰撞。
  - `horizontal_position_outside_viewport`：机体水平方向飞出视野/安全区。
- **模糊终止**：无特殊额外模式，上述三种为唯一终止触发条件。
- **truncation**：未指定最大步长，但通常会有时间上限（若存在，算作截断而非成功/失败）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，`info` 恒为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何 `info` 字段均不可用；官方原始奖励被屏蔽，不可使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`：上一步观察（shape (8,)）
- `action`：刚刚执行的动作（0～3）
- `next_obs`：当前步观察（shape (8,)）
- `info`：已确认恒为空，不提供有效信息
- `training_progress`：仅在明确指示时才可使用，本任务中未要求，**默认禁止**

禁止使用：
- `original_reward`（官方奖励被屏蔽，严禁泄露）
- 任何未在上述清单中声明的 `info` 字段
- 任何未在观察空间说明中包含的 `obs` 切片
- 直接或间接还原官方奖励

## 7. 可用于奖励函数的信号
- 位置：`obs[0]`（x）、`obs[1]`（y），表示与目标垫的相对位置，理想为目标点 (0, 0)
- 线速度：`obs[2]`（vx）、`obs[3]`（vy），理想着陆时接近 0
- 姿态：`obs[4]`（倾角），理想接近 0 （竖直）
- 角速度：`obs[5]`，理想接近 0
- 接触：`obs[6]`、`obs[7]`，左右腿是否着地，两腿同时接触可判断稳定着陆
- 动作/引擎使用：`action` 是否为 2（主引擎）可作为燃料消耗的惩罚信号；动作 1,3 可引导姿态修正

## 8. 不确定或不可用的信号
- 官方原始奖励被完全屏蔽，无法参考
- `info` 为空，不提供任何成功/失败/碰撞类型等辅助信息
- “crash_or_body_contact” 终止条件的具体判定逻辑隐藏，只能通过接触标志与是否终止间接推断：如果终止时 `obs[6]` 或 `obs[7]` 为 1 但 `crash` 为真，可能指示错误接触；若同时满足 `body_not_awake_or_settled` 且双腿接触，则大概率成功
- 环境内部的重力、推力大小、视野边界、物理参数等细节不可直接观测，只能通过位置/速度变化间接感知



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
- best_score_so_far: -108.370

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_reward + landing_quality + stability_penalty | 3 | -108.370 | -109.790 | unsolved |
| distance_reward + soft_landing_bonus + stability_penalty | 1 | -110.970 | -110.970 | unsolved |

## Previous interventions

- iter 2 (score=-112.960, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2** — the 0.5% active_rate binary bonus matches the `sparse_to_dense` evidence pattern exactly; a coefficient tweak cannot fix structural sparsity. | 5. selected_intervention
- iter 4 (score=-109.790, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 1 —— 稳定性惩罚的职责和数学形态合理，但尺度过强阻碍早期探索和必要机动。上一轮修改了landing_quality形态（已生效），本轮不应重复修改同一组件；stability_penalty系数尚未被调整过，且ratio虽低于0.5但处于crash场景下的"过度约束"状态。 | selected_intervention**：仅调整stability_penalty的四个权重系数，将其整体幅度降至约40%（w_vx: 0.15→0.06, w_vy: 0.05→0.02, w_angle: 0.2→0.08, w_angvel: 0.2→0.08）。distance_reward和landing_quality保持不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
