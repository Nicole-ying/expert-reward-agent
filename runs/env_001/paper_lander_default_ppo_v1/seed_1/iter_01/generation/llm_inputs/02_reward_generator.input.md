# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器着陆任务。飞行器从视口上部中心附近出发，受随机初始力影响。目标是**尽快平稳地降落在中央的着陆垫上**，同时**尽可能少地使用引擎推力**。智能体需要学习：向目标靠近、减速、保持机身姿态稳定、实现安全触垫。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x位置（相对着陆垫的水平坐标）
- obs[1]: y位置（相对垫高度的垂直坐标）
- obs[2]: x方向线速度
- obs[3]: y方向线速度
- obs[4]: 机身角度（俯仰角）
- obs[5]: 角速度
- obs[6]: 左支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: 右支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎（不施加推力）
- action 1: 左侧姿控引擎点火
- action 2: 主引擎点火（向下推力）
- action 3: 右侧姿控引擎点火

## 5. step 与终止条件分析
### 5.1 终止模式
Termination 由以下三种条件触发（任一为真即结束）：
1. `crash_or_body_contact`：坠毁或身体与地面/其他物体发生接触（大概率是失败）。
2. `horizontal_position_outside_viewport`：水平位置超出边界（失败）。
3. `body_not_awake_or_settled`：机体进入休眠/稳定状态。当飞行器完全静止、不再受物理唤醒时触发。若此时已处于着陆垫上且支撑腿接触，则通常表示成功着陆；若在坠毁后静止，则被前面的 crash 条件先行触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（step 返回的 info 为空，无显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`，不可使用任何字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。由于源码显式返回 `{}`，不能假设存在 `success`、`termination_reason` 等键。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`：当前观察向量（完整 8 维）
- `action`：当前执行的动作（0,1,2,3）
- `next_obs`：下一时刻观察向量（完整 8 维）
- `info`：空字典（实际无可用信息）
- `training_progress`：禁止使用（除非 prompt 明确允许）

禁止使用：
- `original_reward`（官方奖励被遮蔽，不可重建或依赖）
- `official_reward` 或其它变体
- 未声明的 `info` 字段（例如 `success`、`failure`、`termination_reason` 均不存在）
- 任何未在上述“允许使用”中列出的外部信号

## 7. 可用于奖励函数的信号
- **位置**：相对着陆垫的 x, y 坐标（obs[0], obs[1]），可直接用于评估靠近/悬停/触垫。
- **速度**：x, y 线速度（obs[2], obs[3]），可用于惩罚高速撞击或奖励稳定减速。
- **姿态**：机身角度（obs[4]），角速度（obs[5]），用于鼓励保持直立姿态。
- **接触**：左右支撑接触标志（obs[6], obs[7]），可用于检测是否着陆成功、是否平稳（双脚着垫）。
- **动作/引擎使用**：动作编号本身，可以用于惩罚推力使用（动作 1,2,3 为引擎点火）以鼓励省油。

## 8. 不确定或不可用的信号
- **明确成功标志**：info 为空，无法获得。
- **燃料消耗量**：环境未提供燃料量字段，只能从动作序列中自行累加推断。
- **着陆垫位置**：未单独提供绝对坐标，仅通过相对坐标隐含在观察中。
- **是否在着陆垫正上方**：必须结合 y 位置和接触标志间接推断。
- **终止原因**：没有 termination_reason，无法直接区分是成功着陆还是坠毁。只能从最后时刻的观察（如接触标志、位置、速度）逻辑推断。



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