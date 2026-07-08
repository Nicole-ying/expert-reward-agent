# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
在一个 2D 物理环境中，控制一个带有主引擎和两个姿态引擎的飞行器。飞行器初始位置在视口顶部中央附近，受到随机初始力影响。核心目标是从初始位置出发，**尽可能快地降落到画面中央的目标平台上**，并稳定地停在平台上（速度趋于零、姿态水平、支撑腿安全接触）。次要要求是在完成主要目标的前提下，**尽量减少引擎推力使用量**（节省燃料）。简而言之：到达中央平台、快速稳定、省燃料。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 通常为 float32
- obs[0]: 飞行器相对于目标平台的水平坐标 x（目标平台中心 x=0）
- obs[1]: 飞行器相对于目标平台高度的垂直坐标 y（平台表面高度为 0）
- obs[2]: 水平线速度 vx
- obs[3]: 垂直线速度 vy
- obs[4]: 飞行器身体朝向角（弧度），0 表示水平
- obs[5]: 角速度（弧度/秒）
- obs[6]: 左侧支撑腿接触标志（0.0：未接触，1.0：接触）
- obs[7]: 右侧支撑腿接触标志（0.0：未接触，1.0：接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine（不点火，仅受重力/惯性影响）
- action 1: left_orientation_engine（点燃左侧姿态引擎，产生向右的力矩或侧向推力）
- action 2: main_engine（点燃主引擎，产生向上的推力）
- action 3: right_orientation_engine（点燃右侧姿态引擎，产生向左的力矩或侧向推力）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（身体静止/稳定）。当飞行器停止运动（可能已着陆且满足稳定条件）时触发，很可能代表成功降落并稳定。但需结合位置、接触等信息进一步确认。
- failure-like termination: `crash_or_body_contact`（碰撞地面或不当身体接触），`horizontal_position_outside_viewport`（水平飞出可视区域）。这两种情况通常意味着撞击地面、偏离目标区域等失败结局。
- ambiguous termination: `body_not_awake_or_settled` 也可能在失败后（如侧翻卡住）触发，不能绝对等同于成功。
- truncation: 环境步始终返回 truncated=False，无时间限制信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 字典为空，无可用的成功/失败标记字段）
- forbidden_or_uncertain_info_fields: info 中不提供任何字段；终止条件的具体判定逻辑不透明，不能作为奖励计算依据；禁止使用任何模拟器内部信号。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前观测，可索引全部 8 个字段
- `action`：执行的动作（0~3 整数）
- `next_obs`：下一时刻观测，同样可索引全部字段
- `info`：空字典，无任何可用字段
- `training_progress`：未在任务描述中授权使用，因此**禁止使用**

禁止使用：
- `original_reward`（被遮蔽的官方奖励）
- 任何基于终止条件（如 crash、outside、settled）的硬编码成功/失败信号
- 任何未经声明的 info 字段
- 模拟器内部状态或额外全局信息

## 7. 可用于奖励函数的信号
- position: `obs[0]`（x 相对目标），`obs[1]`（y 相对平台高度）；`next_obs[0]`, `next_obs[1]`
- velocity: `obs[2]`（vx），`obs[3]`（vy）；以及下一时刻的对应值
- orientation: `obs[4]`（角度），`obs[5]`（角速度）
- contact: `obs[6]`（左腿接触），`obs[7]`（右腿接触），可用于判断着陆状态
- action/engine: 动作 ID 可以反映引擎使用情况（主引擎、姿态引擎、无操作），可用于惩罚燃料消耗
- 可构建的组合特征：与目标的距离、速度大小、是否水平、是否双触地、摆动幅度等

## 8. 不确定或不可用的信号
- 明确的成功/失败标记：info 为空，终止原因无法直接获取，不能用于奖励
- 官方原始奖励：`original_reward` 被遮蔽，严禁使用
- 终止条件的内部实现：`body_not_awake_or_settled`、`crash_or_body_contact` 等判断逻辑未知，不能作为奖励的直接输入
- 任何需要直接访问物理引擎或真实任务名的信息



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

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 使用说明: 连续负距离信号，引导 agent 靠近目标。与 progress_delta_reward 同时大权重使用会产生重复信号。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### progress_delta_reward
- 角色: 密集学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 使用说明: 奖励每一步更接近目标。适合目标位置已知的导航/到达任务。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 使用说明: 基于势能差分的塑形信号。比 progress_delta 更抽象，当任务有明确的势能定义时可使用。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

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



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- landing_bonus + orientation_penalty + progress_reward

上述骨架在前序迭代中已尝试但未取得突破。
请基于训练证据选择改进方向：
- 如果认为同一骨架仍有可修复空间（如系数调节、条件化约束），可以继续在当前骨架上修改。
- 如果诊断表明当前骨架存在结构性问题（如信号冲突、梯度消失），请从 expert_reward_context.md 中选择不同的数学形态。
- 不要机械重复已失败的骨架。
