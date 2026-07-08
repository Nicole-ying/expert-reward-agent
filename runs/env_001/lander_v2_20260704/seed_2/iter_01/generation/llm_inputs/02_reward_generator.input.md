# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。智能体从靠近视口顶部中心的位置出发，并承受一个随机初始作用力。目标是**尽快飞行并稳定降落在中央目标平台上，同时尽可能节省发动机推力**。智能体需要学习：平滑接近目标区域、控制下落速度、保持机身姿态稳定，并安全地让两侧支撑腿同时接触平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box  
- shape: [8,]  
- dtype: 通常为 float32（环境未显式声明，按连续控制环境惯例）  
- 各维度含义（index 从 0 开始）：
  - `obs[0]`: **x_position** —— 机体相对于目标平台中心的水平距离（相对坐标）  
  - `obs[1]`: **y_position** —— 机体相对于目标平台支撑面高度的垂直距离（相对坐标）  
  - `obs[2]`: **x_velocity** —— 机体水平线速度  
  - `obs[3]`: **y_velocity** —— 机体垂直线速度  
  - `obs[4]`: **body_angle** —— 机身倾斜角度（朝向）  
  - `obs[5]`: **angular_velocity** —— 机身绕质心的角速度  
  - `obs[6]`: **left_support_contact** —— 左侧支撑腿是否接触（0.0 或 1.0）  
  - `obs[7]`: **right_support_contact** —— 右侧支撑腿是否接触（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)  
- 离散动作含义：
  - `action 0`: **no_engine** —— 不开启任何发动机，被动受重力与惯性影响  
  - `action 1`: **left_orientation_engine** —— 点燃左侧姿态控制发动机（产生旋转力矩并伴随微小推力）  
  - `action 2`: **main_engine** —— 点燃主发动机（产生向上的主推力，同时可能伴随偏航力矩）  
  - `action 3`: **right_orientation_engine** —— 点燃右侧姿态控制发动机（与 action 1 方向相反的姿态控制）

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：`body_not_awake_or_settled` 当机身静止/稳定时触发。若此时机身位于目标平台区域、姿态接近水平且两腿均接触，则该终止可能对应成功着陆。但由于步骤函数未提供成功标记，此终止本质上仍具有歧义性。
- **failure-like termination**：  
  - `crash_or_body_contact` —— 机身本体（非支撑腿）撞击地面或其它障碍物，几乎一定意味着失败。  
  - `horizontal_position_outside_viewport` —— 水平位置超出允许边界，通常为飞出任务区域，属于失败。  
- **ambiguous termination**：  
  - `body_not_awake_or_settled` 若在平台外触发（如悬空稳定、在边界外停住或翻倒后静止），则属于失败；但因缺乏显式的成功/失败旗标，无法在终止时直接分辨。  
- **truncation**：代码片段未展示步数限制，但许多类似环境有 `max_steps` 截断；当前信息不足，暂不列入可用信号。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
- `explicit_failure_flag_available`: **false**  
- `allowed_info_fields`: `[]`（`info` 始终返回空字典 `{}`，未提供任何额外字段，亦无 `success`/`failure` 等字段）  
- `forbidden_or_uncertain_info_fields`: 所有 `info` 字段均不可用（因信息为空，强行使用会导致 KeyError 或不可预料行为）。特别说明：`info["success"]`、`info["failure"]`、`info["termination_reason"]` 等均不存在，严禁假设。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：上一时刻的观察数组（8维）  
- `action`：当前步骤执行的动作（0/1/2/3）  
- `next_obs`：执行动作后的观察数组（8维）  
- `info` 中明确允许的字段（当前为空，即不可用）  
- `training_progress`：**仅当 prompt 中显式说明可以使用时才可访问**；此处无说明，故应视为禁止使用

严格禁止使用：
- `original_reward`（被遮盖的官方奖励，严禁读取或拟合）  
- 任何未声明的 `info` 字段  
- 任何未在 3 中声明含义的 `obs` / `next_obs` 索引切片（如超出 0-7 的索引）  
- 环境内部状态或通过其他手段获得的 ground truth (如绝对位置、目标平台坐标等)

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`（水平相对距离）、`obs[1]`（垂直相对距离）及 `next_obs` 对应值，可构建距离惩罚或接近奖励  
- **速度**：`obs[2]`、`obs[3]`，用于惩罚过大的线速度（软着陆要求）  
- **朝向/角速度**：`obs[4]`、`obs[5]`，用于惩罚倾斜姿态或旋转  
- **接触**：`obs[6]`、`obs[7]`，两腿接触状态可用于奖励安全着陆（两腿同时接触）或惩罚单腿/无接触  
- **动作/发动机使用**：`action` 本身可用于计算节省燃料的代价（如动作=2 主发动机时惩罚，动作=0 时奖励）  
- 上述信号均可在 `compute_reward` 中通过 `obs`、`action`、`next_obs` 安全获取。

## 8. 不确定或不可用的信号
- 环境在终止时**不提供任何显式的成功/失败标记**，因此无法从 `info` 或特殊字段直接判断任务是否完成。  
- 原始官方奖励 `original_reward` 已被遮蔽，不可用。  
- `info` 中不存在任何字段，故无法使用 `info` 进行奖励塑形。  
- 无法获取绝对目标坐标或绝对视口边界，只能依赖相对位置 `obs[0:1]`。  
- 由于无显式成功标记，终止条件 `body_not_awake_or_settled` 自身不能直接作为奖励信号（需要结合位置/接触判断）。



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