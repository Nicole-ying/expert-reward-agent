# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器（或着陆器）从视口顶部中心初始区域出发，在随机初始扰动下，**尽可能快地到达并稳定降落在中央目标着陆垫上**，同时**尽量减少引擎推力的使用**。智能体需要学会接近目标、减速、保持姿态稳定，并安全地与垫子接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: `x_position` — 飞行器相对于目标垫的**水平坐标**（可能经过缩放/归一化）
- obs[1]: `y_position` — 飞行器相对于垫面高度的**垂直坐标**
- obs[2]: `x_velocity` — 水平线速度
- obs[3]: `y_velocity` — 垂直线速度
- obs[4]: `body_angle` — 机体朝向角（弧度）
- obs[5]: `angular_velocity` — 角速度
- obs[6]: `left_support_contact` — 左侧支撑/着陆脚接触标志（0.0 或 1.0）
- obs[7]: `right_support_contact` — 右侧支撑/着陆脚接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` — 不进行任何引擎点火，无推力。
- action 1: `left_orientation_engine` — 点火左侧姿态引擎（产生逆时针旋转力矩）。
- action 2: `main_engine` — 点火主引擎（通常提供向上的主推力，可能同时产生微小转矩或水平分量，具体取决于机体朝向）。
- action 3: `right_orientation_engine` — 点火右侧姿态引擎（产生顺时针旋转力矩）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` — 当飞行器在目标垫上稳定着陆并静止时，物理引擎将身体标记为不活跃（settled），这可能对应成功着陆。但需要结合位置和接触状态确认是否真正在目标上。
- **failure-like termination**:  
  - `crash_or_body_contact` — 飞行器发生坠毁（如高速碰撞地面、壁面）或身体某些部分与不应接触的物体接触，导致坠毁或损坏。  
  - `horizontal_position_outside_viewport` — 飞行器水平方向飞出有效范围，无法再返回。
- **ambiguous termination**:  
  `body_not_awake_or_settled` 也可能在非目标区域发生（例如坠毁后静止在视口边缘），此时视为失败。需要利用位置观测（obs[0], obs[1]）和接触标志（obs[6], obs[7]）进一步区分。
- **truncation**: 未在掩码源代码中显式声明，但环境可能内置最大步数截断。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: `{}` （空字典，无任何字段）
- **forbidden_or_uncertain_info_fields**: `info` 中不包含任何键，**禁止依赖 `info` 判断成功或失败**；`terminated` 真值未传入 `compute_reward`，**禁止使用终止标志**。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs` — 长度为 8 的 NumPy 数组（含义见观察空间）
- `action` — 整数动作（0,1,2,3）
- `next_obs` — 同样形状的数组，表示执行动作后的新观测
- `info` — 允许使用但内容为空，因此无有效信号
- `training_progress` — **仅当任务描述明确允许时才使用**；此处未明确允许，禁止使用

**严格禁止使用**：
- `original_reward`
- 任何名为 `official_reward`、`true_reward` 的变量
- 未在观察空间中声明的 `obs` 或 `next_obs` 的额外维度
- 未声明的 `info` 字段（`info` 为空，不得假设其包含 `success`、`failure` 等）
- 来自环境步进返回值的 `terminated` 或 `truncated` 标志（它们未传入该函数）

## 7. 可用于奖励函数的信号
基于 `obs` 和 `next_obs` 可以直接使用以下信号构建奖励：
- **位置**：
  - `next_obs[0]` — 水平位置误差（希望趋近 0）
  - `next_obs[1]` — 垂直位置误差（希望趋近某个理想值，通常为 0 表示刚好在垫面高度）
- **速度**：
  - `next_obs[2]`, `next_obs[3]` — 线性速度的大小，着陆时应接近于 0
- **姿态**：
  - `next_obs[4]` — 机体角度，着陆时希望接近于 0（竖直）
  - `next_obs[5]` — 角速度，着陆过程中应可控且最终为 0
- **接触**：
  - `next_obs[6]`, `next_obs[7]` — 左右支撑接触标志，着陆成功时两者通常为 1
- **动作**：
  - `action` — 可用于惩罚非零引擎使用，以促进节能
- **变化量**：
  - 可计算 `obs[:6]` 与 `next_obs[:6]` 的差分，评估朝向目标状态的改善（如距离减小、速度降低等）

## 8. 不确定或不可用的信号
- **info 中无任何字段**：无法直接获得成功/失败标志、接近程度、燃料消耗量等辅助信息。
- **引擎推力强度**：只知道是否点火，但无法获得实际推力大小、燃料消耗的具体数值。
- **终止原因**：`compute_reward` 接口未传入 `terminated`，无法在奖励函数内准确判断当前步是否触发了成功或失败终止；只能通过 `next_obs` 的取值间接推测（如位置、速度是否在安全范围内，但这并非百分百可靠，且边界未知）。
- **视口边界信息**：没有直接提供视口尺度，因此很难通过观测唯一地判定“出界”终止，但可以根据位置数值的突变或极端值做经验性猜测（不推荐作为稳定信号）。
- **任何与任务原始奖励相关的分解项**：完全不可用。



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