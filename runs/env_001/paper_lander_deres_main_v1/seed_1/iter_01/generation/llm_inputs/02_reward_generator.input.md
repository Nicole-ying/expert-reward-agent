# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个2D飞行器/着陆器轨迹优化任务。飞行器初始位于画面顶部中央附近，带有随机初始作用力。  
**核心目标**：飞行器应尽快且平稳地降落在场地中央的“目标垫”上，并保持稳定的姿态与相对静止。  
**附属约束**：尽可能少地使用引擎推力，但这不是与核心目标冲突的多目标优化任务，而是围绕“高效着陆”的自然偏好。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推断为 float (位置、速度、角度、角速度为连续值，接触标志用 1.0/0.0 表示)  
- obs[0]: x_position — 飞行器相对于目标垫中心的水平距离 (接近0表示水平对齐)  
- obs[1]: y_position — 飞行器相对于目标垫高度的垂直距离 (接近0表示正好在垫面高度)  
- obs[2]: x_velocity — 水平线速度  
- obs[3]: y_velocity — 垂直线速度  
- obs[4]: body_angle — 机体倾斜角 (接近0表示竖直)  
- obs[5]: angular_velocity — 角速度  
- obs[6]: left_support_contact — 左支撑脚触地标志 (1.0 触碰, 0.0 未触碰)  
- obs[7]: right_support_contact — 右支撑脚触地标志 (1.0 触碰, 0.0 未触碰)

## 4. 动作空间 action_space
- type: Discrete (4)  
- action 0: no_engine — 不点火，无推力输出  
- action 1: left_orientation_engine — 侧向引擎喷火，产生侧向推力（用于调整机头朝向/水平速度）  
- action 2: main_engine — 主引擎喷火，产生向上的主要推力（用于减速/悬停）  
- action 3: right_orientation_engine — 与左侧引擎对称的侧向喷火

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` 触发时，若飞行器双支撑脚均触地 (`left_support_contact == 1.0` 且 `right_support_contact == 1.0`)、水平/垂直位置接近 0、速度很小、机体角度近乎竖直，则很可能是一次成功着陆。  
- failure-like termination:  
  - `crash_or_body_contact` — 机体与地面非目标区域剧烈接触（如翻倒、撞击地面），通常视为失败。  
  - `horizontal_position_outside_viewport` — 飞出水平边界，视为失败。  
- ambiguous termination:  
  `body_not_awake_or_settled` 也可能出现在非成功场景（例如飞行器靠某种方式静止在错误位置），因此单靠终止信号无法完全确定成功与否，需要结合观察信号自行判定。  
- truncation: 源信息中未提及 episode 步长上限，因此无显式 truncation；但实际使用中可能存在外部限制（如最大步数），非环境内建。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: (空) — 该环境 `info` 返回空字典 `{}`，因此 `info` 不提供任何额外字段。  
- forbidden_or_uncertain_info_fields: `info` 所有字段均不可用，`info["success"]`、`info["termination_reason"]` 等均不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` (上一个状态的完整8维观察)
- `action` (当前执行的动作)
- `next_obs` (动作执行后的新状态)
- **无** — `info` 为空字典，禁止假定任何字段存在。
禁止使用：
- `original_reward` — 被遮蔽，不可用。
- 任何 `info` 字段。
- `training_progress` — 未授权，不可用。
- 任何未声明的 `obs` / `next_obs` 切片维度外信息。

## 7. 可用于奖励函数的信号
从 `next_obs` 中可直接提取的、可量化的信号包括：
- **位置**：`next_obs[0]` (x), `next_obs[1]` (y) — 表示与目标垫的相对距离，可用于塑造接近/居中的奖励。  
- **速度**：`next_obs[2]` (vx), `next_obs[3]` (vy) — 可用于惩罚过高触地速度。  
- **姿态**：`next_obs[4]` (角度) — 可用于奖励竖直姿态（接近0）。  
- **角速度**：`next_obs[5]` — 可用于惩罚快速旋转。  
- **接触状态**：`next_obs[6]`, `next_obs[7]` — 双脚同时触地可作为成功着陆的有力判据，也可用于塑造中间奖励。  
- **动作/引擎使用**：`action` 为非零（使用引擎）时，可考虑付出燃油成本。

## 8. 不确定或不可用的信号
- `info` 全部内容（为空）  
- 任何显式的成功/失败标记  
- 官方的 `original_reward`（已遮蔽，禁止重建）  
- 超出 8 维的观察值  
- 真实的连续时间步序号或进度信息（除非外部提供，但未被环境声明可用）



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。

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
- 从上述任务路由推荐的骨架中选择，优先选择所需信号在环境接口中可用的骨架。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。