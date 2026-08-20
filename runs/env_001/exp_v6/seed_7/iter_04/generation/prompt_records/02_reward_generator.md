# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将读取：

1. `environment_card.md`：任务目标、观察空间、动作空间和接口约束；
2. `expert_reward_context.md`：按任务类型检索的候选奖励骨架及风险；
3. 可选的 `restart_context.md`：之前尝试过的奖励和失败证据。

你的任务是生成第一版候选奖励函数 `reward_v1.py`，并附带简短设计说明。

# 核心原则

- 先根据任务目标和可用接口选择奖励结构，不预设任何骨架必然是主信号。
- 专家知识中的推荐骨架只是候选，不是必须使用的模板。
- 每个奖励项必须对应明确的任务角色、可用变量和预期行为。
- 优先生成能够提供有效学习信号的最小设计，但不机械限制组件数量。
- 不为了形式完整而堆叠目标、惩罚、proxy 或课程项。
- 检查不同奖励项是否重复表达同一目标，或者在行为上互相冲突。
- 检查信号的触发频率、数值尺度、作用区域和潜在利用方式。
- 如果接口不足以可靠表达某个目标，就推迟该奖励项，而不是发明变量。

# 骨架选择

你必须自己完成以下判断：

1. 当前任务最重要的行为目标是什么；
2. 哪些观测、动作或已声明的 info 字段能够表达该目标；
3. 哪种候选骨架与这些信号匹配；
4. 是否需要过程引导、任务目标、约束、效率或探索项；
5. 哪些候选骨架应当推迟或拒绝，以及原因；
6. 所选奖励是否可能产生停滞、保守、震荡、投机或长期刷取代理信号。

不要默认选择 `progress_delta_reward`、`distance_reward`、`stability_penalty`、任务完成 proxy 或任何其他固定组合。只有环境证据和接口支持时才选择它们。

# 接口与安全约束

- 不得使用 `original_reward`，也不得重构或估计官方奖励。
- 不得计算 `fitness_score` 或其组成部分。
- 只能使用环境卡明确声明的 `obs`、`next_obs`、`action` 和 `info` 字段。
- 不得发明 `info["success"]`、`info["failure"]`、`termination_reason` 等未声明字段。
- 不得假设未声明的观测维度、坐标含义、动作含义或终止原因。
- 如果没有可靠的 success/failure 信号，不得伪造 terminal success/failure 奖励。
- 连续变量优先使用数值稳定、范围可解释的表达式。
- 不允许 `import`；需要平方根时可使用 `** 0.5`。

# 可观测组件

`components` 只记录直接相加得到 `total_reward` 的奖励项，以及 `total_reward` 本身。

例如：

```python
total_reward = task_reward + constraint_penalty
components = {
    "task_reward": task_reward,
    "constraint_penalty": constraint_penalty,
    "total_reward": total_reward,
}
```

门控系数、距离、速度、阈值判断和其他中间变量不能伪装成独立奖励组件。如果某个门控变量只参与计算一个奖励项，就只记录门控后的最终奖励项。

# 输出格式

函数签名必须完全一致：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

返回格式：

```python
return float(total_reward), components
```

第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为解析器会提取第一个 Python code block。

禁止：

- `import`、`class`、额外函数；
- `try/except`、`eval`、`exec`、`open`；
- `self` 或额外输入变量；
- 未声明的 observation 切片或 info 字段；
- 在函数体中读取 `original_reward`。

# 设计说明

代码后简要说明：

- 选择了哪些奖励项及其角色；
- 每个奖励项使用哪些已声明信号；
- 为什么选择这些骨架，而没有选择其他候选；
- 预期需要观察的失败模式；
- 哪些目标因接口不足或最小设计原则留待后续迭代。

```

## User Prompt

```markdown
# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个2D飞行器从上方区域出发，尽快飞向并稳定降落在画面中央的目标平台上。  
在此过程中，需要尽可能减少引擎推力，同时保持合理姿态与安全接触。  
主要目标是快速、平滑、安全地到达目标平台并稳定（settle）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定的中央目标平台并稳定（goal reaching），尽管也包含节省燃料、稳定姿态等子目标，但这些属于到达目标后的效率与质量约束，并不改变“到达目标”这一本质。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32 (推测，接触标志为 0.0/1.0)  
- obs[0]: x_position —— 相对于目标平台中心点的水平坐标  
- obs[1]: y_position —— 相对于目标平台高度的垂直坐标  
- obs[2]: x_velocity —— 水平方向线速度  
- obs[3]: y_velocity —— 垂直方向线速度  
- obs[4]: body_angle —— 飞行器朝向角  
- obs[5]: angular_velocity —— 角速度  
- obs[6]: left_support_contact —— 左支撑（腿）是否接触平台（0/1 浮点数）  
- obs[7]: right_support_contact —— 右支撑（腿）是否接触平台（0/1 浮点数）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: `no_engine` —— 不做任何操作  
- action 1: `left_orientation_engine` —— 启动一个转向引擎（使飞行器向左旋转）  
- action 2: `main_engine` —— 启动主引擎（产生向上的推力）  
- action 3: `right_orientation_engine` —— 启动另一个转向引擎（使飞行器向右旋转）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  - `body_not_awake_or_settled` 可能是成功的一种形式，即飞行器在平台上方或平台附近稳定（settled）并停止活动。但这需要结合位置、速度等条件进一步判断，不能直接作为成功标志。
- **failure-like termination**:  
  - `crash_or_body_contact` —— 通常表示坠毁或与地面等不期望的物体发生碰撞。  
  - `horizontal_position_outside_viewport` —— 水平位置超出视野边界，明显是失败。
- **ambiguous termination**:  
  - `body_not_awake_or_settled` 也可能发生在非目标区域的静止状态，因此仅凭该条件无法确知是否成功。
- **truncation**:  
  - 环境中未出现 `Truncated` 信息（完整 mask 中的 `return` 只有 `terminated, False`），因此环境不存在基于步数或时间的截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 为空字典，且环境未声明任何额外字段）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；`success`、`failure`、`termination_reason` 等均不存在且不可假设。

**重要提醒**：奖励函数中不能依赖任何 info 中的成功/失败标志，必须基于观测、动作和终止状态自行推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`  —— 当前步骤的观测向量（含 8 个维度）
- `action` —— 所执行的动作索引 (0-3)
- `next_obs` —— 执行动作后下一时刻的观测向量（同 8 维）
- `info` 中明确允许的字段（本例中不存在）
- `training_progress` **不得使用**（任务描述未明确允许，遵循“只有 prompt 明确允许时才用”的原则）

禁止使用：
- `original_reward`（原始奖励已被屏蔽，不可重建或依赖）
- 任何未声明的 info 字段
- 任何未明确列在观测空间或禁止字段列表之外的辅助量

## 7. 可用于奖励函数的信号
- signal name: x_position
  - source: obs[0] / next_obs[0]
  - meaning: 飞行器相对于目标平台中心的水平距离
  - availability: available

- signal name: y_position
  - source: obs[1] / next_obs[1]
  - meaning: 飞行器相对于目标平台高度的垂直距离（目标平台高度处为 0）
  - availability: available

- signal name: x_velocity
  - source: obs[2] / next_obs[2]
  - meaning: 水平方向线速度
  - availability: available

- signal name: y_velocity
  - source: obs[3] / next_obs[3]
  - meaning: 垂直方向线速度
  - availability: available

- signal name: body_angle
  - source: obs[4] / next_obs[4]
  - meaning: 飞行器朝向角
  - availability: available

- signal name: angular_velocity
  - source: obs[5] / next_obs[5]
  - meaning: 角速度
  - availability: available

- signal name: left_contact
  - source: obs[6] / next_obs[6]
  - meaning: 左支撑腿是否接触平台（1.0 表示接触）
  - availability: available

- signal name: right_contact
  - source: obs[7] / next_obs[7]
  - meaning: 右支撑腿是否接触平台（1.0 表示接触）
  - availability: available

- signal name: action_main_engine_used
  - source: action == 2
  - meaning: 是否使用了主引擎（可作为推力/燃料消耗的惩罚依据）
  - availability: available

- signal name: action_orientation_engine_used
  - source: action in {1, 3}
  - meaning: 是否使用了转向引擎（同样可视为能耗或控制惩罚）
  - availability: available

## 8. 不确定或不可用的信号
- 官方原始奖励 (original_reward) —— 被明确屏蔽，不可用
- 任何 info 字段（包括 success, failure, episode_length 等）—— 不可用
- training_progress —— 未声明允许，不可用
- 对外部环境知识的假设（例如精确物理参数、目标区域形状、燃料量等）—— 不可用
- 显式的成功/失败标志 —— 不存在，不可用



# expert_reward_context.md

# 专家奖励知识上下文

本上下文按环境分析得到的任务类型检索。推荐骨架是候选集合，不是固定答案，也不要求全部使用。
Reward Generator 必须先检查环境卡中的变量可用性，再独立决定选择、组合、推迟或拒绝骨架。

## 任务路由

- selected_route_id: navigation_goal_reaching
- route_recommended_candidates: terminal_success_reward, terminal_failure_penalty, time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward

## 路由候选骨架

## Skeleton 1：终点成功奖励

```yaml
skeleton_id: terminal_success_reward
中文名称: 终点成功奖励
功能角色:
  - 任务目标奖励
数学形态: r_success = R_success * I[success]
适用场景:
  - 到达目标
  - 完成任务
  - 抓取成功
  - 游戏胜利
需要变量:
  - success flag
风险:
  - 奖励稀疏
  - 随机探索难以触发
常见失败模式:
  - sparse_reward_no_learning
修正:
  - 加过程引导奖励
  - 加阶段性事件奖励
  - 加势能塑形
调用规则:
  - 如果任务有明确成功条件，优先加入
```

---

---

## Skeleton 2：失败惩罚

```yaml
skeleton_id: terminal_failure_penalty
中文名称: 失败惩罚
功能角色:
  - 安全约束惩罚
数学形态: r_failure = -R_failure * I[failure]
适用场景:
  - 摔倒
  - 碰撞
  - 越界
  - 死亡
  - 任务失败
需要变量:
  - failure flag
  - termination reason
风险:
  - 太小则智能体不怕失败
  - 太大则智能体过度保守
修正:
  - 区分轻微违规和严重失败
  - 使用连续惩罚或门控奖励
调用规则:
  - 如果环境有 failure/terminated 条件，必须考虑
```

---

---

## Skeleton 3：每步时间惩罚

```yaml
skeleton_id: time_penalty
中文名称: 每步时间惩罚
功能角色:
  - 行为效率约束
数学形态: r_time = -lambda_time
适用场景:
  - 希望尽快完成任务
  - 导航
  - 路径规划
  - 迷宫
风险:
  - 太大可能导致冒险行为
  - 对需要等待的任务可能有害
修正:
  - 使用较小权重
  - 与成功奖励配合
调用规则:
  - 如果目标是尽快完成任务，加入
```

---

---

## Skeleton 5：距离型密集奖励

```yaml
skeleton_id: distance_reward
中文名称: 距离型密集奖励
功能角色:
  - 过程引导奖励
数学形态: r_distance = -d(s_t, goal)
适用场景:
  - 目标位置明确
  - 可以定义距离
  - 成功奖励稀疏
需要变量:
  - 当前状态位置
  - 目标位置
风险:
  - 接近目标但不完成任务
  - 距离函数不等于真实任务进度
修正:
  - 加终点成功奖励
  - 加时间惩罚
  - 改成增量进步奖励
调用规则:
  - 如果可定义距离且稀疏奖励难学，可加入
```

---

---

## Skeleton 6：增量进步奖励

```yaml
skeleton_id: progress_delta_reward
中文名称: 增量进步奖励
功能角色:
  - 过程引导奖励
数学形态: r_progress = d(s_t, goal) - d(s_{t+1}, goal)
适用场景:
  - 导航
  - 接近目标
  - 机器人移动
  - 机械臂接近物体
需要变量:
  - 上一步距离
  - 当前距离
风险:
  - 目标附近震荡刷分
  - 只优化局部进展
常见失败模式:
  - goal_near_oscillation
修正:
  - 加终点奖励
  - 加时间惩罚
  - 裁剪 progress
调用规则:
  - 如果任务有“越来越接近目标”的结构，优先考虑
```

---

---

## Skeleton 14：势能塑形奖励

```yaml
skeleton_id: potential_based_shaping
中文名称: 势能塑形奖励
功能角色:
  - 过程引导奖励
  - 奖励塑形
数学形态:
  F(s,a,s') = gamma * Phi(s') - Phi(s)
  R' = R + F
适用场景:
  - 稀疏奖励
  - 可以定义状态好坏函数 Phi
风险:
  - Phi 设计错误会误导学习
  - 塑形项压过原始任务奖励
修正:
  - 让 Phi 与真实目标一致
  - 记录 shaping term
文献依据:
  - Ng et al. 1999 potential-based reward shaping
调用规则:
  - 如果希望提供学习引导且尽量保持原目标，考虑
```

---

---

## Skeleton 10：门控/层级奖励

```yaml
skeleton_id: gated_reward
中文名称: 门控/层级奖励
功能角色:
  - 安全优先结构
数学形态:
  if unsafe:
      r = -R_large
  else:
      r = r_task
适用场景:
  - 安全优先
  - 自动驾驶
  - 医疗控制
  - 碰撞不可接受
风险:
  - 太保守
  - unsafe 判断过宽导致学习困难
修正:
  - 区分硬约束和软约束
  - 硬约束门控，软约束连续惩罚
调用规则:
  - 如果某个约束不能被其他奖励抵消，使用
```

---

---

## 其他可用骨架目录

terminal_success_reward, terminal_failure_penalty, time_penalty, alive_bonus, distance_reward, progress_delta_reward, forward_progress_reward, event_reward, weighted_sum_reward, gated_reward, energy_penalty, action_smoothness_penalty, stability_penalty, potential_based_shaping, intrinsic_exploration_reward, dynamic_curriculum_reward, learned_preference_reward

## 选择规则

- 不机械照抄路由推荐，也不默认任何骨架是主信号。
- 只选择环境卡中有可靠输入支持的骨架。
- 根据任务目标判断是否需要任务目标、过程引导、约束、效率、探索或课程角色。
- 最小设计意味着删除无证据的项，而不是固定组件数量或固定组件组合。
- 没有显式 success/failure 信号时，不得发明对应字段。
- 需要其他骨架时，可以从完整目录中选择，但必须说明接口依据和风险。
- 中间量和门控系数不是独立 reward term，不应放入 components 与奖励贡献比较。



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- landing_bonus + progress_delta + stability_penalty

请选择一个**完全不同的主信号骨架**。例如如果上述列表都是 progress_delta 系列，请尝试 potential_based_shaping 或 bounded_proximity。不要重复已失败的骨架。

```
