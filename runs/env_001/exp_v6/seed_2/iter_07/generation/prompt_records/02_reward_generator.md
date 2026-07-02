# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境背景；
2. expert_reward_context.md：RAG 检索并压缩后的专家知识；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务：
直接生成第一版奖励函数 `reward_v1.py`，并附带一份简短设计说明。

# 总体设计原则

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 不要机械照抄 route 推荐公式。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 对 Env_001 这类二维任务，禁止把位置写成三维。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# role-based component budget

reward_v1 推荐使用 2~4 个组件，每个必须有明确角色，不能为了显得完整而堆叠。

必须包含：
- 1 个主学习信号：通常选择 progress_delta_reward；如果无法计算 delta，用 distance_reward。也可以尝试 potential_based_shaping（Φ(s')-γΦ(s)），但需写出明确的势能函数。

允许包含：
- 0~2 个稳定/安全约束：例如速度、姿态角、角速度惩罚。
- 0~1 个任务完成 proxy：soft proxy，不能伪造 success flag。
- 0~1 个效率/动作代价：权重必须很小。

默认不在 v1 使用：
- terminal_success_reward（需显式 success flag）
- terminal_failure_penalty（需显式 failure flag）
- gated_reward（多阶段，后续迭代再加）
- dynamic curriculum

components dict 里只能出现 total_reward 公式中等号右边直接出现的变量名。如 `total_reward = A + B + C`，则 components 只有 A、B、C、total_reward。不要把 A 的计算中间变量（如 A = a1+a2+a3 中的 a1）放进 components。

避免重复：
- 不要同时大权重使用 distance_reward 和 progress_delta_reward。
- 如果同时使用，progress_delta_reward 是主信号，distance_reward 只能是小权重辅助 anchor。

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

为了兼容旧 wrapper，也可以把 components 写入 `info["reward_terms"]`，但仍然建议返回 `(float(total_reward), components)`。

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 至少包含所有被加到 total_reward 的组件，以及 total_reward。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- 使用了哪些奖励组件；
- 每个组件的角色；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些组件留到后续迭代；
- 训练后应该观察哪些 failure mode。
```

## User Prompt

```markdown
# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
一个类似飞行器的物体从视口顶部中央附近开始运动（有初始随机力），目标是**尽快到达并稳定停留在画面中央的目标着陆垫上**，同时**尽可能少地使用引擎推力**。
智能体需要学习：
- 向目标点靠近（调整位置和速度），
- 在到达目标区域时将速度减到很小（“settle”），
- 保持平稳的姿态（角度接近零），
- 安全着陆（左右支撑脚都与垫子接触）。
整个过程在时间和燃油消耗之间寻求平衡。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是“让智能体到达并稳定在指定的目标位置上”，属于典型的导航/目标到达任务。最低消耗和快速完成是对到达策略的附加约束，并不改变任务的根本性质。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `(8,)`
- dtype: 提取的特征均为 `float32`（接触标志为 0.0 或 1.0）
- 各维度含义：

| 索引 | 名称                      | 含义与说明 |
|------|---------------------------|------------|
| 0    | x_position                | 机体水平坐标，相对于目标着陆垫的横向偏移 |
| 1    | y_position                | 机体垂直坐标，相对于着陆垫高度的偏移 |
| 2    | x_velocity                | 机体水平方向线速度 |
| 3    | y_velocity                | 机体竖直方向线速度 |
| 4    | body_angle                | 机体俯仰/倾斜角度（例如与水平面的夹角） |
| 5    | angular_velocity          | 机体角速度 |
| 6    | left_support_contact      | 左侧支撑/起落架是否接触目标垫，接触为 1.0，否则 0.0 |
| 7    | right_support_contact     | 右侧支撑/起落架是否接触目标垫，接触为 1.0，否则 0.0 |

## 4. 动作空间 action_space
- type: `Discrete(4)`（共 4 种动作）
- 每个动作的含义：

| action id | 名称                         | 含义 |
|-----------|------------------------------|------|
| 0         | no_engine                    | 不点火，不施加任何推力 |
| 1         | left_orientation_engine      | 点燃左侧姿态调节引擎（产生一个旋转力矩） |
| 2         | main_engine                  | 点燃主引擎（产生向上的推力） |
| 3         | right_orientation_engine     | 点燃右侧姿态调节引擎（产生与动作 1 方向相反的旋转力矩） |

本质上是一个**执行器集合**：不动、左转、加推力、右转。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  body_not_awake_or_settled（机体稳定或不再活动） **有可能** 意味着成功——如果此时机体在目标垫上、接触标志均为 1、速度极小、角度接近 0，可以判定为成功着陆。但该条件本身只是“身体沉睡或稳定”，也可能在垫子外发生（坠落后静止），因此不能直接等同于成功。
- **failure-like termination**:  
  - crash_or_body_contact（除目标垫以外的碰撞/身体接触）——大概率是与其他结构或地面发生撞击，非安全着陆。  
  - horizontal_position_outside_viewport（水平位置超出视口）——机体飞出画面，显然失败。
- **ambiguous termination**: body_not_awake_or_settled 在接触标志为 0 或位置远离原点时，也可能是失败（卡在角落静止）。
- **truncation**: masked step source 中未见时间/步数截断参数，但环境可能设有最大步数（常见），本卡片不依赖该信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: (空，step 返回的 info 为空字典 `{}`)
- forbidden_or_uncertain_info_fields: 禁止假设 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等字段存在。

**结论**：必须通过 `obs` 和 `next_obs` 中的位置、速度、接触标志等信息自行推断成功或失败。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`（上一时刻观测）
- `action`（刚执行的动作）
- `next_obs`（新观测）
- `info` 中的字段（当前 info 为空，故无可用）
- `training_progress` **仅在 prompt 明确指示允许使用时才可用**，当前任务未提及，因此一律视为**禁止使用**。

**禁止使用**：
- `original_reward`（即官方原始奖励，被 mask 隐藏）
- 任何称为 `official_reward` 或从环境中直接读取的奖励值
- `info` 中不存在或未声明的任何键
- 观测中未声明的切片或编码

## 7. 可用于奖励函数的信号
可以从观测、动作和下一观测中提取以下有意义的信号：

- **position**: `next_obs[0]`（x位置）、`next_obs[1]`（y位置）——描述机体相对于目标垫的位置；距离目标越近越好。
- **velocity**: `next_obs[2]`（水平速度）、`next_obs[3]`（垂直速度）——用于鼓励低速稳定着陆，或在飞行过程中保持适当速度。
- **orientation**: `next_obs[4]`（机体角度）——目标角度应为 0（水平），角度小有利于着陆。
- **angular_velocity**: `next_obs[5]`——小角速度表明姿态稳定。
- **contact**: `next_obs[6]` 和 `next_obs[7]`——两侧接触标志。两者均为 1.0 且位置、速度接近零时，可视为成功着陆的强烈信号。
- **action/engine**: `action` 自身——可用于衡量推进器使用情况（非 0 动作代表消耗能量/燃料）。例如惩罚使用主引擎或姿态引擎。

## 8. 不确定或不可用的信号
- **显式成功标志**：无，不可从 info 获取。
- **显式失败标志**：无，不可从 info 获取。
- **crash/碰撞的具体对象或接触力信息**：不包含在观测中，无法直接获知碰撞类型，只能通过位置、接触标志间接推断。
- **剩余的步数或时间限制信息**：观测中未提供，不可用。
- **环境内部的燃料/能量计算**：未被暴露，仅能通过动作使用的引擎类型近似衡量。
- **水平或垂直方向的绝对坐标（世界坐标）**：观测给出的是相对目标垫的偏移量，只能获取相对位置，不能知道绝对范围边界。
- **任何与 map/terrain 相关的传感器信息**：观测中无，不可用。



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



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- ---
- approach_bonus + distance_penalty + progress_delta_reward + stability_penalty
- distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty
- skeleton

请选择一个**完全不同的主信号骨架**。例如如果上述列表都是 progress_delta 系列，请尝试 potential_based_shaping 或 bounded_proximity。不要重复已失败的骨架。

```
