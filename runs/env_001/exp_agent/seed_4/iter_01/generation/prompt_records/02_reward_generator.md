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
- 控制一个 2D 飞行器（或类车辆体），从其初始位置（视口顶部中心附近，带随机初始力）移动到画面中央的圆形目标垫上，并稳定停靠。
- 任务同时要求**尽快完成**（路径时间短）且**尽量减少发动机推力使用**（省油）。
- 理想行为：飞行器接近目标垫、减速、保持正确姿态（角度接近垂直）、平稳触垫且不再弹起或滑动。

## 2. 任务类型选择
- selected_route_id: `navigation_goal_reaching`
- confidence: high
- reason: 核心是到达一个明确的目标位置（中央目标垫）并稳定停靠，且需要路径优化（速度、姿态）。虽然包含推力最小化的子目标，但本质仍为带约束的有目标导航任务，而非纯生存平衡或多目标强化学习。

## 3. 观察空间 observation_space
- type: Box (连续值向量)
- shape: (8,)
- dtype: float32（典型值）
- obs[0]: **x_position** —— 飞行器相对于目标垫中心的水平坐标。
- obs[1]: **y_position** —— 飞行器相对于目标垫高度的垂直坐标（向上为正）。
- obs[2]: **x_velocity** —— 水平线性速度。
- obs[3]: **y_velocity** —— 垂直线性速度。
- obs[4]: **body_angle** —— 飞行器本体的倾斜角（弧度，0 表示垂直向上）。
- obs[5]: **angular_velocity** —— 角速度。
- obs[6]: **left_support_contact** —— 左支撑脚/腿与地面（或垫子）接触标志（1.0 表示接触）。
- obs[7]: **right_support_contact** —— 右支撑脚/腿与地面接触标志（1.0 表示接触）。

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: **no_engine** —— 不开启任何引擎，飞行器仅受重力和惯性影响。
- action 1: **left_orientation_engine** —— 点燃左侧姿态调整引擎，产生逆时针旋转力矩（通常使角速度正向增加）。
- action 2: **main_engine** —— 点燃主引擎，沿飞行器当前朝向方向产生推力，使飞行器加速。
- action 3: **right_orientation_engine** —— 点燃右侧姿态调整引擎，产生顺时针旋转力矩（通常使角速度负向增加）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 当飞行器长时间无运动（物理引擎判定为 asleep）或已在地面稳定静止时触发。如果此时飞行器位于目标垫上且姿态合理，则可视为成功着陆；但若停在错误位置（例如视口边缘地面）也属于该终止，因此单独看并不可靠。
- **failure-like termination**:  
  `crash_or_body_contact` —— 飞行器主体（非支撑脚）触碰地面或其它障碍，代表坠毁；  
  `horizontal_position_outside_viewport` —— 水平位置越界，代表飞出允许区域。
- **ambiguous termination**:  
  `body_not_awake_or_settled`（如上），需要结合位置信息（obs[0], obs[1]）来判断是否真正成功。
- **truncation**:  
  源码中返回的第四个值为 `False`，说明环境没有设置时间上限截断（或未显式给出），但可能的 `training_progress` 参数未在此环境中引入。因此 **无显式 truncation**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  原因：`step` 返回的 info 为 `{}`，没有任何 `success`、`failure` 或 `termination_reason` 字段。
- explicit_failure_flag_available: false  
  同上。
- allowed_info_fields: 无（info 为空字典，所以无可用字段）。
- forbidden_or_uncertain_info_fields:  
  - `original_reward` / `official_reward` (禁止使用)  
  - 任何未在观察空间或明确授权中出现的键（例如假定存在的 `"success"` 等）均为危险且不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观察的 8 维向量）
- `action`（当前执行的动作 ID）
- `next_obs`（下一时刻的 8 维向量）
- `info` 中**明确允许的字段**（当前环境下无任何允许字段）
- `training_progress` **仅在 prompt 明确要求使用时可用**（本环境未提供，禁止使用）

禁止使用：
- `original_reward`（已被掩码，严禁以任何方式利用或近似反推）
- `official_reward` 或任何变体
- 未声明的 `info` 字段
- 未声明的 `obs` 切片含义（例如不能假设 obs[6] 和 obs[7] 之外还有其他接触信息）

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` 和 `next_obs[1]` 提供相对于目标垫的水平与垂直距离，可直接用于距离导向奖励。
- **velocity**: `next_obs[2]` 和 `next_obs[3]` 可用于鼓励在接近目标时减速，或惩罚高速撞击。
- **orientation**: `next_obs[4]` 和 `next_obs[5]` 可用于惩罚倾斜角过大（不垂直）或高速旋转。
- **contact**: `next_obs[6]` 和 `next_obs[7]` 表示支撑脚是否触地，可用于检测着陆状态，需结合位置判断是否为成功着陆。
- **action/engine**: `action` 可用于惩罚引擎使用（动作1,2,3）以达成省油目标。

## 8. 不确定或不可用的信号
- **explicit success / failure flags**: info 完全为空，不能直接判定任务成功与否，必须通过位置、速度、接触等自身信息推断。
- **original_reward**: 掩码不可用。
- **training_progress**: 未提供，不可用。
- **任何未声明的物理量**（如剩余燃料、力矩大小）均不可假设存在。



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
```
