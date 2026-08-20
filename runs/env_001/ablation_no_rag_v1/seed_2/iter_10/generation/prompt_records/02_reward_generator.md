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
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

以下原则适用于所有环境和任务类型。expert_reward_context.md 中的骨架是奖励设计原语、风险提示和参考起点，不构成封闭候选集合。你可以直接采用、组合、变形这些原语，也可以根据任务目标和可用环境信号提出新的数学结构。

## 原则 1：信号可用性优先

- 先检查 environment_card.md 和 expert_reward_context.md 中声明的可用信号（obs indices、info 字段、termination 信息）。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号（如只在特定区域触发的 bonus）触发率过低时等于摆设。
- 连续函数（线性、bounded、乘积）比离散条件更有利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应在数值上无条件压制任务驱动力；具体尺度必须结合其触发频率、数学形态和预期行为判断，不能套用固定比例阈值。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同的时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号（如 distance 和 progress_delta），它们会产生重复梯度。
- 不要让惩罚项压制探索——过于严厉的姿态/速度约束可能导致 agent 不敢行动。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价（energy_penalty、time_penalty），agent 应先学会完成任务再优化效率。
- 复杂门控（gated_reward）、动态课程（dynamic curriculum）默认后续迭代再加入。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。例如：只奖励接近目标可能导致在目标附近震荡而不完成；只奖励接触可能诱导 agent 反复轻触地面。
- 如果使用 contact 或类似离散事件，必须和其他连续条件组合，不能单独作为成功信号。

## 原则 7：最小设计

- reward_v1 推荐使用 2~4 个组件，每个必须有明确角色，不能为了显得完整而堆叠。
- 信号多的时候要主动取舍——不是所有可用信号都需要放进 v1。优先选择对任务目标最核心的信号。
- 把 expert_reward_context.md 中按任务路由推荐的骨架作为思考面，而不是允许列表。最终设计由任务目标、可用信号和预期策略行为决定，不要求组件必须对应某个 skeleton_id。
- components dict 里只放 total_reward 公式中等号右边的变量。如 `total_reward = A + B + C`，则 components 只有 A、B、C。不要把 `total_reward` 放进 components，也不要把 A 的计算中间变量（如 A = a1+a2+a3 中的 a1）放进 components。

# role-based component budget

以上原则的具体落地方式：v1 推荐使用 2~4 个组件，按以下角色组织。检索骨架只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent "做什么能得分"。主信号的特征：
- 每步都有梯度（稠密），不是只在终点或特定事件触发
- 与任务目标直接相关（靠近目标、前进、保持存活等）
- 在策略学习中承担主要任务驱动作用；它不一定是每一步数值最大的正向项
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`

主信号可以来自检索骨架，也可以是对多个设计原语的组合、变形或基于环境事实构造的新形式。不要为了匹配骨架名称而牺牲任务语义。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/平滑约束。** 如果任务需要控制速度、姿态、动作抖动等，可以加入轻量惩罚项。约束的角色是"方向盘"而非"刹车"——权重应明显小于主信号。如果环境不需要（如离散动作空间不需要 action_smoothness），不加。
- **0~1 个任务完成近似信号（proxy）。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合（不是单一二值条件），且不能直接伪造 success flag。
- **0~1 个效率/动作代价。** 权重必须很小。v1 阶段 agent 应先学会完成任务，效率优化后续迭代再加入。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确的 termination_reason）
- gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）

## 避免重复

- 不要同时大权重使用两个计算同一物理量的信号（如 distance_reward 和 progress_delta_reward）。如果两个都用了，一个必须是主信号，另一个只能是小权重辅助锚点——但更建议只选一个。

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
- components 只包含被加到 total_reward 的组件（A、B、C），**不包含 total_reward 本身**。

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

# 匿名环境理解卡片

## 1. 任务目标
智能体控制一个2D飞行器，从视口顶部中心附近以随机初始力出发。  
核心目标是尽快飞行并**稳定降落在视口中央的目标垫上**。  
辅助要求：尽量减少主发动机推力消耗，保持姿态稳定，并安全接触目标垫。  
智能体需要学习接近目标、减速、调整姿态并让两个支撑脚接触目标垫，最终稳定停在垫子上。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32
- obs[0]: x_position – 水平位置（相对于目标垫中心）
- obs[1]: y_position – 垂直位置（相对于垫子高度）
- obs[2]: x_velocity – 水平线性速度
- obs[3]: y_velocity – 垂直线性速度
- obs[4]: body_angle – 主体旋转角度
- obs[5]: angular_velocity – 角速度
- obs[6]: left_support_contact – 左脚接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact – 右脚接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine – 不启动任何发动机，仅靠惯性运动
- action 1: left_orientation_engine – 启动左侧姿态发动机（产生旋转力矩）
- action 2: main_engine – 启动主发动机（产生向下的推力，即让飞行器向上减速或悬停）
- action 3: right_orientation_engine – 启动右侧姿态发动机（产生反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  飞行器与目标垫接触且稳定下来（body_not_awake_or_settled）且没有发生 crash 或越界。  
  具体表现为：在某一 episode 中，终止条件 `crash_or_body_contact` 为 false、`horizontal_position_outside_viewport` 为 false，而且 `body_not_awake_or_settled` 为 true。  
  （即飞行器在接触目标垫后速度/角速度变得极小，进入休眠状态。）
- failure-like termination:  
  - `crash_or_body_contact` 触发，例如飞行器主体或支撑脚撞击到垫子以外的地面或其他物体。  
  - `horizontal_position_outside_viewport` 触发，飞行器飞出水平边界。
- ambiguous termination:  
  `body_not_awake_or_settled` 可能在没有接触目标垫的情况下触发（比如飞行器飘到某处后静止），这种情况可能是失败，但难以仅靠观察空间区分究竟是哪种终止。
- truncation: 无（返回值 truncated 固定为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 始终为空字典，无任何额外信号）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（info 为空），不能依赖任何来自 info 的成功/失败标志。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前观察）
- action（执行的动作）
- next_obs（下一观察）
- training_progress （仅在 prompt 明确允许时使用，当前未明确允许，建议不用）

禁止使用：
- original_reward（官方奖励已屏蔽，不得计算、拷贝或近似）
- info（为空，无可用字段）
- 任何未在上述观察空间中声明的 obs 切片或外部信号

## 7. 可用于奖励函数的信号
以下信号可直接从 obs / next_obs 中提取，供设计奖励函数使用：
- position: next_obs[0]（水平坐标），next_obs[1]（垂直坐标）  
- velocity: next_obs[2]（水平速度），next_obs[3]（垂直速度）  
- orientation: next_obs[4]（角度），next_obs[5]（角速度）  
- contact: next_obs[6]（左脚接触），next_obs[7]（右脚接触）  
- action: action 本身可用于限制发动机使用（0/2 主发动机是否使用，1/3 姿态发动机使用）

## 8. 不确定或不可用的信号
- 无法获取绝对成功/失败标签（info 为空，termination 原因未暴露具体类型）。  
- 无法区分 `body_not_awake_or_settled` 是由成功着陆引发的，还是其他静止导致的失败终止。  
- 无法获得“与目标垫接触”的直接标志（左右支撑接触可能还接触了非目标地面），因此需要组合位置、速度、接触、角速度等信号间接推断是否成功着陆。  
- 无法获得燃料消耗量、剩余时间、主体碰撞细节等任何 info 字段。  
- original_reward 被完全屏蔽，不可用。



# expert_reward_context.md

# Expert Reward Context Disabled

This run is the w/o Expert RAG ablation. Design from environment facts only.




# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 0.620

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angle_penalty + landing_quality_reward + potential_diff | 2 | 0.620 | -16.610 | unsolved |
| angle_penalty + potential_diff | 1 | -24.460 | -24.460 | unsolved |
| progress_reward + stable_landing_reward | 4 | -37.290 | -47.110 | unsolved |
| angle_penalty + potential_diff + success_bonus | 1 | -112.790 | -112.790 | unsolved |
| angle_penalty + landing_reward + potential_diff | 1 | -150.570 | -150.570 | unsolved |

## Previous interventions

- iter 2 (score=-43.990, structure=progress_reward + stable_landing_reward): selected_level:: Level 2，因为证据直接否定了当前组件的数学形态（乘积塌缩），且必要着陆引导职责缺失。 | selected_intervention:: 将 stable_landing_reward 从乘积形式改为加性组合，具体包括：
- iter 6 (score=-112.790, structure=angle_penalty + potential_diff + success_bonus): 4. `selected_level`: Level 2，触发条件为缺失必要职责（完成信号），且当前过程组件已能引导agent到达目标附近，需要将代理目标与任务完成对齐（proxy_to_completion_alignment）。 | 5. `selected_intervention`: 新增一个稀疏完成奖励组件`success_bonus` —— 当检测到双脚接触且姿态、位置、速度均满足软着陆条件时给予固定正奖励；其他所有组件保持不变。
- iter 7 (score=0.620, structure=angle_penalty + landing_quality_reward + potential_diff): 4. `selected_level`：Level 2，触发条件为sparse_to_dense——上一轮稀疏bonus失败，且任务需要基于接触、位置、速度、角度的连续完成证据。 | 5. `selected_intervention`：移除success_bonus，新增landing_quality_reward（双脚接触时对位置、速度、角度按乘积阈值输出连续值，系数5.0）。
- iter 8 (score=-150.570, structure=angle_penalty + landing_reward + potential_diff): 4. `selected_level`：Level 2 — the evidence pattern “task event almost never triggers, local feedback missing” matches `sparse_to_dense`; a structural change from a multiplicative sparse bonus to a continuous, additive pr | 5. `selected_intervention`：Change the `landing_quality_reward` component from a product of hard thresholds to a dense sum of independent proximity measures (position, velocity, angle, contact) using bounded linear decays
- iter 9 (score=-16.610, structure=angle_penalty + landing_quality_reward + potential_diff): 4. `selected_level`: Level 2 —— 证据模式为“proxy提高但外部任务不升”，触发proxy_to_completion_alignment变换。 | 5. `selected_intervention`: 将landing_reward从稠密加权和的proximity形态恢复为稀疏乘积形态(与best相同)，仅在有双脚接触时才能获得正奖励，并将系数k_landing由5.0提升至20.0以放大正确行为信号。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
