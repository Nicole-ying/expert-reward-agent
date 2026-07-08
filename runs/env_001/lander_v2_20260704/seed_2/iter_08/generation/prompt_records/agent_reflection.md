# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率（旧日志中的`nonzero_rate`）。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。

# 唯一决策流程

按顺序完成，不能因知识库或工具命中某个变换就跳级。

## 1. 行为与历史诊断

拿到训练反馈后，必须先明确回答原有三个问题：

1. **这个agent发生了什么？** episode相对正常长度明显偏短，可能在快速失败；episode很长但得分差，可能在徘徊；得分已经好但某组件尺度异常，可能存在exploit，必须用行为证据验证。
2. **哪个组件最值得干预？** 结合组件数学形态、episode_sum_mean、signed_share、magnitude_share、active_rate、外部score和episode_length判断，不得把数值占比直接写成因果贡献。
3. **我之前改了什么？** 从Agent Memory检查上一轮动作、预测和实际效果。如果上次改了A但得分没有实质变化，这次不要再次修改A。

随后确认当前失败是否来自该组件或某个缺失职责，一次只选择一个干预目标。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

如果必要职责缺失、active_rate接近0、数学形态使反馈塌缩，或proxy与外部任务明显错位，进入Level 2。若职责基本完备、符号与数学形态合理，只是相对尺度异常，先进入Level 1。

## 3. 选择干预层级

### Level 1：尺度修复

适用条件：组件职责、符号和数学形态合理，证据主要表明单个组件过强或过弱。

- 只调整一个组件的系数，其他组件保持不变。
- 对同为逐步稠密信号、且progress明确承担主引导职责的早期学习阶段，旧实验中的`ratio_to_progress`思想仍然适用：`|penalty/progress| > 0.5`可作为优先检查并尝试降系数的经验触发器；`penalty/progress ≈ 0.1`可作为轻约束起点。
- 上述比例是v6实验中的有效经验先验，不是因果结论或通用硬阈值；事件bonus、持续状态奖励和正负抵消严重的差分均值不能直接套用。
- 若一次明确的尺度修复后，尺度异常已经消失但外部行为没有实质改善，不继续反复调同一系数，转Level 2。

### Level 2：有方向的数学结构变换

适用条件：必要信号缺失/不可达，或证据直接否定当前数学形态，或Level 1已修复尺度但策略仍失败。每轮只改变一个目标组件；改变该组件形态时同步设置与新值域匹配的系数，仍算一次组件干预。

| 证据模式 | 结构变换 | 下一轮验证 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | sparse_to_dense：稀疏事件→连续过程证据 | active_rate及外部表现改善，且不产生proxy徘徊 |
| 极端值支配奖励 | unbounded_to_bounded：无界→归一化有界 | 极端轨迹支配下降，得分方差下降 |
| 占据好状态即可持续获奖 | state_to_improvement：状态值→状态改善量/有效势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | global_to_local_gate：全局→阶段相关/局部门控 | 早期探索与局部约束同时改善；先确认不是单纯尺度过强 |
| 独立目标可互相补偿 | independent_to_joint：加权和→联合满足 | 单项刷分减少，必要条件共同改善 |
| 多个小因子相乘导致塌缩 | product_to_noncollapsing_joint：乘积→几何平均/软最小/门控和 | 非零反馈增多且联合约束保留 |
| 持续事件被重复领取 | persistent_to_transition_event：持续状态→有效状态转移 | 重复积累下降，外部完成保持或改善 |
| proxy提高但外部任务不升 | proxy_to_completion_alignment：代理目标→任务完成对齐 | proxy与外部分数重新同向 |
| 复杂耦合无法诊断 | coupled_to_diagnostic_components：耦合→少量直接组件 | 组件可解释并形成单一干预假设 |
| 稠密proxy形成中等分平台 | dense_to_task_event：全程proxy→局部/转移任务信号 | 刷新best，完成相关行为增加 |

常用数学性质：二值稀疏条件信用分配困难；连续乘积表达联合满足但可能塌缩；加权和反馈稠密但允许目标补偿；bounded函数限制极端值但输入仍需按环境尺度归一化；门控只应在证据表明“作用阶段错误”时使用。

### Level 3：重建主骨架

满足任一条件时停止局部修补：

- 同一骨架家族已迭代2轮以上，且历史最佳得分仍未超过target的25%；
- 同一结构家族连续2轮以上未刷新best，且至少做过一次Level 2；
- Level 2改变数学形态后没有实质改善；
- 同一结构家族连续3轮未刷新best且仍未达到target，即使已超过target的25%也要警惕中等分平台。

Level 3可以更换主信号框架或重新组合少量组件。expert_reward_context中的骨架是设计原语和风险提示，不是封闭候选列表；可以采用、组合、变形或基于环境事实创建新结构。

# 工具

核心Level判断必须依靠本Prompt完成。仅在根因不确定、多个Level 2变换难以区分或需要骨架细节时调用一次最相关工具：

- `search_reward_design_knowledge(query)`：检索相似失败模式和经验修复。
- `get_skeleton_detail(skeleton_name)`：查看数学形态、原理和陷阱。
- `get_reward_transformation(query)`：查看结构变换的原理、风险和验证目标。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 除Level 3重建外，每轮只修改一个目标组件，不顺带调整其他组件。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先用以下固定字段各写一句，不复述输入表格：

1. `evidence`：支持判断的外部结果、组件证据和上一轮结果；
2. `behavior_diagnosis`：策略当前的失败行为；
3. `signal_completeness`：必要职责是否完备、可达；
4. `selected_level`：Level 1、Level 2或Level 3及触发条件；
5. `selected_intervention`：唯一目标组件及具体修改；
6. `falsifiable_hypothesis`：为什么该修改应改善策略；
7. `expected_next_round`：下一轮哪些指标应如何变化；
8. `main_risk`：最可能引入的新漏洞。

然后立即输出完整Python代码。预期必须能被下一轮反馈证伪。

```

## User Prompt

```markdown
# 上一轮奖励函数代码（该轮得分: -108.687525）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: continuous negative Euclidean distance to goal
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Light stability penalty (preserved from best)
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: sum-based joint satisfaction, coefficient reduced from 0.2 to 0.1
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    sum_of_factors = prox_factor + speed_x_factor + speed_y_factor + angle_factor + contact_factor
    landing_quality = 0.1 * sum_of_factors  # Reduced from 0.2 to strengthen distance gradient

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: continuous negative Euclidean distance to goal
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Light stability penalty (preserved from best)
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: sum-based joint satisfaction
    #    Uses additive combination instead of product to prevent gradient collapse
    #    when any single factor is zero during early approach.
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    sum_of_factors = prox_factor + speed_x_factor + speed_y_factor + angle_factor + contact_factor
    landing_quality = 0.2 * sum_of_factors  # max=1.0 when all conditions perfect

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-108.687525, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.839570, -91.911322]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -66.484895 | -72.9% | 72.9% | 100.0% |
| landing_quality | 16.487447 | 18.1% | 18.1% | 100.0% |
| stability_penalty | -8.169597 | -9.0% | 9.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。智能体从靠近视口顶部中心的位置出发，并承受一个随机初始作用力。目标是**尽快飞行并稳定降落在中央目标平台上，同时尽可能节省发动机推力**。智能体需要学习：平滑接近目标区域、控制下落速度、保持机身姿态稳定，并安全地让两侧支撑腿同时接触平台。

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

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`（水平相对距离）、`obs[1]`（垂直相对距离）及 `next_obs` 对应值，可构建距离惩罚或接近奖励  
- **速度**：`obs[2]`、`obs[3]`，用于惩罚过大的线速度（软着陆要求）  
- **朝向/角速度**：`obs[4]`、`obs[5]`，用于惩罚倾斜姿态或旋转  
- **接触**：`obs[6]`、`obs[7]`，两腿接触状态可用于奖励安全着陆（两腿同时接触）或惩罚单腿/无接触  
- **动作/发动机使用**：`action` 本身可用于计算节省燃料的代价（如动作=2 主发动机时惩罚，动作=0 时奖励）  
- 上述信号均可在 `compute_reward` 中通过 `obs`、`action`、`next_obs` 安全获取。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_bonus + stability_penalty | -110.97 | -110.97 | 0.00 | 68.40 | distance_reward=-0.974 soft_landing_bonus=0.002 stability_penalty=-0.127 | new_best |
| 2 | distance_reward + landing_quality + stability_penalty | -112.96 | -110.97 | -1.99 | 68.40 | distance_reward=-0.972 landing_quality=0.010 stability_penalty=-0.127 | no_meaningful_improvement |
| 3 | distance_reward + landing_quality + stability_penalty | -108.37 | -108.37 | 0.00 | 68.45 | distance_reward=-0.973 landing_quality=0.139 stability_penalty=-0.128 | new_best |
| 4 | distance_reward + landing_quality + stability_penalty | -109.79 | -108.37 | -1.42 | 68.35 | distance_reward=-0.972 landing_quality=0.140 stability_penalty=-0.052 | same_skeleton_persistent_negative_fresh_restart |
| 5 | progress_delta + weighted_stability_penalty | -109.22 | -108.37 | -0.85 | 68.55 | progress_delta=0.161 weighted_stability_penalty=-0.385 | no_meaningful_improvement |
| 6 | distance_reward + landing_quality + stability_penalty | -33.72 | -33.72 | 0.00 | 254.80 | distance_reward=-0.922 landing_quality=0.495 stability_penalty=-0.118 | new_best |
| 7 | distance_reward + landing_quality + stability_penalty | -108.69 | -33.72 | -74.96 | 68.45 | distance_reward=-0.972 landing_quality=0.241 stability_penalty=-0.125 | no_meaningful_improvement |

```
