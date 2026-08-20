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
# 上一轮奖励函数代码（该轮得分: -33.974125）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraints: light penalty on speed, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * speed - 0.01 * angle - 0.005 * ang_vel

    # 3. Proximity quality: always-active bounded signal for landing-ready configuration.
    #    Rational functions 1/(1+k*x) never collapse to exactly zero,
    #    providing gradient toward low distance, low speed, and upright attitude.
    dist = d_next
    proximity_quality = 2.0 * (1.0 / (1.0 + 2.0 * dist)) * (1.0 / (1.0 + speed)) * (1.0 / (1.0 + 5.0 * angle))

    # 4. Contact bonus: additive reward for foot contact, decoupled from proximity product.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    contact_bonus = 2.0 * contact_avg

    total_reward = approach_reward + stability_penalty + proximity_quality + contact_bonus

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "proximity_quality": proximity_quality,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraint: light penalty on large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.001
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: continuous contact signal replaces binary product.
    # (left + right)/2 gives 0/0.5/1.0 gradient for partial landings.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    dist = d_next
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_avg *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-33.974125, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-68.988164, 4.597475]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_quality | 809.850599 | 99.8% | 99.8% | 100.0% |
| approach_reward | 1.065147 | 0.1% | 0.2% | 100.0% |
| stability_penalty | -0.612114 | -0.1% | 0.1% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从起点（视口顶部中心附近，有随机初始受力）移动并平稳降落到画面中央的目标平台。  
要求尽可能**快**地到达目标，同时**尽量少使用引擎推力**，并在接触时保持**姿态稳定**和**安全接触**。  
核心目标：到达目标平台并稳定停驻。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64（默认连续值）
- obs[0]：x_position —— 相对于目标平台的水平坐标  
- obs[1]：y_position —— 相对于平台高度的垂直坐标  
- obs[2]：x_velocity —— 水平线速度  
- obs[3]：y_velocity —— 垂直线速度  
- obs[4]：body_angle —— 机体倾角  
- obs[5]：angular_velocity —— 角速度  
- obs[6]：left_support_contact —— 左脚接触标志（1.0/0.0）  
- obs[7]：right_support_contact —— 右脚接触标志（1.0/0.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0：no_engine —— 无推力
- action 1：left_orientation_engine —— 启动左侧姿态引擎（产生角速度/微小推力）
- action 2：main_engine —— 启动主引擎（产生主要上升/减速推力）
- action 3：right_orientation_engine —— 启动右侧姿态引擎（与左边方向相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 机体停止运动并稳定（即成功着陆并停驻）
- failure-like termination: 
  - `crash_or_body_contact` —— 任何非法的碰撞或身体接触（如撞到非平台区域、过猛撞击）
  - `horizontal_position_outside_viewport` —— 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 未出现在 step 中（`truncated` 返回 `False`），说明无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`info` 为空字典 `{}`）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用

## 7. 可用于奖励函数的信号
- position: next_obs[0] (水平位置误差), next_obs[1] (垂直位置/接近平台高度)  
- velocity: next_obs[2] (水平速度), next_obs[3] (垂直速度) —— 可用于惩罚硬着陆  
- orientation: next_obs[4] (倾角) —— 可用于要求平稳姿态  
- contact: next_obs[6], next_obs[7] (左右接触标志) —— 可奖励双脚稳定着地  
- action/engine: action id —— 可惩罚使用引擎推力以促进节能

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | approach_reward + soft_landing_proxy + stability_penalty | -95.25 | -95.25 | 0.00 | 68.80 | approach_reward=0.016 soft_landing_proxy=0.025 stability_penalty=-0.019 | new_best |
| 2 | approach_reward + soft_landing_proxy + stability_penalty | 149.44 | 149.44 | 0.00 | 564.95 | approach_reward=0.005 soft_landing_proxy=1.728 stability_penalty=-0.002 | new_best |
| 3 | approach_reward + soft_landing_proxy + stability_penalty | -10.51 | 149.44 | -159.95 | 1000.00 | approach_reward=0.088 soft_landing_proxy=1.673 stability_penalty=-0.002 | no_meaningful_improvement |
| 4 | approach_reward + soft_landing_proxy + stability_penalty | 136.73 | 149.44 | -12.71 | 237.45 | approach_reward=0.027 soft_landing_proxy=0.690 stability_penalty=-0.002 | same_skeleton_oscillation_fresh_restart |
| 5 | landing_reward + shaping_reward | -236.15 | 149.44 | -385.59 | 835.50 | landing_reward=0.002 shaping_reward=0.951 | no_meaningful_improvement |
| 6 | landing_reward + shaping_reward | -236.15 | 149.44 | -385.59 | 835.50 | landing_reward=0.002 shaping_reward=0.951 | no_meaningful_improvement |
| 7 | approach_reward + soft_landing_proxy + stability_penalty | 198.14 | 198.14 | 0.00 | 406.05 | approach_reward=0.003 soft_landing_proxy=1.691 stability_penalty=-0.002 | new_best |
| 8 | approach_reward + engine_penalty + soft_landing_proxy + stability_penalty | -7.39 | 198.14 | -205.53 | 1000.00 | approach_reward=0.004 engine_penalty=-0.004 soft_landing_proxy=1.661 stability_penalty=-0.002 | no_meaningful_improvement |
| 9 | approach_reward + contact_bonus + proximity_quality + stability_penalty | -33.97 | 198.14 | -232.12 | 1000.00 | approach_reward=0.002 contact_bonus=1.019 proximity_quality=0.845 stability_penalty=-0.002 | no_meaningful_improvement |

```
