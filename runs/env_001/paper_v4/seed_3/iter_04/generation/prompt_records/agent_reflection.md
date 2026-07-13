# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率（旧日志中的`nonzero_rate`）。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现“跳跃、着陆、抓取”等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写“未知”，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

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
# Search objective
- target_score: 200.000000
- current_score: -15.416144
- gap_to_target: 215.416144
- target_achievement_ratio: -7.708%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -15.416144）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function with progress-based guidance and settling quality incentive.

    Components:
    - progress: rewards reducing distance to target center (0,0)
    - velocity_penalty: speed penalty gated by proximity to target
    - orientation_penalty: penalty for tilt and angular velocity
    - settling_quality: rewards dual-leg contact near target with low velocity and stable attitude
      (replaces persistent contact_bonus to prevent proxy farming without actual settling)
    """
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward: positive for moving toward target, negative for moving away
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty – active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Settling quality – rewards contact + proximity + low velocity/attitude
    #    Replaces flat contact_bonus to prevent farming without actually settling.
    #    settling_factor approaches 1.0 only when velocity, angle, and angvel are all near zero.
    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_weight = 2.0
    settling_quality = settling_weight * contact_product * proximity_factor * settling_factor

    total_reward = progress_reward + velocity_penalty + orientation_penalty + settling_quality

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'settling_quality': float(settling_quality)
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-15.416144, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-51.588635, 19.512293]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 6.258317 | 50.6% | 58.0% | 100.0% |
| velocity_penalty | -3.151573 | -25.5% | 25.5% | 100.0% |
| orientation_penalty | -2.040824 | -16.5% | 16.5% | 100.0% |
| settling_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体是一个二维飞行器，从视口顶部中心附近以随机初始速度开始运动。核心目标：以最小的引擎推力、最短的时间，到达画面中央的目标停靠平台，并在目标位置保持稳定姿态与安全接触。智能体必须学会精确导航、减速软着陆、维持水平姿态并最终使身体稳定（settled）。混淆目标：不能将“尽量少用引擎”误解为完全不使用引擎；不能把“尽快到达”与“粗暴着陆”等价；不能只优化到达而忽略着陆稳定性。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- 字段详解：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|----------------|
| 0 | x_position | 本体相对目标平台的水平坐标 | true |
| 1 | y_position | 本体相对平台高度的垂直坐标（正向可能为上方） | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 本体倾角（弧度） | true |
| 5 | angular_velocity | 角速度（弧度/单位时间） | true |
| 6 | left_support_contact | 左支撑腿/触地点接触标志（1.0接触，0.0离地） | true |
| 7 | right_support_contact | 右支撑腿/触地点接触标志 | true |

所有观测均可用于奖励设计，尤其是位置、速度、角度、接触标志。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作值 | 名称 | 含义 |
|--------|------|------|
| 0 | no_engine | 无推力，依靠当前动量滑行 |
| 1 | left_orientation_engine | 启动左姿态引擎（产生逆时针或顺时针力矩） |
| 2 | main_engine | 启动主引擎（产生垂直向上推力，可能伴随小力矩） |
| 3 | right_orientation_engine | 启动右姿态引擎（产生相反方向力矩） |

动作直接影响本体推力和姿态力矩，间接影响位置、速度、角度、角速度。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` – 当本体与平台稳定接触且速度/角速度极小，进入物理引擎的“休眠”状态。结合任务目标，若该事件发生在目标平台附近且接触成功，即为成功着陆。但该条件本身仅依赖物理状态，不检查位置是否在目标附近，因此单靠此终止可能包含非目标位置的过早休眠。

- **failure-like termination**:  
  `crash_or_body_contact` – 任何非支撑腿的身体部分接触地面或发生猛烈碰撞，通常表示硬着陆、倾覆或偏离安全姿态。  
  `horizontal_position_outside_viewport` – 水平漂移出画面边界，显然失败。

- **ambiguous termination**: 无。

- **truncation**: step 源码中未显示任何时间步截断，返回 `terminated` 且 `truncated=False`。因此没有时间上限导致的截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: `{}`（step 返回空字典）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为未提供任何信息。

终止条件仅能从 `terminated` 标志判断 episode 结束，无法直接读取 success/failure 标签。任何基于成功/失败的奖励必须从观测状态中自行推断（例如结合位置、接触和休眠）。

## 7. 可用于奖励函数的信号
- **位置信号**：`x_position` (obs[0])，`y_position` (obs[1]) – 用于计算到目标的距离、高度误差。
- **速度信号**：`x_velocity` (obs[2])，`y_velocity` (obs[3]) – 用于鼓励减速、控制着陆冲击。
- **姿态信号**：`body_angle` (obs[4])，`angular_velocity` (obs[5]) – 用于维持水平姿态、稳定角速度。
- **接触信号**：`left_support_contact` (obs[6])，`right_support_contact` (obs[7]) – 用于检测着陆成功、双腿接地、避免单腿/侧倾。
- **动作/引擎信号**：`action` – 用于惩罚引擎使用、鼓励节约。
- **其他组合**：依据连续两帧的 `obs` 与 `next_obs` 可计算位置、速度的变化，评估进展或风险。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + orientation_penalty + proximity + velocity_penalty | -19.59 | -19.59 | 0.00 | 78.25 | contact_bonus=0.003 orientation_penalty=-0.044 proximity=-2.196 velocity_penalty=-0.211 | new_best |
| 2 | contact_bonus + orientation_penalty + progress + velocity_penalty | 147.71 | 147.71 | 0.00 | 1000.00 | contact_bonus=0.241 orientation_penalty=-0.010 progress=0.013 velocity_penalty=-0.018 | new_best |
| 3 | orientation_penalty + progress + settling_quality + velocity_penalty | -15.42 | 147.71 | -163.13 | 1000.00 | orientation_penalty=-0.012 progress=0.014 settling_quality=0.614 velocity_penalty=-0.020 | no_meaningful_improvement |

```
