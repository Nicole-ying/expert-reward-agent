# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 决策流程

按顺序完成。如果 Formula Operator Reference 在输入中可用，参考其中的算子定义和切换指南来选择数学形式。

## 1. 行为诊断（3 个必答问题）

1. **这个agent发生了什么？** 用 terminated/truncated 比例、episode_length、score_range 和组件 active_rate 做行为推断（快速失败 / 慢速徘徊 / 刷分exploit）。
2. **哪个目标最值得干预？** 结合组件数学形态、episode_sum_mean、active_rate 和外部score判断。不要把数值占比直接写成因果贡献。
3. **我之前改了什么？** 从 Agent Memory 检查上一轮动作、预测和实际效果。如果上次改了A但得分没有实质变化，这次不要再次修改A。

一次只选择一个干预目标。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

## 2. 选择干预层级

判断标准：职责基本完备、符号与数学形态合理，只是相对尺度异常 → Level 1。必要职责缺失、active_rate接近0、数学形态使反馈塌缩，或proxy与外部任务明显错位 → Level 2。

### Level 1：尺度修复

只调整一个组件的系数，其他保持不变。
- `|penalty/progress| > 0.5` 且惩罚的 active_rate ≈ 100%：优先降系数，目标调到 0.1~0.3。
- 若一次尺度修复后尺度异常已消失但外部行为没有实质改善，不继续反复调同一系数，转 Level 2。

### Level 2：数学结构变换

每轮只改变一个目标组件。改变形态时同步设置与新值域匹配的系数。

| 证据 | 变换方向 | 要点 |
|---|---|---|
| 事件几乎不触发（active_rate < 5%） | 稀疏→连续 proxy | 二值条件换为连续 bounded factor，确保每步有梯度 |
| 极端值支配奖励 | 无界→有界 | 归一化/压缩，使极端值不再支配 |
| 占据好状态即可持续获奖 | 状态值→改善量 | `next - current` 替代绝对值，停留不再积累收益 |
| 约束在无关阶段妨碍探索 | 全局→局部门控 | gate只在危险区生效，安全区不干预 |
| 独立目标可互相补偿 | 加权和→联合满足 | 改为乘积或几何平均，防止单项刷分 |
| 乘积经常塌缩为0 | 乘积→几何平均 | `(f1*f2*...)**(1/n)` 替代裸乘积 |
| 持续事件被重复领取 | 持续→转移事件 | 只在状态转移时发放，而非每步 |
| proxy提高但外部分数不升 | proxy→对齐任务完成 | 调整proxy使其与外部reward同向 |
| 稠密proxy形成中等分平台 | 全程proxy→局部任务信号 | 在接近完成时才给强信号 |
| 复杂耦合无法诊断 | 耦合→少量直接组件 | 拆成2-3个独立可解释组件 |

### Level 3：重建主骨架

满足任一条件时停止局部修补：
- 同一骨架家族已迭代≥3轮，且历史最佳得分仍未超过target的5%；
- 同一结构家族连续≥2轮未刷新best，且至少做过一次 Level 2；
- Level 2 改变数学形态后没有实质改善。

Level 3 可以更换主信号框架或重新组合少量组件。参考 Formula Operator Reference 中的算子，但最终设计由环境事实和训练证据驱动。

# 代码约束

- 遵守证据边界中的所有信号约束，不得发明 obs/next_obs/action/info 字段或切片维度。
- 第一个 Python code block 只能包含一个完整的 `compute_reward` 函数；不要写 import、class、try/except 或额外函数，不要使用 self。
- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止 eval/exec/open。
- 需要平方根时使用 `** 0.5`，禁止 import numpy。需要指数形式时使用 `2.718281828 ** exponent`。
- 除 Level 3 重建外，每轮只修改一个目标组件，不顺带调整其他组件。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`；components 只放总公式中直接出现的奖励组件。

# 设计校准（写代码前检查）

根据上面 Level 2 表选择的变换方向，确定具体数学形式和参数后，检查以下 4 条：

1. **新惩罚的系数**：估算目标行为的 per-step 量级。如果主进展信号的 per-step 典型值约 X，新惩罚的 per-step 预期值应控制在 0.1X ~ 0.3X。
2. **hinge/边界惩罚的阈值**：阈值应设在终止边界的 60-80% 处（如终止边界是 body_z<0.2，hinge 起点可设在 0.30-0.35），给 agent 足够的警告步数来纠正。
3. **gate 的衰减区间**：确保 gate 在"仍安全但不理想"的区域不低于 0.3，否则等于切断了该区域的所有学习信号。
4. **单组件不超过主信号的 2x**：如果修改后任何单组件的预期 per-step 绝对值超过主进展信号的 2 倍，减小其系数。

# 输出格式

先输出代码和设计理由，再输出诊断摘要。

```markdown
# 设计理由
（2-4 句话：我改了什么组件、为什么、选了什么数学形式、系数/阈值是怎么校准的）

```python
def compute_reward(...):
    ...
```

# 诊断摘要
- **evidence**: （1 句：关键数字）
- **behavior**: （1 句：agent 在做什么）
- **signal**: （1 句：缺什么或什么过强）
- **level**: Level 1/2/3，触发条件
- **hypothesis**: （1 句：为什么这个修改应改善行为）
- **risk**: （1 句：最可能的副作用）
```

```

## User Prompt

```markdown
# Search objective
- target_score: 200.000000
- current_score: -138.372823
- gap_to_target: 338.372823
- target_achievement_ratio: -69.186%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -138.372823）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # 进步奖励：向目标靠近
    progress = prev_dist - dist
    progress_reward = 2.0 * progress

    # 速度惩罚：高速时惩罚，接近目标时放大
    vel_gate = 1.0 / (1.0 + 3.0 * dist)
    velocity_penalty = -5.0 * (speed**2) * vel_gate

    # 姿态惩罚
    orientation_penalty = -0.1 * (angle**2) - 0.05 * (av**2)

    # 接触奖励：双接触 + 低速 + 竖直
    contact_bonus = 10.0 * lc * rc * max(0.0, 1.0 - speed) * max(0.0, 1.0 - abs(angle))

    total = progress_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'progress_reward': progress_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=-138.372823, len=668.300000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-213.278637, -47.881227]

## Reward component values (mean per episode)
- velocity_penalty: -34.067878
- progress_reward: 0.611853
- orientation_penalty: -0.248246
- contact_bonus: 0.000000

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该环境是一个 2D 飞行器／着陆器的轨迹优化任务。主体从视口顶部中央附近起始，并受到一个随机初始力。主要目标是**以最短时间**将主体移动到视口中心的着陆平台上，并**稳定停靠**（即速度归零、姿态平稳、支撑接触）。次要目标是**尽可能少用引擎推力**，以节省燃料。智能体需要学会：
- 朝目标移动（x,y 位置逼近 0）；
- 在接近目标时减速；
- 保持竖直姿态（body_angle ≈ 0）；
- 最终让左、右支撑腿同时接触目标平台（left_support_contact=1 且 right_support_contact=1）且主体静止。

**不要混淆**：仅快速到达而不稳定停靠不算成功；仅节省燃料而不及时到达也不算成功。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（接触标志为 0.0/1.0）
- obs[0]: x_position，相对于目标着陆平台中心的水平坐标，reward_usable: true
- obs[1]: y_position，相对于着陆平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，无推力，主体自由漂移
- action 1: left_orientation_engine，启动一个姿态引擎（产生偏转力矩和微小推力）
- action 2: main_engine，启动主引擎（产生沿机体方向的推力）
- action 3: right_orientation_engine，启动另一侧姿态引擎（相反偏转力矩）

注意：动作是离散的，每个时间步只能选择一种引擎或不做任何操作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能表示主体已静止并停靠在平台上（左右支撑接触均为 1，速度很小，姿态稳定）。这很可能就是成功到达并停靠的结局。
- failure-like termination: crash_or_body_contact（碰撞或不当身体接触）、horizontal_position_outside_viewport（水平位置超出视口范围）
- ambiguous termination: body_not_awake_or_settled 在没有支撑接触时也可能因能量耗尽而“沉睡”，此时可视为失败；但由于环境没有提供更细的 info，我们只能依据接触和位置判断。
- truncation: 无明确时间截断，但可以限制最大步数，本环境未说明。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）
- explicit_failure_flag_available: false（无 failure 字段）
- allowed_info_fields: {} （空字典，无可用的额外字段）
- forbidden_or_uncertain_info_fields: 所有未在 step 源码中出现的字段，例如 "success", "failure", "reason" 等均不可用。终止信号仅由 terminated 布尔值给出，且 masked step source 未提供终止原因分离信息，因此我们不能直接依赖 terminated 的标签来区分成功/失败。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（均相对于着陆平台）
- velocity: x_velocity, y_velocity
- orientation: body_angle, angular_velocity
- contact: left_support_contact, right_support_contact
- action/engine: 当前动作（可用来惩罚引擎使用）
- other: 从 obs 可计算的状态量（如到目标的距离、速度模、角度绝对值等）

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 39.50 | 39.50 | 0.00 | 713.90 | contact_bonus=2.672 orientation_penalty=-0.020 proximity_reward=-0.073 velocity_penalty=-0.017 | new_best |
| 2 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 170.40 | 170.40 | 0.00 | 471.50 | contact_bonus=0.512 orientation_penalty=-0.018 proximity_reward=-0.057 velocity_penalty=-0.012 | new_best |
| 3 | contact_reward + orientation_penalty + proximity_reward + velocity_penalty | -9.34 | 170.40 | -179.74 | 1000.00 | contact_reward=0.331 orientation_penalty=-0.019 proximity_reward=-0.063 velocity_penalty=-0.015 | no_meaningful_improvement |
| 4 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 109.84 | 170.40 | -60.55 | 842.75 | contact_bonus=0.030 orientation_penalty=-0.014 proximity_reward=-0.076 velocity_penalty=-0.019 | no_meaningful_improvement |
| 5 | contact_proximity + orientation_penalty + proximity_reward + velocity_penalty | 33.65 | 170.40 | -136.75 | 850.95 | contact_proximity=0.207 orientation_penalty=-0.016 proximity_reward=-0.063 velocity_penalty=-0.016 | unsolved_high_achievement_continue_from_best |
| 6 | contact_bonus + orientation_penalty + proximity_reward + velocity_penalty | 34.67 | 170.40 | -135.73 | 321.25 | contact_bonus=5.363 orientation_penalty=-0.019 proximity_reward=-0.069 velocity_penalty=-0.014 | no_meaningful_improvement |
| 7 | orientation_penalty + proximity_reward + success_reward + velocity_penalty | -115.04 | 170.40 | -285.44 | 68.45 | orientation_penalty=-0.031 proximity_reward=-0.486 success_reward=0.067 velocity_penalty=-0.037 | no_meaningful_improvement |
| 8 | contact_bonus + orientation_penalty + progress_reward + velocity_penalty | -138.37 | 170.40 | -308.77 | 668.30 | contact_bonus=3.203 orientation_penalty=-0.010 progress_reward=0.006 velocity_penalty=-0.331 | unsolved_high_achievement_continue_from_best |

```
