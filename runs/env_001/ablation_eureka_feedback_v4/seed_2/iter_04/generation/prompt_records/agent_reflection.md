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
# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: 代码无法解析 AST: invalid character '、' (U+3001) (<unknown>, line 2) (record: runs\env_001\ablation_eureka_feedback_v4\seed_2\iter_04\generation\validations\reward_v4.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
上一轮奖励函数用正向 soft_landing 奖励低速、小角度和接触，但缺乏对坠毁前危险状态的负反馈，导致策略在下降段频繁高速撞击。本轮重建主骨架：保留 **progress** 和 **fuel_cost**，将 soft_landing 替换为 **hinge 形式的垂直速度惩罚 + 角度惩罚（仅在接近地面时激活）** 和一个 **低系数接触奖励（门控）**，以提供清晰的抑制信号。惩罚阈值参考终止边界（y≈0）将安全速度设为 0.5、安全角度 0.2，门控高度为 2.0；系数经过校准：惩罚组件每步预期 ≤ 0.03，不超过主信号 progress 的 2 倍（progress 典型 per-step ≈ 0.016）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前位置
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    # 距离目标
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5

    # 1. 目标进展：向目标移动（保持线性以提供稳定梯度）
    w_progress = 1.0
    progress = w_progress * (dist_old - dist_new)

    # 2. 安全着陆惩罚（仅低空激活）
    height = ny  # y 

# Search objective
- target_score: 200.000000
- current_score: -125.271689
- gap_to_target: 325.271689
- target_achievement_ratio: -62.636%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -125.271689）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    # 距离
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5

    # 1. 目标进展
    w_progress = 1.0
    progress = w_progress * (dist_old - dist_new)

    # 2. 软着陆：靠近目标时激活，鼓励低速、正姿态、双足接触
    landing_threshold = 0.5
    gate = max(0.0, 1.0 - dist_new / landing_threshold)

    w_contact = 0.5
    contact_signal = w_contact * (n_lc + n_rc)   # 连续梯度，最大 1.0

    safe_speed = 0.2
    w_speed = 0.5
    speed_mag = (nvx**2 + nvy**2)**0.5
    speed_bonus = w_speed * max(0.0, safe_speed - speed_mag)

    safe_angle = 0.1
    w_angle = 0.5
    angle_abs = abs(n_angle)
    angle_bonus = w_angle * max(0.0, safe_angle - angle_abs)

    soft_landing = gate * (contact_signal + speed_bonus + angle_bonus)

    # 3. 燃料效率
    w_fuel = 0.02
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = progress + soft_landing + fuel_cost
    components = {
        "progress": progress,
        "soft_landing": soft_landing,
        "fuel_cost": fuel_cost
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=-125.271689, len=78.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-253.603378, 12.368148]

## Reward component values (mean per episode)
- soft_landing: 1.618251
- progress: 1.237086
- fuel_cost: -0.487000

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个2D飞行器轨迹优化任务。飞行器从视口顶部中央附近出发，受到一个初始随机力的干扰。核心目标是尽快到达中央的目标着陆板并稳定停靠（低速度、正姿态、双足接触）。次要目标是尽量节省引擎推力。智能体必须学会在有限的离散推力动作下，平顺地接近目标、减速、保持姿态稳定，并实现安全的 soft landing 接触。不应把生存、保持平衡或无关区域的探索当作主目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: `x_position`，相对于目标板的水平坐标，reward_usable: true
- obs[1]: `y_position`，相对于目标板高度的垂直坐标，reward_usable: true
- obs[2]: `x_velocity`，水平线速度，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，reward_usable: true
- obs[4]: `body_angle`，机体倾斜角度，reward_usable: true
- obs[5]: `angular_velocity`，角速度，reward_usable: true
- obs[6]: `left_support_contact`，左足接触标志（0.0/1.0），reward_usable: true
- obs[7]: `right_support_contact`，右足接触标志（0.0/1.0），reward_usable: true

所有 obs 维度均可用于奖励计算，但必须在适当上下文中使用（例如 contact 只在接近地面时才有意义）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不启动任何引擎，仅依靠惯性/重力
- action 1: `left_orientation_engine` — 启动一个侧向姿态引擎（产生旋转推力）
- action 2: `main_engine` — 启动主引擎（产生主要向上的推力，可能同时影响旋转）
- action 3: `right_orientation_engine` — 启动相反侧的姿态引擎

动作本身是离散的，不能直接输出连续推力大小。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 中的 `body_settled` 部分（如果环境以此表征成功着陆并稳定）. 任务的期望结束状态是低速、双足接触、位置在目标板附近且机体稳定，所以达到这种状态并触发“settled”应视为成功。
- failure-like termination:
    - `crash_or_body_contact` — 机体与障碍物碰撞或身体非正常触地（除目标板外的接触）
    - `horizontal_position_outside_viewport` — 水平位置超出允许边界
- ambiguous termination: `body_not_awake_or_settled` 中的 `body_not_awake` 可能表示机体翻转或失去平衡（无法再唤醒），也可能是安全停止；但描述中“not awake or settled”暗示两种情形，其中 `body_not_awake` 更可能是不良终止（如翻倒失去动力），而 `body_settled` 是成功。需根据实际环境验证。
- truncation: 无明确时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （空字典，没有字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何隐藏的 success、failure、termination_reason

**重要**：由于 info 为空且禁止推断其内容，奖励函数不能直接依赖终止原因；只能通过 `obs` 和 `next_obs` 中的状态来启发式地判断接近成功（如接近目标、低速、接触）或惩罚失败（如坠毁前的剧烈状态变化），但不能假定一定能访问终止原因。

## 7. 可用于奖励函数的信号
- position: `x_position`, `y_position`
- velocity: `x_velocity`, `y_velocity`
- orientation: `body_angle`
- angular_velocity: `angular_velocity`
- contact: `left_support_contact`, `right_support_contact`
- action/engine: 通过动作选择可推论引擎使用情况（动作 0 为 no_engine，其他为使用引擎）
- other: 无其他
```
