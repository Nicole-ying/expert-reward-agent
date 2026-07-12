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
错误信息：Reward v4 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs/env_002/chat_gen_pro_reflect_v1/seed_1/iter_04/generation/validations/reward_v4.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
当前angle_penalty触发率为0，说明摔倒不以持续角度超限的形式出现；问题更可能是在崎岖路段缺乏速度抑制导致失控。我在gated_forward_reward的gate中引入地形粗糙度因子：用10个LIDAR读数的标准差度量地形崎岖度，乘以 `1/(1 + lidar_std)` 衰减速度奖励。这样崎岖时前进信号变弱，鼓励agent减速以保持稳定，但平坦时与原来一致。系数选择让粗糙度因子在平坦段接近1，在std=1时降至0.5，保持适度信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 信号提取
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]
    lidar = obs[14:24]  # 10个值

    # 组件1: 前进奖励（线性，仅正向）
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # 组件2: 平衡+地形门控
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))

    # 地形粗糙度因子: 基于LIDAR标准差
    lidar_mean = sum(lidar) / 10.0
    lidar_var = sum((x - lidar_mean) ** 2 for x in lidar) / 10.0
    lidar_std = lidar_var ** 0.5
    terrain_factor = 1.0 / (1.0 + lidar_std)  # std=0时=1.0, std增大时缓慢衰减

    # 综合门控: 取最保守值
    balance_gate = min(angle_factor, angular_velocity_factor)
    gate = min(balance_gate, terrain_factor)
    gated_forward_reward = forward_reward * gate

    # 组件3: 能耗惩罚
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # 组件4: 额外平衡惩罚（保留，但历史触发率为0，不影响）
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 20条episode全部terminated，长度均值913，得分范围[-58,300]，angle_penalty触发率0%，gated_forward_reward占signed_share 96.6%，说明某些episode摔倒导致大幅负分。
- **behavior**: agent在崎岖路段可能因未减速而失控摔倒，拉低平均分数。
- **signal**: 缺乏对前方地形的感知信号，无法在危险路段提前抑制速度。
- **level**: Level 2（数学结构变换，在门控中引入地形粗糙度因子）
- **hypothesis**: 地形粗糙度门控使agent在崎岖处减速，降低摔倒概率，从而提升最低分并提高平均得分。
- **risk**: 速度过度抑制可能导致探索缓慢，损失极致速度得分；若LIDAR标准差常态偏高，整体速度下降会拉低平均分。

# Search objective
- target_score: 300.000000
- current_score: 213.705914
- gap_to_target: 86.294086
- target_achievement_ratio: 71.235%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 213.705914）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 信号提取 ==========
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]

    # ========== 组件 1: 前进奖励 (线性，仅正向) ==========
    # 使用线性速度，避免平方对高速的过度奖励
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # ========== 组件 2: 平衡约束 (soft_health_gate) ==========
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))
    balance_gate = min(angle_factor, angular_velocity_factor)
    gated_forward_reward = forward_reward * balance_gate

    # ========== 组件 3: 能耗惩罚 (quadratic_penalty) ==========
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # ========== 组件 4: 额外平衡惩罚 (hinge) ==========
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    # ========== 总奖励 ==========
    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=213.705914, len=913.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-58.217024, 300.779514]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_reward | 400.811413 | 96.6% | 96.6% | 99.3% |
| energy_penalty | -14.212596 | -3.4% | 3.4% | 100.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：任务不是到达某个特定目标点，也不是单纯保持平衡，而是持续前进与能耗之间的权衡。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，身体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，身体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形的距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- continuous: true
- bounds: [-1.0, 1.0] per joint
- action[0]: hip_torque_leg1，腿1髋关节施加的扭矩
- action[1]: knee_torque_leg1，腿1膝关节施加的扭矩
- action[2]: hip_torque_leg2，腿2髋关节施加的扭矩
- action[3]: knee_torque_leg2，腿2膝关节施加的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（成功到达地形终点）
- failure-like termination: body_fallen_over（摔倒，任务失败）
- ambiguous termination: 无
- truncation: 无（truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（terminated=True 但无法区分是成功还是失败）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: 无（info 字典为空 {}）
- forbidden_or_uncertain_info_fields: 无（但无法从 terminated 中区分成功/失败）

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 LIDAR 间接推断前进距离
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3])
- orientation: hull_angle (obs[0]), hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action[0..3]（扭矩值，可用于能耗惩罚）
- other: 关节角度/速度 (obs[4..7], obs[9..12])，LIDAR (obs[14..23])
```
