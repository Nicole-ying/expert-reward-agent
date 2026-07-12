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
- target_score: 2000.000000
- current_score: -14.407533
- gap_to_target: 2014.407533
- target_achievement_ratio: -0.720%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -14.407533）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键信号
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]

    # 姿态健康因子：用指数衰减将四元数虚部平方和映射到 (0,1]
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 前进奖励（基础量 × 姿态门控）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 高度硬约束（仅在越出安全范围时激活，作为后备保护）
    height_exceed = max(0.0, 0.2 - body_z) + max(0.0, body_z - 1.0)
    height_penalty = -10.0 * height_exceed

    total_reward = forward_reward + lateral_penalty + height_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "height_penalty": height_penalty
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-14.407533, len=938.850000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-382.488862, 71.997505]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1191.718771 | 94.1% | 94.1% | 96.4% |
| lateral_penalty | -74.461135 | -5.9% | 5.9% | 99.5% |
| height_penalty | -0.021792 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务主目标是控制一个3D四足机器人稳定地向前行走或奔跑（持续前进运动），同时保持身体直立并确保身体高度始终处于健康范围（0.2m–1.0m）。  
次要目标可能是降低能耗（动作幅度控制在合理范围内）以及维持平稳的运动姿态（避免剧烈翻滚或抖动）。  
不应将单纯保持静止直立或仅避免摔倒作为核心目标——必须持续产生正向的前进速度。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float（通常float64或float32，不指定具体）
- 所有维度均可用于奖励设计（reward_usable: true），除非特别说明。

逐维含义：

| 索引 | 名称                    | 含义                                | reward_usable | 备注 |
|------|-------------------------|-------------------------------------|---------------|------|
| 0    | body_z                  | 机体垂直高度                        | true          | 用于生存/高度约束 |
| 1    | quat_w                  | 机体姿态四元数实部 w                | true          | 用于计算直立程度 |
| 2    | quat_x                  | 四元数虚部 x                        | true          | 用于计算直立程度 |
| 3    | quat_y                  | 四元数虚部 y                        | true          | 用于计算直立程度 |
| 4    | quat_z                  | 四元数虚部 z                        | true          | 用于计算直立程度 |
| 5    | joint_1_angle           | 髋关节1角度                         | true          | 可不直接用于奖励 |
| 6    | joint_2_angle           | 踝关节1角度                         | true          | 可不直接用于奖励 |
| 7    | joint_3_angle           | 髋关节2角度                         | true          | 可不直接用于奖励 |
| 8    | joint_4_angle           | 踝关节2角度                         | true          | 可不直接用于奖励 |
| 9    | joint_5_angle           | 髋关节3角度                         | true          | 可不直接用于奖励 |
| 10   | joint_6_angle           | 踝关节3角度                         | true          | 可不直接用于奖励 |
| 11   | joint_7_angle           | 髋关节4角度                         | true          | 可不直接用于奖励 |
| 12   | joint_8_angle           | 踝关节4角度                         | true          | 可不直接用于奖励 |
| 13   | body_x_velocity         | 世界坐标系下机体前向速度 (x)        | true          | 主前进信号 |
| 14   | body_y_velocity         | 世界坐标系下机体侧向速度 (y)        | true          | 方向偏离惩罚 |
| 15   | body_z_velocity         | 机体垂直速度                        | true          | 稳定性惩罚 |
| 16   | body_roll_velocity      | 机体滚转角速度                      | true          | 稳定性惩罚 |
| 17   | body_pitch_velocity     | 机体俯仰角速度                      | true          | 稳定性惩罚 |
| 18   | body_yaw_velocity       | 机体偏航角速度                      | true          | 方向控制/偏航惩罚 |
| 19   | joint_1_velocity        | 髋关节1角速度                       | true          | 运动平滑/能量惩罚 |
| 20   | joint_2_velocity        | 踝关节1角速度                       | true          | 同上 |
| 21   | joint_3_velocity        | 髋关节2角速度                       | true          | 同上 |
| 22   | joint_4_velocity        | 踝关节2角速度                       | true          | 同上 |
| 23   | joint_5_velocity        | 髋关节3角速度                       | true          | 同上 |
| 24   | joint_6_velocity        | 踝关节3角速度                       | true          | 同上 |
| 25   | joint_7_velocity        | 髋关节4角速度                       | true          | 同上 |
| 26   | joint_8_velocity        | 踝关节4角速度                       | true          | 同上 |

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- 连续值，每维范围 [-1.0, 1.0]，代表关节扭矩。

| 维度索引 | 名称             | 含义                   | 备注 |
|----------|------------------|------------------------|------|
| 0        | hip_1_torque     | 第1髋关节扭矩          |      |
| 1        | ankle_1_torque   | 第1踝关节扭矩          |      |
| 2        | hip_2_torque     | 第2髋关节扭矩          |      |
| 3        | ankle_2_torque   | 第2踝关节扭矩          |      |
| 4        | hip_3_torque     | 第3髋关节扭矩          |      |
| 5        | ankle_3_torque   | 第3踝关节扭矩          |      |
| 6        | hip_4_torque     | 第4髋关节扭矩          |      |
| 7        | ankle_4_torque   | 第4踝关节扭矩          |      |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 不存在明确的成功终止，任务无到达目标、无指定终点。
- failure-like termination:
  1. 身体高度超出健康范围 (0.2, 1.0)，包括摔倒（过低）或异常跃起（过高）。
  2. 任何状态量变为 NaN 或 inf。（物体飞出、数值不稳定）
- ambiguous termination: 无。
- truncation: 达到时间限制（time_limit_reached）截断，非成功非失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
  （环境仅返回 terminated boolean，但该变量不传入奖励函数；info 字典为空，无法从中读取成功或失败标志。）
- allowed_info_fields: [] （info 为空，无可用字段）
- forbidden_or_uncertain_info_fields:
  - official reward terms (reward_forward, reward_ctrl, reward_contact, reward_survive) —— 已明确屏蔽
  - x_position, y_position, distance_from_origin —— 不可用
  - 任何接触信息 —— 不可用

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
| 1 | forward_reward + height_penalty + lateral_penalty + posture_penalty | -2.50 | -2.50 | 0.00 | 15.70 | forward_reward=0.632 height_penalty=-0.026 lateral_penalty=-0.340 posture_penalty=-1.948 | new_best |
| 2 | forward_reward + height_penalty + lateral_penalty | -14.41 | -2.50 | -11.90 | 938.85 | forward_reward=0.552 height_penalty=-0.001 lateral_penalty=-0.093 | no_meaningful_improvement |

```
