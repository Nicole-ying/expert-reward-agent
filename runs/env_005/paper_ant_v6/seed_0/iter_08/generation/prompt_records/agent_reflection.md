# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的目标不是匹配某个已知环境，而是根据训练证据改善外部任务表现。每次只做一个可验证的修改。

# 你收到的数据（按顺序）

1. **Search objective** — 目标分数、当前分数、差距。只用于判断搜索进度，不用于反推官方公式。
2. **上一轮奖励函数代码** — 刚被训练过的 reward 源码。
3. **上一轮你的诊断** — risk 和 hypothesis 行（本次迭代修改前的诊断预测）。仅第 1 步 Q3 使用。
4. **训练反馈** — Final-policy outcome（score, len, terminated/truncated）、组件表格（episode_sum_mean 是每回合有符号累计量，active_rate 是非零触发率）、Evaluation distribution。
5. **环境事实** — 任务目标（§1）、观测空间（§3）、动作空间（§4）、终止条件（§5）。声明的 obs/action 维度是唯一可用接口。
6. **Compact expert route context** — 公式算子切换指南。
7. **历史记忆** — 迭代历史表（iter, skeleton, score, len, decision）。decision 只看 score，不代表修改质量。

# 决策流程

## 0. 信号覆盖审计（先于诊断，逐项过）

a) **终止 → 前兆**：#5 环境事实 §5 声明了哪些终止条件？#2 代码里每个终止条件都有前兆软信号吗？"高度出界"的前兆是"身体开始倾斜或下降"，不是出界本身。

b) **目标 → 进度**：#5 环境事实 §1 声明的任务目标是什么？#2 代码有没有组件直接给它梯度？如果主信号是 body_x_vel 但 agent 不走，考虑是否缺了正向引导。

c) **效率信号**：#5 环境事实 §4 动作维度 ≥ 6 且 #2 代码无 action penalty → 备选方向。若 #4 训练反馈中 forward 总量高但 score 低，优先考虑。

d) **僵尸组件**：#4 训练反馈组件表中 active_rate < 2% 的组件 → 应删除或改造为连续信号。

e) **vs-previous**：对比 #7 历史记忆中上轮和本轮的 len。若 len 从 >500 暴跌至 <50，而 decision 是 new_best → 大概率是"伪改善"（agent 不动所以不扣分），优先回退或削弱上轮新增的惩罚项。若 len 在某个历史轮次发生过断崖式暴跌（>10x）且至今未恢复，回溯到那轮新增的组件——它大概率是根因，优先修正它而非继续修补症状。

f) **一句话结论**：当前 reward 漏了什么信号？

## 1. 行为诊断

综合第 0 步结论和 #4 训练反馈：

1. **agent 在做什么？** 快速失败（term 多、len 短）？慢速徘徊（len 长、score 低）？还是刷分 exploit（forward 极高但 term 也多）？若 len 暴跌而 score 微升，考虑伪改善。

2. **干预哪个目标？** 结合第 0 步的缺口判断、组件 active_rate 和 episode_sum_mean。只干预一个目标。

3. **上轮改了什么、预测了什么？** 从 #7 历史记忆看上轮的 skeleton 变化和 decision。#3 上轮诊断中若有 risk 行 → 检查本轮数据是否验证了它。若验证了 → 先修副作用，不重复上轮操作。

## 2. 选择干预层级

**Level 1 — 尺度修复**：职责完备、数学形态合理，只是系数/阈值异常。
- `|penalty per-step| / |progress per-step| > 0.5` 且惩罚 active_rate ≈ 100% → 降系数至 0.1~0.3x。
- 若一次尺度修复后无实质改善 → 升 Level 2。

**Level 2 — 结构变换**：缺职责、active_rate 接近 0、数学形态塌缩、或第 0 步发现信号缺口。每轮只改一个组件。若第 0 步发现缺口 → 优先 add。

| 证据 | 变换 |
|---|---|
| active_rate < 5%（稀疏事件几乎不触发） | 二值条件 → 连续 bounded factor |
| 极端值支配 reward | 无界 → 有界（归一化/压缩） |
| 占据好状态即持续获奖 | 状态绝对值 → 改善量 `next - cur` |
| 约束在无关阶段妨碍探索 | 全局惩罚 → 局部门控 |
| 独立目标可互相补偿 | 加权和 → 乘积或几何平均 |
| 乘积经常塌缩为 0 | 乘积 → `(f1·f2·...) ** (1/n)` |
| proxy 提高但外部分数不升 | proxy → 对齐任务完成信号 |
| 第 0 步发现信号缺口 | **add 新组件**（优先从未使用的 obs 中选） |

**Level 3 — 重建骨架**：同骨架 ≥ 3 轮未超 target 5%、或连续 ≥ 2 轮 Level 2 未刷新 best。

# 设计校准（写代码前，必查）

1. **新惩罚系数**：目标 per-step ≤ 主信号 per-step 的 0.3x。主信号 per-step ≈ episode_sum_mean / len。
2. **hinge 阈值**：设在终止边界的 60-80% 处（如终止边界 body_z<0.2，hinge 起点 0.30-0.35）。
3. **gate 不塌缩**：在"不理想但安全"区域 gate ≥ 0.3，否则切断学习信号。
4. **单组件 ≤ 2x 主信号**：修改后任一组件的预期 per-step 绝对值不超过主信号的 2 倍。

# 代码约束

- 只用 #5 环境事实声明的 obs/action 维度和索引。
- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止 import、class、try/except、eval/exec/open。
- 平方根 `** 0.5`；指数 `2.718281828 ** exponent`。
- 每轮只改一个组件（Level 3 除外）。
- 签名 `def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`

# 输出

```markdown
# 设计理由
（改了什么组件、为什么、数学形式、系数怎么校准的）

```python
def compute_reward(...):
    ...
```

# 诊断摘要
- **audit**: （第 0 步的一句话结论）
- **evidence**: （关键数字）
- **behavior**: （agent 在做什么）
- **signal**: （缺什么或什么过强）
- **level**: Level 1/2/3 + 触发条件
- **hypothesis**: （为什么这个修改应改善）
- **risk**: （最可能的副作用）
```

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 2000.000000
- current_score: 4.444416
- gap_to_target: 1995.555584
- target_achievement_ratio: 0.222%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 2. 上一轮奖励函数代码（该轮得分: 4.444416）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]
    vz = next_obs[15]

    # 姿态健康因子（连续，保持直立）
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 前进奖励（不再使用高度 gate，保证任何存活状态下都有梯度）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 动作幅度惩罚
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    # 高度惩罚：低于 0.35 m 开始线性增大，最大惩罚量级受控
    height_penalty = -0.5 * max(0.0, 0.35 - body_z)

    # 下降速度惩罚：阻止过快坠落
    descend_penalty = -0.2 * max(0.0, -0.5 - vz)

    total_reward = forward_reward + lateral_penalty + action_penalty + height_penalty + descend_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty,
        "height_penalty": height_penalty,
        "descend_penalty": descend_penalty
    }
    return float(total_reward), components
```

# 3. 上一轮你自己的诊断（risk / hypothesis）
- **hypothesis**: 移除 gate 后，agent 在整个存活区间内都能为前进速度获得奖励，同时 height_penalty 提供恢复高度的梯度，应能延长 episode 长度并提升总得分。
- **risk**: 若高度惩罚过弱，agent 可能维持贴近 0.2 m 的低姿势前进，导致 forward_reward 增加但目标得分提升有限；必要时可上调 hinge 系数或斜率。
(第 1 步 Q3 必须回检：本轮数据是否验证了 risk？)

# 4. 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=4.444416, len=17.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-21.057211, 21.900659]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 9.357899 | 53.7% | 53.7% | 86.5% |
| action_penalty | -4.869778 | -28.0% | 28.0% | 100.0% |
| lateral_penalty | -2.321523 | -13.3% | 13.3% | 100.0% |
| descend_penalty | -0.864603 | -5.0% | 5.0% | 42.2% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
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

# 6. Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 7. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + height_penalty + lateral_penalty + posture_penalty | -2.50 | -2.50 | 0.00 | 15.70 | forward_reward=0.632 height_penalty=-0.026 lateral_penalty=-0.340 posture_penalty=-1.948 | new_best |
| 2 | forward_reward + height_penalty + lateral_penalty | -14.41 | -2.50 | -11.90 | 938.85 | forward_reward=0.552 height_penalty=-0.001 lateral_penalty=-0.093 | no_meaningful_improvement |
| 3 | action_penalty + forward_reward + height_penalty + lateral_penalty | 18.41 | 18.41 | 0.00 | 20.85 | action_penalty=-0.278 forward_reward=0.336 height_penalty=-0.012 lateral_penalty=-0.157 | new_best |
| 4 | action_penalty + forward_reward + lateral_penalty | 8.92 | 18.41 | -9.50 | 17.45 | action_penalty=-0.311 forward_reward=0.333 lateral_penalty=-0.181 | unsolved_improving_continue_from_best |
| 5 | action_penalty + forward_reward + height_penalty + lateral_penalty | 2.52 | 18.41 | -15.89 | 13.75 | action_penalty=-0.323 forward_reward=0.298 height_penalty=-0.032 lateral_penalty=-0.218 | no_meaningful_improvement |
| 6 | action_penalty + descend_penalty + forward_reward + lateral_penalty | 5.70 | 18.41 | -12.71 | 11.45 | action_penalty=-0.328 descend_penalty=-0.036 forward_reward=0.259 lateral_penalty=-0.184 | no_meaningful_improvement |
| 7 | action_penalty + descend_penalty + forward_reward + height_penalty + lateral_penalty | 4.44 | 18.41 | -13.97 | 17.05 | action_penalty=-0.333 descend_penalty=-0.034 forward_reward=0.244 height_penalty=-0.014 lateral_penalty=-0.168 | unsolved_improving_continue_from_best |

```
