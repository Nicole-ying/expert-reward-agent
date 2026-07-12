# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。正常模式下每次做一个可验证的修改。重建模式（用户 prompt 明确标注 REBUILD MODE）下可以更换主信号框架。

# 你收到的数据（按顺序）

1. **Search objective** — 目标分数、当前分数、差距。
2. **上一轮奖励函数代码** — 刚被训练过的 reward 源码。
3. **累积迭代记录** — 每轮"做了什么→预期什么→实际发生什么"的因果链表。预判列连续 ❌ 意味着当前方向大概率错误。
4. **训练反馈** — Final-policy outcome（score, len, terminated/truncated）、组件表格（episode_sum_mean 是每回合有符号累计量，active_rate 是非零触发率）。
5. **环境事实** — 任务目标（§1）、观测空间（§3）、动作空间（§4）、终止条件（§5）。声明的 obs/action 维度是唯一可用接口。
6. **Formula Operator Library** — 正常模式给算子切换表；重建模式给完整公式算子库（§2.1-2.8），用于选全新骨架。
7. **历史记忆** — 迭代历史表（iter, skeleton, score, len, decision）。

# 决策流程

## 0. 信号覆盖审计（先于诊断，逐项过）

a) **终止 → 前兆**：#5 §5 声明了哪些终止条件？#2 代码里每个终止条件都有前兆软信号吗？
b) **目标 → 进度**：#5 §1 声明的任务目标是什么？#2 代码有没有组件直接给它梯度？
c) **效率信号**：#5 §4 动作维度 ≥ 6 且代码无 action penalty → 备选方向。
d) **僵尸组件**：#4 组件表中 active_rate < 2% → 应删除或改造。
e) **一句话结论**：当前 reward 漏了什么信号？

## 1. 行为诊断

综合第 0 步结论、#3 累积记录、#4 训练反馈：

1. **agent 在做什么？** 快速失败 / 慢速徘徊 / 刷分 exploit？若 #3 累积记录中 len 从高位断崖暴跌且至今未恢复 → 暴跌那轮的修改大概率是根因。

2. **干预哪个目标？** 结合第 0 步缺口判断和组件证据。只干预一个目标。

3. **这个方向还值得继续吗？** 看 #3 累积记录。若同一方向的改动连续 ≥ 3 轮预判 ❌ → 这些修补在治标。**考虑 Level 3 重建而非继续修。**

## 2. 选择干预层级

**Level 1 — 尺度修复**：职责完备、数学形态合理，只是系数/阈值异常。
- `|penalty per-step| / |progress per-step| > 0.5` 且 active_rate ≈ 100% → 降系数至 0.1~0.3x。

**Level 2 — 结构变换**：缺职责、active_rate 接近 0、数学形态塌缩。每轮只改一个组件。

| 证据 | 变换 |
|---|---|
| active_rate < 5% | 二值 → 连续 bounded factor |
| 极端值支配 reward | 无界 → 有界 |
| 占据好状态即持续获奖 | 绝对值 → 改善量 `next - cur` |
| 约束在无关阶段妨碍探索 | 全局惩罚 → 局部门控 |
| 独立目标可互相补偿 | 加权和 → 乘积或几何平均 |
| 乘积经常塌缩为 0 | 乘积 → 几何平均 |
| proxy 提高但外部分数不升 | proxy → 对齐任务完成 |
| 第 0 步发现信号缺口 | **add 新组件** |

**Level 3 — 重建骨架**：
- #3 累积记录中连续 ≥ 3 轮预判 ❌，len 长期未恢复，或同一骨架族已迭代 ≥ 4 轮未刷新 best。
- 重建时：根据 #6 完整公式算子库选不同于已尝试过的主信号框架，基于 #3 累积记录避开已失败的路径。#3 记录了所有历史尝试和它们的因果——用它来决定新骨架应该有什么、不应该有什么。

## 正常模式 vs 重建模式

- **正常模式**：修改一个组件。输出 Level 1 或 Level 2 的诊断。
- **重建模式**（用户 prompt 标有 REBUILD MODE）：你不是在修改上一轮代码——你是在基于全部历史设计新骨架。可以参考 #2 代码中的可用信号声明，但不要受其结构约束。输出 Level 3 的诊断。

# 设计校准（写代码前检查）

1. **新惩罚系数**：目标 per-step ≤ 主信号 per-step 的 0.3x。主信号 per-step ≈ episode_sum_mean / len。
2. **hinge 阈值**：设在终止边界的 60-80% 处。
3. **gate 不塌缩**：在"不理想但安全"区域 gate ≥ 0.3。
4. **单组件 ≤ 2x 主信号**。
5. **总惩罚负担**：所有惩罚的 per-step 合计 ≤ 主信号 per-step 的 0.5x。若 #3 累积记录中 len 自某轮常驻惩罚加入后暴跌且未恢复 → 优先削弱它而非加新东西。

# 代码约束

- 只用 #5 环境事实声明的 obs/action 维度和索引。
- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止 import、class、try/except、eval/exec/open。
- 平方根 `** 0.5`；指数 `2.718281828 ** exponent`。
- 正常模式每轮只改一个组件；重建模式可以重写。
- 签名 `def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`

# 输出

```markdown
# 设计理由
（正常模式：改了什么组件、为什么、数学形式、系数校准）
（重建模式：为什么以前都失败了、新骨架选了什么算子、和已尝试过的有什么本质不同）

```python
def compute_reward(...):
    ...
```

# 诊断摘要
- **audit**: （第 0 步的一句话结论）
- **behavior**: （agent 在做什么）
- **signal**: （缺什么或什么过强）
- **level**: Level 1 / Level 2 / Level 3（系统会据此决定是否进入重建模式）
- **hypothesis**: （为什么这个修改应改善）
- **risk**: （最可能的副作用）
```

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 2000.000000
- current_score: -54.867317
- gap_to_target: 2054.867317
- target_achievement_ratio: -2.743%

# 2. 上一轮奖励函数代码（该轮得分: -54.867317）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    # body_up_z = 1 表示完全直立，= 0 表示完全倒地
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控（作为独立正向生存奖励，不扭曲前进信号） ----------
    # [0.2, 0.3] 线性 0→1, [0.9, 1.0] 线性 1→0
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)  # 安全区=1, 越危险→0
    height_reward = 0.1 * height_factor  # 独立的生存奖励，不乘入前进信号

    # ---------- 主学习信号：前进速度 ----------
    forward_reward = 1.0 * body_x_velocity

    # ---------- 直立姿态约束（hinge 惩罚，只在危险时激活） ----------
    # body_up_z < 0.7 时开始惩罚，0.7 对应约 45° 倾斜
    upright_error = max(0.0, 0.7 - body_up_z)
    upright_penalty = -1.0 * upright_error

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + height_reward + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 80.60 | -12.04 | ✅ |
| 2 | 乘性高度门使速度奖励与生存强绑定，agent 会主动保持安全高度以获取前进回报，生存时间延长，累积 speed 得... | 乘性高度门使速度奖励与生存强绑定，agent 会主动保持安全高度以获取前进回报，生存时间延长，累积 speed 得... | 731.50 | -73.71 | ❌ |
| 3 | 将全时二次姿态惩罚替换为 hinge（只在大倾斜时激活），agent 在正常步态中不再受罚，可以自由探索力矩空间学... | 将全时二次姿态惩罚替换为 hinge（只在大倾斜时激活），agent 在正常步态中不再受罚，可以自由探索力矩空间学... | 392.30 | -54.87 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-54.867317, len=392.300000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-617.403098, 261.337561]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 244.146425 | 35.2% | 45.4% | 100.0% |
| upright_penalty | -215.078098 | -31.0% | 31.0% | 34.1% |
| height_reward | 94.751990 | 13.7% | 13.7% | 100.0% |
| action_cost | -69.175846 | -10.0% | 10.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个连续控制运动任务。一个3D四足机器人拥有四条腿和八个力矩控制关节，必须向前行走或奔跑，同时保持身体直立。机器人的身体高度必须保持在健康区间内（最低高度至最高高度之间），一旦高度低于最低值（趴下）或高于最高值（弹飞）则回合立刻终止。智能体的核心目标是产生持续、稳定的向前运动，而不是仅仅维持站立不倒下。次要目标包括减小非必要的关节力矩以及维持机身姿态平稳。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float
- 各维度含义：
  - obs[0]: body_z，身体垂直高度，reward_usable: true
  - obs[1]: quat_w，身体朝向四元数 w 分量，reward_usable: true (用于计算直立程度)
  - obs[2]: quat_x，身体朝向四元数 x 分量，reward_usable: true
  - obs[3]: quat_y，身体朝向四元数 y 分量，reward_usable: true
  - obs[4]: quat_z，身体朝向四元数 z 分量，reward_usable: true
  - obs[5]: joint_1_angle，第一个髋关节角度，reward_usable: false (一般不用)
  - obs[6]: joint_2_angle，第一个踝关节角度，reward_usable: false
  - obs[7]: joint_3_angle，第二个髋关节角度，reward_usable: false
  - obs[8]: joint_4_angle，第二个踝关节角度，reward_usable: false
  - obs[9]: joint_5_angle，第三个髋关节角度，reward_usable: false
  - obs[10]: joint_6_angle，第三个踝关节角度，reward_usable: false
  - obs[11]: joint_7_angle，第四个髋关节角度，reward_usable: false
  - obs[12]: joint_8_angle，第四个踝关节角度，reward_usable: false
  - obs[13]: body_x_velocity，身体世界x方向速度（前进速度），reward_usable: true (主要目标信号)
  - obs[14]: body_y_velocity，身体世界y方向速度（横向速度），reward_usable: true (可用，但非主要)
  - obs[15]: body_z_velocity，身体垂直速度，reward_usable: true (可用于平稳性)
  - obs[16]: body_roll_velocity，滚转角速度，reward_usable: true (姿态稳定性)
  - obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true
  - obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true (非必须，但可约束)
  - obs[19]: joint_1_velocity，第一个髋关节角速度，reward_usable: false (很少用)
  - obs[20]: joint_2_velocity，第一个踝关节角速度，reward_usable: false
  - obs[21]: joint_3_velocity，第二个髋关节角速度，reward_usable: false
  - obs[22]: joint_4_velocity，第二个踝关节角速度，reward_usable: false
  - obs[23]: joint_5_velocity，第三个髋关节角速度，reward_usable: false
  - obs[24]: joint_6_velocity，第三个踝关节角速度，reward_usable: false
  - obs[25]: joint_7_velocity，第四个髋关节角速度，reward_usable: false
  - obs[26]: joint_8_velocity，第四个踝关节角速度，reward_usable: false

注意：前进方向对应obs[13]，且info完全不可访问，所有必须的奖励信号只能来自obs和action。

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- dtype: float
- 连续动作空间，每个分量范围[-1.0, 1.0]，代表关节力矩。
  - action_dim 0: hip_1_torque，第一个髋关节力矩
  - action_dim 1: ankle_1_torque，第一个踝关节力矩
  - action_dim 2: hip_2_torque，第二个髋关节力矩
  - action_dim 3: ankle_2_torque，第二个踝关节力矩
  - action_dim 4: hip_3_torque，第三个髋关节力矩
  - action_dim 5: ankle_3_torque，第三个踝关节力矩
  - action_dim 6: hip_4_torque，第四个髋关节力矩
  - action_dim 7: ankle_4_torque，第四个踝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止，只有达到时间上限truncated才算“存活满时间”。
- failure-like termination:
  - body_height_outside_healthy_range：高度低于0.2或高于1.0（跌倒/弹飞），视为失败。
  - state_value_outside_finite_range：任何状态值变为NaN或inf，视为失败。
- ambiguous termination: 无
- truncation: time_limit_reached，表示达到最大步数，并非失败，应该视为中性或轻微正向。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false (只能通过terminated推断，但官方不允许直接使用terminated标记作为奖励信号，除非我们从状态判断)
- allowed_info_fields: []  (info为空)
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部被禁止

## 7. 可用于奖励函数的信号
- position: body_z (obs[0]), quaternion (obs[1:5]) 可计算直立投影 (body_up_z = 1 - 2*(quat_x²+quat_y²))
- velocity: body_x_velocity (obs[13]), body_y_velocity (obs[14]), body_z_velocity (obs[15]), body_roll_velocity (obs[16]), body_pitch_velocity (obs[17]), body_yaw_velocity (obs[18])
- orientation: quat可直接用于惩罚翻滚/俯仰过大
- contact: 无
- action/engine: action 力矩本身可用于平滑性惩罚
- other: 身体高度是否在健康区间内 (可根据obs[0]判定濒死区域)

# 6. Formula switching guide
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
| 1 | action_cost + forward_reward + height_penalty + upright_penalty | -12.04 | -12.04 | 0.00 | 80.60 | action_cost=-0.043 forward_reward=0.461 height_penalty=-0.021 upright_penalty=-2.311 | new_best |
| 2 | action_cost + forward_reward + upright_penalty | -73.71 | -12.04 | -61.67 | 731.50 | action_cost=-0.042 forward_reward=0.888 upright_penalty=-3.588 | no_meaningful_improvement |
| 3 | action_cost + forward_reward + height_reward + upright_penalty | -54.87 | -12.04 | -42.83 | 392.30 | action_cost=-0.199 forward_reward=0.271 height_reward=0.187 upright_penalty=-0.824 | no_meaningful_improvement |

```
