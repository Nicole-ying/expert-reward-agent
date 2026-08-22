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

## 0. 信号覆盖审计（先于诊断，必须逐项完成）

**在诊断现有组件之前，首先判断失败是因为信号缺失还是信号校准问题。** 这个区分决定后续所有方向。

### 0.1 终止模式分析

从 #4 的 terminated/truncated 数量和 episode length 分布，推断 agent 主要以什么方式结束 episode：
- 如果大部分 episode 是 truncated（超时）→ agent 存活但未完成任务目标
- 如果大部分 episode 是 terminated 且长度短 → agent 触发了某种终止条件
- 如果 terminated 的 episode 中有长有短 → 可能存在多种终止原因

结合 #5 §5 声明的终止条件列表，推断当前 episode 的终止主要是哪种条件触发的，以及是否有证据表明 agent 已经接近任务完成。

### 0.2 观测使用扫描

逐项检查 #5 中声明的观测维度在 #2 代码中的使用情况：
- 哪些观测维度被使用了？（列出索引和含义）
- 哪些观测维度未被使用？（列出索引和含义）
- 未使用的观测中，是否有维度能提供关于"agent 为什么会以当前模式终止"的信息？
- 未使用的观测中，是否有维度能提供关于"接下来会发生什么"的预判信息？

### 0.3 信号缺口判断

综合 0.1 和 0.2，判断当前奖励函数的信号覆盖状态：
- **信号齐全但校准问题**：所有相关观测已被使用，终止模式与组件激活模式一致 → 问题在权重/阈值/数学形式。走 §1 行为诊断。
- **信号缺失**：存在未使用的观测维度，且该维度可能解释当前终止模式 → 优先考虑新增组件使用该维度。走 §2 的"第0步发现信号缺口 → add 新组件"路径。
- **不确定**：在 §1 诊断中同时保留两种可能性。

### 0.4 僵尸组件检查

#4 组件表中 active_rate < 2% 的组件 → 该组件设计意图未实现，应删除、替换或改造其触发条件。

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
# ⚠️ 上一版代码验证失败
错误信息：Reward v1 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs\env_006\paper_bipedal_hardcore_v1\seed_3\iter_08\generation\validations\reward_v1.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 修复说明
原代码使用了 `obs` 的切片（`obs[0]`, `obs[2]`, `obs[14:24]`），但验证要求仅允许使用 `next_obs`（与历史实现一致）。现已将所有观测信号改为从 `next_obs` 读取，逻辑不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward: progress guided by preview of upcoming terrain roughness,
    with a hinge penalty on excessive torso tilt.
    """
    # --- Extract observations from NEXT_OBS ---------------------------------
    hull_angle = next_obs[0]                # torso tilt (rad)
    horizontal_speed = next_obs[2]          # forward speed (m/s)
    lidar = next_obs[14:24]                 # 10 terrain height readings

    # --- Component A: forward progress ---------------------------------------
    # Encourage forward motion; ignored when moving backwards.
    progress_raw = max(0.0, horizontal_speed)
    w_progress = 1.0

    # --- Component B: preview factor from lidar roughness --------------------
    # Higher terrain roughness -> reduce effective progress, encouraging
    # the agent to slow down / adjust gait before obstacles.
    n_lid = len(lidar)
    if n_lid > 0:
        mean_l = sum(lidar) / n_lid
        # variance of lidar readings as roughness measure
        var_l = sum((l - mean_l) ** 2 for l in lidar) / n_lid
        roughness = var_l ** 0.5
    else:
        roughness = 0.0

    # preview_factor in [preview_min, 1.0]; 1.0 on flat ground, decays with roughness
    k_preview = 2.0          # sensitivity to roughness
    preview_factor = 1.0 / (1.0 + k_preview * roughness)

    progress_reward = w_progress * progress_raw * preview_factor

    # --- Component C: posture hinge penalty ----------------------------------
    # Penalize dangerous torso tilt beyond a safe threshold.
    torso_threshold = 0.5          # ~28.6 degrees
    w_posture = 1.0
    excess_tilt = max(0.0, abs(hull_angle) - torso_threshold)
    posture_penalty = -w_posture * excess_tilt

    # --- Combine -------------------------------------------------------------
    total_reward = progress_reward + posture_penalty
    components = {
        "progress_reward": progress_reward,
        "posture_penalty": posture_penalty
    }

    return float(total_reward), components
```

# 1. Search objective
- target_score: 300.000000
- current_score: -52.725987
- gap_to_target: 352.725987
- target_achievement_ratio: -17.575%

# 2. 上一轮奖励函数代码（该轮得分: -52.725987）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度（线性）
    horizontal_speed = next_obs[2]
    progress = 2.0 * horizontal_speed

    # 稳定/安全约束：姿态角度超出健康范围时软惩罚（hinge）
    hull_angle = next_obs[0]
    max_allowed_angle = 0.3
    posture_penalty = -5.0 * max(0.0, abs(hull_angle) - max_allowed_angle)

    # 稳定/安全约束：角速度惩罚，抑制剧烈摇晃
    ang_vel = next_obs[1]
    ang_vel_penalty = -0.05 * (ang_vel ** 2)

    # 效率/动作代价：轻微二次惩罚
    action_cost = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # 空中惩罚：基于地面接触信号，抑制双脚同时离地
    contact_sum = next_obs[12] + next_obs[13]
    air_penalty = -0.2 * max(0.0, 1.0 - contact_sum)

    # 新增：垂直速度异常惩罚，仅惩罚过快的向下坠落（hinge）
    vertical_speed = next_obs[3]
    max_allowed_fall = 2.0  # 正常跳跃落地的下降速度极限
    vertical_speed_penalty = -0.08 * max(0.0, -vertical_speed - max_allowed_fall)

    total_reward = (progress +
                    posture_penalty +
                    ang_vel_penalty +
                    action_cost +
                    air_penalty +
                    vertical_speed_penalty)

    components = {
        'progress_reward': progress,
        'posture_penalty': posture_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'action_cost': action_cost,
        'air_penalty': air_penalty,
        'vertical_speed_penalty': vertical_speed_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-52.725987, len=323.500000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-94.679558, 0.556516]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 175.944875 | 79.3% | 79.4% | 100.0% |
| air_penalty | -34.680991 | -15.6% | 15.6% | 71.2% |
| posture_penalty | -6.256749 | -2.8% | 2.8% | 4.3% |
| action_cost | -4.756964 | -2.1% | 2.1% | 100.0% |
| ang_vel_penalty | -0.017643 | -0.0% | 0.0% | 78.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
训练一个双足机器人通过布满梯子、树桩、坑洼和不平地形的复杂地面，目标是尽可能远且高效地前进，最终到达地形另一端。不允许摔倒，同时希望最小化不必要的关节力矩以实现节能。核心是稳健前进，附属目标是抑制摔倒和降低功耗。注意：任务描述中“到达尽头、避免摔倒、最小化力矩”三者都可取，但主次关系明确——前进到达尽头是最高目标，生存（不摔倒）是必要条件，力矩最小化属于锦上添花的次生需求。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32
- obs[0]: hull_angle – 身体倾角，reward_usable: true（可用于检测摔倒或大扰动）
- obs[1]: hull_angular_velocity – 身体角速度，reward_usable: true（惩罚急剧旋转）
- obs[2]: horizontal_speed – 水平速度（前进方向），reward_usable: true（直接作为前进主奖励）
- obs[3]: vertical_speed – 垂直速度，reward_usable: true（惩罚异常跳动或坠落）
- obs[4]: joint_0_angle (髋关节1角度)，reward_usable: true（用于姿态约束）
- obs[5]: joint_0_speed (髋关节1角速度)，reward_usable: true（平滑项）
- obs[6]: joint_1_angle (膝关节1角度)，reward_usable: true
- obs[7]: joint_1_speed (膝关节1角速度)，reward_usable: true
- obs[8]: joint_2_angle (髋关节2角度)，reward_usable: true
- obs[9]: joint_2_speed (髋关节2角速度)，reward_usable: true
- obs[10]: joint_3_angle (膝关节2角度)，reward_usable: true
- obs[11]: joint_3_speed (膝关节2角速度)，reward_usable: true
- obs[12]: leg_1_ground_contact (0/1)，reward_usable: true（用于步态模式识别）
- obs[13]: leg_2_ground_contact (0/1)，reward_usable: true
- obs[14~23]: lidar_1~lidar_10 – 前方地形激光测距值，reward_usable: true（可通过差分检测障碍冲击或预测危险，但不建议直接用作奖励信号）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- action_dim 0: hip_1_torque – 髋关节1力矩
- action_dim 1: knee_1_torque – 膝关节1力矩
- action_dim 2: hip_2_torque – 髋关节2力矩
- action_dim 3: knee_2_torque – 膝关节2力矩
四个关节均独立力矩控制，连续动作空间。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 到达地形尽头（reached_end_of_terrain），导致episode终止。
- failure-like termination: 身体摔倒（body_fallen_over），导致episode终止。
- ambiguous termination: 无。所有终止情况必为上述之一。
- truncation: step 返回 truncated=False，不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（未在info中提供）
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 始终为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，不允许在奖励函数中依赖 info。

尽管info无可信标签，但可通过观测信号间接推断终止原因：  
- 摔倒推断：hull_angle 超过阈值（如 >1.0 rad）、身体垂直速度突变向下、或两个腿部接触信号同时长时间为0（失去立足）等组合信号可作为 derived_possible 摔倒信号。  
- 到达终点推断：agent 水平速度持续非零，episode 突然终止且无明显摔倒迹象（hull_angle 正常，垂直速度平稳），此逻辑可用于判断成功，但只能在 episode 结束时进行，奖励函数可在观察到终止时用 next_obs 判断。

## 7. 可用于奖励函数的信号
以下信号可直接或间接用于奖励设计：
- 前进速度：obs[2] (horizontal_speed) 可在每一步提供连续正向激励。
- 身体姿态/稳定：obs[0] (hull_angle) 可惩罚大倾角；obs[1] (hull_angular_velocity) 可惩罚快速旋转。
- 垂直方向异常：obs[3] (vertical_speed) 可惩罚异常跳动（绝对值过大）。
- 关节平滑与能量：action 本身（力矩）可用于二次惩罚（\|action\|²），也可对相邻步的动作差施加惩罚。
- 接触信号：obs[12], obs[13] 可用于生成优雅离地、着地模式，或提供 foot-air-time 奖励（derived_possible）。
- 雷达测距：obs[14:23] 可用于检测极度近距离（即将碰撞）提供的警示信号，但不建议直接用作奖励，可作为惩罚条件。
- 终止推断信号：从 next_obs 中提取 hull_angle、vertical_speed、contact 的组合，以识别摔倒或成功（derived_possible），限用于 episode 结束时的特殊奖励/惩罚。

## 8
```
