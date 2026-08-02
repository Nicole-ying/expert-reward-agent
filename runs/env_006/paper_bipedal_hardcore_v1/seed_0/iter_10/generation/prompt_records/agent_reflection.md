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
错误信息：Reward v10 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs\env_006\paper_bipedal_hardcore_v1\seed_0\iter_10\generation\validations\reward_v10.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
审计发现：当前奖励函数完全未使用 LIDAR 测距信号（obs[14:24]），而任务描述明确指出机器人必须利用前方 10 个 LIDAR 预判崎岖地形以动态调整步态。该信号缺口直接导致 agent 在阶梯、树桩、坑洞等突变地形上没有预警信息，最终摔倒终止。现有组件 active_rate 均 >83%，无僵尸组件；连续三轮预判 ❌ 表明仅靠步态激励+姿态门控在此地形上已经走到尽头。因此本轮引入一个地形粗糙度惩罚（基于 LIDAR std），在崎岖地形上按速度比例惩罚，迫使 agent 在危险地形减速调整步态，平滑地形上仍可全速前进。惩罚尺度已按主信号 0.3× 校准（系数 0.02，预计 per-step 约 -0.02，远低于主信号 per-step 0.258）。此修改为核心信号缺口补齐，而非修补现有逻辑。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    hull_angle = obs[0]
    hull_ang_vel = obs[1]
    horizontal_speed = obs[2]
    leg_1_contact = obs[12]
    leg_2_contact = obs[13]
    # LIDAR 地形预览
    lidar = obs[14:24]

    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门
    # =========================================================================
    tilt_safe_bound = 0.3
    tilt_danger_bound = 0.7
    tilt_margin = tilt_danger_bound - tilt_safe_bound

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    ang_vel_thresh = 2.0
    ang_vel_margin = 4.0
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 接触切换奖励
    # =========================================================================
    leg1_change = abs(next_leg_1_contact - leg_1_contact)
    leg2_change = abs(next_leg_2_contact - leg_2_contact)
    contact_transition_reward = 0.05 * (leg1_change + leg2_change)

    # =========================================================================
    # 5. 能量效率惩罚
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum

    # =========================================================================
    # 6. 地形粗糙度惩罚 (新增 — 利用 LIDAR 预判崎岖，减速避险)
    # =========================================================================
    lidar_mean = sum(lidar) / 10.0
    lidar_var = sum((l - lidar_mean) ** 2 for l in lidar) / 10.0
    roughness = lidar_var ** 0.5
    # 仅在有一定前进速度时，按粗糙度惩罚高速冒险行为
    roughness_penalty = 0.02 * roughness * abs(horizontal_speed)

    # =========================================================================
    # 7. 组合并返回
    # =========================================================================
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward + contact_transition_reward - energy_penalty - roughness_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "contact_transition_reward": contact_transition_reward,
        "energy_penalty": -energy_penalty,
        "roughness_penalty": -roughness_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测扫描发现 LIDAR 信号缺失（obs[14:24] 从未使用），而任务明确要求利用 LIDAR 预判崎岖地形；该缺口极可能是 floor_fallen 终止率高的主因。
- **behavior**: agent 在平坦或缓坡上保持姿态稳定、步态交替，但遇到台阶/树桩等突变地形时因无预警而突然摔倒，导致 episode 提前终止。
- **signal**: 缺少地形危险预警信号，导致 agent 在崎岖区域仍维持高速前进，摔倒后无负反馈。
- **level**: Level 2（信号缺口 → 新增组件）
- **hypothesis**: 新增基于 LIDAR std 的地形粗糙度速度乘积惩罚后，agent 将学会在崎岖地形减速以降低惩罚，从而赢得更多调整步态时间，减少无预判摔倒，延长存活距离并提升总 forward reward。
- **risk**: 若 roughness 阈值不匹配实际地形尺度，可能过度惩罚平坦区域或对极端崎岖仍不足以减速；但系数已降至主信号的 0.08x，预期副作用为轻度减速，不会冻结探索。

# 1. Search objective
- target_score: 300.000000
- current_score: -67.143830
- gap_to_target: 367.143830
- target_achievement_ratio: -22.381%

# 2. 上一轮奖励函数代码（该轮得分: -67.143830）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    hull_angle = obs[0]
    hull_ang_vel = obs[1]
    horizontal_speed = obs[2]
    leg_1_contact = obs[12]
    leg_2_contact = obs[13]

    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门
    # =========================================================================
    tilt_safe_bound = 0.3
    tilt_danger_bound = 0.7
    tilt_margin = tilt_danger_bound - tilt_safe_bound

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    ang_vel_thresh = 2.0
    ang_vel_margin = 4.0
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 接触切换奖励 (新增 — 激励交替步态)
    # =========================================================================
    leg1_change = abs(next_leg_1_contact - leg_1_contact)
    leg2_change = abs(next_leg_2_contact - leg_2_contact)
    contact_transition_reward = 0.05 * (leg1_change + leg2_change)

    # =========================================================================
    # 5. 能量效率惩罚 (轻量)
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum

    # =========================================================================
    # 6. 组合并返回
    # =========================================================================
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward + contact_transition_reward - energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "contact_transition_reward": contact_transition_reward,
        "energy_penalty": -energy_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-67.143830, len=256.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-89.220574, -38.093204]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_gate | 245.042124 | 62.4% | 62.4% | 99.9% |
| forward_reward | 66.048404 | 16.8% | 16.8% | 83.2% |
| gated_forward | 64.142264 | 16.3% | 16.3% | 83.1% |
| contact_transition_reward | 14.950065 | 3.8% | 3.8% | 100.0% |
| energy_penalty | -2.391720 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境中的双足机器人需要在不规则的粗糙地形上尽可能向前行进，同时保持能量效率。地形包含阶梯状、树桩、坑洞等变化，因此机体必须利用前方 10 个 LIDAR 测距信号来预判地形，动态调整步态。主要目标是：稳定行走、远离摔倒、尽量走远；次要目标是：最小化不必要的关节扭矩（能耗）。该任务的核心是崎岖地形上的持续运动控制，而非单纯到达指定坐标点。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（默认推断）  
- obs 各维含义：

| index | 名称                     | 含义                                   | reward_usable |
|-------|--------------------------|----------------------------------------|---------------|
| 0     | hull_angle               | 身体基座倾角                           | true          |
| 1     | hull_angular_velocity    | 身体基座角速度                         | true          |
| 2     | horizontal_speed         | 质心水平速度                           | true          |
| 3     | vertical_speed           | 质心垂直速度                           | true          |
| 4     | joint_0_angle (hip_1)    | 髋关节 1 角度                          | true          |
| 5     | joint_0_speed (hip_1)    | 髋关节 1 角速度                        | true          |
| 6     | joint_1_angle (knee_1)   | 膝关节 1 角度                          | true          |
| 7     | joint_1_speed (knee_1)   | 膝关节 1 角速度                        | true          |
| 8     | joint_2_angle (hip_2)    | 髋关节 2 角度                          | true          |
| 9     | joint_2_speed (hip_2)    | 髋关节 2 角速度                        | true          |
| 10    | joint_3_angle (knee_2)   | 膝关节 2 角度                          | true          |
| 11    | joint_3_speed (knee_2)   | 膝关节 2 角速度                        | true          |
| 12    | leg_1_ground_contact     | 腿 1 是否接地（0 或 1）                | true          |
| 13    | leg_2_ground_contact     | 腿 2 是否接地（0 或 1）                | true          |
| 14    | lidar_1                  | 第一根 LIDAR 测距值（前方地形高度）    | true          |
| 15    | lidar_2                  | 第二根 LIDAR 测距值                    | true          |
| 16    | lidar_3                  | 第三根 LIDAR 测距值                    | true          |
| 17    | lidar_4                  | 第四根 LIDAR 测距值                    | true          |
| 18    | lidar_5                  | 第五根 LIDAR 测距值                    | true          |
| 19    | lidar_6                  | 第六根 LIDAR 测距值                    | true          |
| 20    | lidar_7                  | 第七根 LIDAR 测距值                    | true          |
| 21    | lidar_8                  | 第八根 LIDAR 测距值                    | true          |
| 22    | lidar_9                  | 第九根 LIDAR 测距值                    | true          |
| 23    | lidar_10                 | 第十根 LIDAR 测距值                    | true          |

注：接地信号为 0/1 标量，间接反映了支撑相，可用于步态激励或摔倒检测。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- bounds: [-1.0, 1.0]  
- 各维含义：
  - action_dim 0: hip_1_torque，第一髋关节力矩
  - action_dim 1: knee_1_torque，第一膝关节力矩
  - action_dim 2: hip_2_torque，第二髋关节力矩
  - action_dim 3: knee_2_torque，第二膝关节力矩

所有动作均为连续值，力矩限幅在 [-1, 1] 内。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形末端，视为成功）
- failure-like termination: body_fallen_over（机体摔倒）
- ambiguous termination: 无
- truncation: 无时间上限截断（隐含 episode 可能在短步数内因摔倒而终止，但未明确提供 truncation 信号）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无法直接读取 success 标志）
- explicit_failure_flag_available: false （同上，无直接 failure 字段）
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；original_reward 被强制屏蔽

终止条件的判断只能通过观测间接进行：
- 摔倒：可依据 hull_angle 绝对值超过某经验阈值（如 >0.5 rad）且可能伴随 vertical_speed 突变或 leg 接触异常；标记为 derived_possible。
- 到达终点：在 episode 结束时若 terminated=True 且未检测到摔倒，可推测为成功。但 compute_reward 中无法直接获取 terminated 标志，只能通过最后一步的 next_obs 状态推测，存在误判风险。

因此，成功/失败的信号是弱可用的，理想情况下应避免依赖终点信号，而是专注于持续前进和生存的激励。

## 7. 可用于奖励函数的信号
- position: 无直接位置（但 horizontal_speed 可积分得到水平位移增量）；垂直位移可从 vertical_speed 累积或间接通过高度变化推断（但无绝对高度观测）。
- velocity: horizontal_speed (obs[2])，vertical_speed (obs[3])，各关节角速度 (obs[5,7,9,11])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg_1_ground_contact (obs[12])，leg_2_ground_contact (obs[13])，二值信号，用于检测支撑相或摔倒（例如连续若干步双脚未接地即可能摔倒）。
- action/engine: action 四维力矩（hip_1, knee_1, hip_2, knee_2），可直接用于扭矩惩罚。
- other: LIDAR 读数 (obs[14:24])，提供地形预览，可用于鼓励预判性步态调整，但不易直接转化为标量奖赏，通常用于辅助特征而非独立 reward 项；也可用于检测极端地形。
```
