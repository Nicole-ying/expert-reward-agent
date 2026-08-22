# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs\env_006\paper_bipedal_hardcore_v1\seed_2\iter_03\generation\validations\reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
本轮干预：**Level 2 结构变换 — 新增组件填补信号缺口**。  
核心问题：
- 第 0 步审计发现 **激光雷达（obs[14:23]）完全未被使用**，而环境事实明确指出机器人需利用激光雷达感知前方地形调整步态。
- 第二轮加入 air_stability_penalty 后存活步数从 246 暴跌至 105，且几乎所有 episode 均为早期失败终止，说明 agent 在没有地形感知的情况下无法有效规避障碍，惩罚反而干扰了学习。
- 当前 agent 行为：快速摔倒（短 episode），累积前进奖励很少。
- 信号缺口：缺前方地形粗糙度信息 → agent 在未知障碍上高速冲撞 → 摔倒。

修改方案：
- **修改 forward_progress 组件**，将其由恒等水平速度改为 **地形感知门控速度**：`forward_reward = horizontal_speed * terrain_gate`。
- 用 lidar 读数计算粗糙度（标准差），并映射到 `[0.3, 1.0]` 的 gate 上：平坦地形 gate→1.0（全速奖励），崎岖地形 gate→0.3（大幅减速，降低摔倒风险）。
- 保留 balance_penalty 和 air_stability_penalty 不变，维持姿态约束。
- 该变换为 **Level 2 add 新组件（但形式上通过修改 forward_progress 实现）**，填补信号缺口，且符合“乘法门控优于累加惩罚”的原则。
- 设计校准：gate 最小值 0.3 防止完全抑制前进；总惩罚负担不变（约 0.11/step），主信号平均约 0.15～0.28/step，惩罚/主信号比依然可控。

预期：agent 学会在前方地形崎岖时主动减速，从而减少摔倒，延长存活，累积更高前进回报。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # ------------------------------------------------------------
    # 1. Terrain awareness gate (NEW - replaces raw forward_progress)
    #    Use lidar readings (obs[14:24]) to measure terrain roughness.
    # ------------------------------------------------------------
    lidar_vals = [obs[i] for i in range(14, 24)]
    n_lidar = len(lidar_vals)
    mean_lidar = sum(lidar_vals) / n_lidar
    variance = sum((v - mean_lidar) ** 2 for v in lidar_vals) / n_lidar
    roughness = variance ** 0.5

    # Map roughness to a gate: 1.0 (smooth) -> 0.3 (rough)
    roughness_threshold = 0.3
    # Clip roughness to threshold so gate saturates at 1.0
    roughness_clipped = min(roughness, roughness_threshold)
    gate = 1.0 - 0.7 * (roughness_clipped / roughness_threshold)  # [0.3, 1.0]

    forward_reward = horizontal_speed * gate

    # ------------------------------------------------------------
    # 2. Balance penalty (unchanged)
    # ------------------------------------------------------------
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # ------------------------------------------------------------
    # 3. Air-stability penalty (unchanged)
    # ------------------------------------------------------------
    both_feet_off = max(0.0, 1.0 - (leg1_contact + leg2_contact))
    air_penalty = -0.3 * both_feet_off
    vertical_fall_penalty = -1.0 * both_feet_off * max(0.0, -vertical_speed)
    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_reward + balance_penalty + air_stability_penalty

    components = {
        'forward_reward': forward_reward,          # now gated
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty,
        'terrain_roughness': roughness,
        'terrain_gate': gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失 — 激光雷达（10维地形高度）完全未被使用，agent 失去前方地形信息，无法解释为何在平坦地形外频繁摔倒。
- **behavior**: agent 在 105 步内快速摔倒（20/20 terminated），存活时间比上一轮暴跌 57%，前进奖励极低。
- **signal**: 缺少地形感知 → 高速撞上未预见障碍物导致摔倒；现有惩罚无法弥补前瞻缺口。
- **level**: Level 2（填补信号缺口，新增 terrain‑aware gate）
- **hypothesis**: 将前进奖励与前方地形粗糙度门控，迫使 agent 在崎岖地形减速，从而减少摔倒、延长存活、最终提高累积前进奖励。
- **risk**: 门控阈值若与地形特性不匹配（ roughness_threshold=0.3 偏小）可能导致 agent 过度减速而在平坦段停滞；需后续依据步态长度进行调整。

# 1. Search objective
- target_score: 300.000000
- current_score: -86.300185
- gap_to_target: 386.300185
- target_achievement_ratio: -28.767%

# 2. 上一轮奖励函数代码（该轮得分: -86.300185）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    leg1_contact = obs[12]
    leg2_contact = obs[13]

    # 1. Forward progress (unchanged)
    forward_progress = horizontal_speed

    # 2. Balance penalty (unchanged)
    angle_threshold = 0.4
    angvel_threshold = 1.0
    angle_excess = max(0.0, abs(hull_angle) - angle_threshold)
    angvel_excess = max(0.0, abs(hull_angvel) - angvel_threshold)
    balance_penalty = -3.0 * (angle_excess ** 2) - 0.1 * (angvel_excess ** 2)

    # 3. Air-stability penalty (NEW)
    #    Punish having both feet off the ground, especially when falling downward.
    #    leg contact is 0/1, so sum 0 => both off, 1 => one on, 2 => both on.
    both_feet_off = max(0.0, 1.0 - (leg1_contact + leg2_contact))
    # Base penalty for any airborne frame (small, to allow natural brief flight)
    air_penalty = -0.3 * both_feet_off
    # Extra penalty when airborne and descending (hard landing / falling)
    vertical_fall_penalty = -1.0 * both_feet_off * max(0.0, -vertical_speed)
    air_stability_penalty = air_penalty + vertical_fall_penalty

    total_reward = forward_progress + balance_penalty + air_stability_penalty

    components = {
        'forward_progress': forward_progress,
        'balance_penalty': balance_penalty,
        'air_stability_penalty': air_stability_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-86.300185, len=105.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.688014, -84.316982]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 29.558881 | 70.5% | 70.6% | 100.0% |
| air_stability_penalty | -10.686779 | -25.5% | 25.5% | 68.0% |
| balance_penalty | -1.615357 | -3.9% | 3.9% | 2.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个双足机器人在崎岖不平、布满障碍（阶梯、树桩、坑洼）的地形上尽可能远且高效地向前行走。机器人配备前向激光雷达，可感知前方地形高度。核心目标是学会利用激光雷达信息调整步态，在不摔倒的前提下持续前进；附属目标是减少不必要的关节扭矩（能量效率）并争取到达地形终点。不应将“到达终点”误解为唯一成功信号——能够稳定行走不摔倒才是关键，终点到达是终止条件之一但无独立奖励标注。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: 连续浮点 + 部分二值
- 详细字段（按索引）：
  - obs[0]: hull_angle (name: hull_angle) — 身体俯仰/倾斜角，reward_usable: true，用于检测摔倒和姿态稳定
  - obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true，辅助姿态惩罚
  - obs[2]: horizontal_speed — 质心水平速度，reward_usable: true，核心前进信号
  - obs[3]: vertical_speed — 质心垂直速度，reward_usable: true，可能帮助判断弹跳或摔倒
  - obs[4]: hip_1_angle — 第1髋关节角度，reward_usable: true（关节状态跟踪）
  - obs[5]: hip_1_speed — 第1髋关节角速度
  - obs[6]: knee_1_angle — 第1膝关节角度
  - obs[7]: knee_1_speed — 第1膝关节角速度
  - obs[8]: hip_2_angle — 第2髋关节角度
  - obs[9]: hip_2_speed — 第2髋关节角速度
  - obs[10]: knee_2_angle — 第2膝关节角度
  - obs[11]: knee_2_speed — 第2膝关节角速度
  - obs[12]: leg_1_ground_contact — 第1腿接地指示（0/1），reward_usable: true，可作为步态接触约束
  - obs[13]: leg_2_ground_contact — 第2腿接地指示，同上
  - obs[14]~[23]: lidar_1~lidar_10 — 10个激光测距仪读数，表示前方地形高度。reward_usable: 谨慎使用，不可直接作为奖励项，但可间接推导预见性调整；初始训练阶段不建议直接奖励，但可帮助分析失败模式。

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- bounds: [-1.0, 1.0] 连续值
- 动作含义：
  - action[0]: hip_1_torque — 施加到第1髋关节的扭矩
  - action[1]: knee_1_torque — 施加到第1膝关节的扭矩
  - action[2]: hip_2_torque — 施加到第2髋关节的扭矩
  - action[3]: knee_2_torque — 施加到第2膝关节的扭矩

所有动作均为连续扭矩控制，无离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头），但无显式成功标志。可视为**成功的行走存活**导致的终止。
- failure-like termination: body_fallen_over（身体摔倒），常见于 hull_angle 过大或质心跳跃、触地异常。
- ambiguous termination: 无。
- truncation: 未定义明确截断（step source 中仅 terminated，无 truncated 分支）。因此所有 episode 结束均由终止条件触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 标志）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: []（interface 规定 info_is_empty，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为环境实际不提供任何 info。

尽管如此，可**从观测推导**终止类型：
- 摔倒 (derived_possible): 身体倾斜角 |hull_angle| 超出临界阈值（如 >0.5 rad），或 hull_angular_velocity 突变，同时 leg contact 可能消失。
- 到达终点 (derived_possible): 水平速度仍较高、姿态稳定时 episode 突然终止；也可结合上一步位置推断（但观测无位置），只能依赖速度与姿态平滑终止时的表现进行事后推测。但其可靠性不足以成为奖励条件，可偶尔用于事后分析。

## 7. 可用于奖励函数的信号
- position: 观察中无绝对位置，仅可通过速度累积间接推断位移；无直接可用位置坐标。
- velocity: horizontal_speed（obs[2]）、vertical_speed（obs[3]）、各关节速度（obs[5,7,9,11]）
- orientation: hull_angle（obs[0]）、hull_angular_velocity（obs[1]）
- contact: leg_1_ground_contact（obs[12]）、leg_2_ground_contact（obs[13]）
- action/engine: 动作本身（4维扭矩）
- other:
  - laser scan（obs[14:23]）——可用于推断地形粗糙度，但需谨慎映射为奖励时容易引入噪声；暂时建议不作为常规奖励信号。
  - derived_possible: 通过 hull_angle 阈值或角速度突变推断摔倒；通过 episode 终止时水平速度 & 姿态推断“疑似成功到达”。