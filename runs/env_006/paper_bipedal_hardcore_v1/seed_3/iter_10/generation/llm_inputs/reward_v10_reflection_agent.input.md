# 1. Search objective
- target_score: 300.000000
- current_score: -29.452195
- gap_to_target: 329.452195
- target_achievement_ratio: -9.817%

# 2. 上一轮奖励函数代码（该轮得分: -29.452195）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extracting relevant observations
    hull_angle_abs = abs(next_obs[0])
    hull_ang_vel_abs = abs(next_obs[1])
    horizontal_speed = next_obs[2]
    vertical_speed = next_obs[3]

    # Core forward progress: only reward positive horizontal speed
    forward_speed = max(0.0, horizontal_speed)

    # Soft health gate: reduces forward reward when posture deteriorates
    # Coefficients are chosen so that typical walking produces gate in [0.4, 0.8],
    # while large tilt or fast rotation significantly attenuate the reward.
    k_angle = 5.0
    k_ang_vel = 0.5
    gate = 1.0 / (1.0 + k_angle * hull_angle_abs + k_ang_vel * hull_ang_vel_abs)

    # Gated forward progress (main learning signal)
    w_fwd = 1.0
    progress_gated = w_fwd * forward_speed * gate

    # Vertical bounce penalty: only penalize excessive up/down oscillations
    vert_threshold = 0.5
    if abs(vertical_speed) > vert_threshold:
        excess = abs(vertical_speed) - vert_threshold
        vert_penalty = -0.1 * (excess ** 2)
    else:
        vert_penalty = 0.0

    total_reward = progress_gated + vert_penalty
    components = {
        'progress_gated': progress_gated,
        'vertical_penalty': vert_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 253.05 | -61.55 | ✅ |
| 2 | posture_gate 乘到 progress_reward 上将迫使 agent 在倾斜时自动减速/纠姿以获得... | posture_gate 乘到 progress_reward 上将迫使 agent 在倾斜时自动减速/纠姿以获得... | 303.45 | -61.57 | ❌ |
| 3 | 引入空中惩罚将迫使 agent 减少双脚同时离地的危险行为，提高落足稳定性，从而降低摔倒率，延长存活时间并为正确步... | 引入空中惩罚将迫使 agent 减少双脚同时离地的危险行为，提高落足稳定性，从而降低摔倒率，延长存活时间并为正确步... | 394.20 | -59.20 | ✅ |
| 4 | 凸化速度奖励使加速的边际收益递增，agent 将被迫从低速保守策略中跳出，在速度和稳定性之间找到更高产出的平衡点，... | 凸化速度奖励使加速的边际收益递增，agent 将被迫从低速保守策略中跳出，在速度和稳定性之间找到更高产出的平衡点，... | 148.40 | -65.67 | ❓ |
| 5 | 骨架变化: action_cost + ang_vel_penalty + posture_penalty +  | — | 190.30 | -65.16 | ❌ |
| 6 | 加入基于接触信号的空中惩罚将抑制双脚同时离地，提高着地稳定性，减少摔倒终止，延长存活时间并积累更多前进奖励。 | 加入基于接触信号的空中惩罚将抑制双脚同时离地，提高着地稳定性，减少摔倒终止，延长存活时间并积累更多前进奖励。 | 323.50 | -52.73 | ✅ |
| 7 | 对过大的向下垂直速度施加软惩罚，可引导 agent 在开始快速下落前调整步态/姿态，减少最终摔倒的概率，拉长 ep... | 对过大的向下垂直速度施加软惩罚，可引导 agent 在开始快速下落前调整步态/姿态，减少最终摔倒的概率，拉长 ep... | 323.50 | -52.73 | ❓ |
| 9 | 骨架变化: progress_gated + vertical_penalty | — | 372.70 | -29.45 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-29.452195, len=372.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.690030, 85.533336]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 104.849838 | 100.0% | 100.0% | 92.4% |
| vertical_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

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

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_penalty + posture_penalty + progress_reward + vertical_penalty | -61.55 | -61.55 | 0.00 | 253.05 | angular_penalty=-0.000 posture_penalty=-0.001 progress_reward=0.069 vertical_penalty=-0.001 | new_best |
| 2 | angular_penalty + posture_gate + progress_reward + vertical_penalty | -61.57 | -61.55 | -0.02 | 303.45 | angular_penalty=-0.000 posture_gate=0.649 progress_reward=0.046 vertical_penalty=-0.000 | no_meaningful_improvement |
| 3 | air_penalty + angular_penalty + posture_gate + progress_reward + vertical_penalty | -59.20 | -59.20 | 0.00 | 394.20 | air_penalty=-0.000 angular_penalty=-0.000 posture_gate=0.664 progress_reward=0.043 vertical_penalty=-0.000 | new_best |
| 4 | air_penalty + angular_penalty + posture_gate + progress_reward + vertical_penalty | -65.67 | -59.20 | -6.47 | 148.40 | air_penalty=-0.000 angular_penalty=-0.000 posture_gate=0.599 progress_reward=0.047 vertical_penalty=-0.001 | unsolved_stagnation_fresh_restart |
| 5 | action_cost + ang_vel_penalty + posture_penalty + progress_reward | -65.16 | -59.20 | -5.96 | 190.30 | action_cost=-0.019 ang_vel_penalty=-0.000 posture_penalty=-0.057 progress_reward=0.382 | no_meaningful_improvement |
| 6 | action_cost + air_penalty + ang_vel_penalty + posture_penalty + progress_reward | -52.73 | -52.73 | 0.00 | 323.50 | action_cost=-0.019 air_penalty=-0.140 ang_vel_penalty=-0.000 posture_penalty=-0.054 progress_reward=0.434 | new_best |
| 7 | action_cost + air_penalty + ang_vel_penalty + posture_penalty + progress_reward + vertical_speed_penalty | -52.73 | -52.73 | 0.00 | 323.50 | action_cost=-0.019 air_penalty=-0.140 ang_vel_penalty=-0.000 posture_penalty=-0.054 progress_reward=0.434 | unsolved_stagnation_fresh_restart |
| 9 | progress_gated + vertical_penalty | -29.45 | -29.45 | 0.00 | 372.70 | progress_gated=0.200 vertical_penalty=-0.000 | new_best |
