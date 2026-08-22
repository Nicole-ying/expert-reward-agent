# 1. Search objective
- target_score: 200.000000
- current_score: -109.576065
- gap_to_target: 309.576065
- target_achievement_ratio: -54.788%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `soft_landing` → 分数 -214.2 → -109.6（+104.6）✅ 大幅改善
- **你做了什么**：唯一目标组件为 soft_landing，将其从“低空门控二次惩罚”改为“全高度线性惩罚”，同时纳入姿态角度和角速度，使其在全程产生负向激励，引导 agent 学习减速与稳定姿态。
- **你为什么这么做**：加入全程速度/姿态惩罚后，agent 将学会使用主引擎抵消下沉、利用姿态引擎抑制翻滚，从而降低惩罚，并与 goal_approach 共同引导向中心垫缓慢降落，最终提高 score 并可能产生腿接触。
- **回合长度**：len: 182→68


**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: -109.576065）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    #    Encourage low speed, upright posture, and slow angular motion.
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. crash_avoidance: unchanged (penalise hard crash below pad without both legs)
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

## 1. 行为事实
- 得分：score = -109.576，范围 [-123.894, -95.059]
- 终止方式：全部 early terminal（score < -50 且 steps < 150），terminated = 20/20，truncated = 0/20；所有 episode 在达到 max steps 前因成功/失败而终止
- 回合长度：平均 68.45 步
- Checkpoint 轨迹：仅提供 40% 和 80% 步数处的得分，均为 0.00（无其他步数数据）

## 2. 骨架演进表
| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active |
|------|--------|------|-------|-----------------------|-----------------------|---------------------|---------------------|------------------------|-------------------------|
| 1    | -214.16 | 182.50 | 20/0 | 0.152 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% |
| 2    | -109.58 | 68.45  | 20/0 | 1.120 | 100.0% | -19.091 | 100.0% | 0.000 | 0.0% |

## 3. 数字标记
- **主导**：soft_landing 的 magnitude_share = 94.3% (> 80%)，占据奖励量级的绝对主导。
- **死组件**：crash_avoidance 的 active_rate = 0.0% (< 2%)，完全未激活。
- **淹没**：无组件同时满足 active_rate > 90% 且 magnitude_share < 3%。
- **全覆盖惩罚**：soft_landing 的 active_rate = 100.0% (> 95%)，且 signed_share = -94.3%（负）。
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - goal_approach: 1.120194 / 68.45 ≈ 0.0164
  - soft_landing: -19.091045 / 68.45 ≈ -0.2789
  - crash_avoidance: 0.0
  - 最大的两个：goal_approach (0.0164)，crash_avoidance (0.0000)
  - 最小的两个：soft_landing (-0.2789)，crash_avoidance (0.0000)
- **修改变化**（与上一轮 iter_01 对比的 Δ）：
  - goal_approach ep_sum: +0.9678，active_rate: 0.0%
  - soft_landing ep_sum: -19.0910，active_rate: +100.0%
  - crash_avoidance ep_sum: 0.0，active_rate: 0.0%

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| crash_avoidance | -0.001 | -0.002 | -0.002 | 0.0% | 0.0% | +0.0pp |
| goal_approach | 0.001 | 0.016 | +0.015 | 100.0% | 100.0% | -0.0pp |
| soft_landing | -0.009 | -0.282 | -0.273 | 0.9% | 100.0% | +99.1pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -214.16 | -109.58 | +104.59 |
| 平均回合长度 | 182.5 | 68.5 | -114.0 |
| 终止回合数 | 20/20 | 20/20 | +0 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-109.576065, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.894147, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.091045 | -94.3% | 94.3% | 100.0% |
| goal_approach | 1.120194 | 5.5% | 5.7% | 100.0% |
| crash_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个2D飞行器从视口顶部中央区域出发，精准降落到中央目标垫上，并保持稳定、安全的接触（两条支撑腿同时着垫、机体姿势竖直、速度极低）。  
次目标：在确保安全着陆的前提下，尽可能缩短总耗时并减少引擎燃料消耗。  
不应混淆的目标：不要单纯追求最快到达而牺牲着陆安全性；也不要只追求零燃料而无限盘旋。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（或 float32，但观察含浮点标志位）
- obs[0]: x_position, 飞行器相对于目标垫的水平坐标 (m)，reward_usable: true
- obs[1]: y_position, 飞行器相对于垫面高度的垂直坐标 (m)，reward_usable: true
- obs[2]: x_velocity, 水平线速度 (m/s)，reward_usable: true
- obs[3]: y_velocity, 垂直线速度 (m/s)，reward_usable: true
- obs[4]: body_angle, 机体偏转角 (rad)，reward_usable: true
- obs[5]: angular_velocity, 角速度 (rad/s)，reward_usable: true
- obs[6]: left_support_contact, 左支撑腿接触标志 (0.0/1.0)，reward_usable: true
- obs[7]: right_support_contact, 右支撑腿接触标志 (0.0/1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: do nothing (no_engine) – 不点火，仅靠惯性运动
- action 1: fire left orientation engine – 产生顺时针力矩，改变姿态
- action 2: fire main engine – 产生向上的推力（同时可能产生微小水平分力？此处按垂直主推力理解）
- action 3: fire right orientation engine – 产生逆时针力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 两腿同时接触目标垫且机体姿态、速度满足稳定条件 → 触发 body_not_awake_or_settled，可能伴随低速度/低角速度，判为成功。
- failure-like termination: ① 机体任何部分与地面或非垫区域接触 (crash_or_body_contact)；② 水平位置超出视口边界 (horizontal_position_outside_viewport)；③ 在未安全着陆的情况下触发休眠 (body_not_awake_or_settled) 也可能失败。
- ambiguous termination: 当 body_not_awake_or_settled 的触发阈值较宽时，可能涵盖成功着陆（两腿着垫、速度低）与卡在边缘/侧翻但速度为零的情况，需由观察进一步区分。
- truncation: 未观察到显式截断，False 表示不使用截断，episode 总是由上述条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} (info 为空字典，不可使用)
- forbidden_or_uncertain_info_fields: 任何 info 字段均禁止使用（因为未声明）。

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: current action (动作索引或 one-hot)，可计算是否使用主引擎/姿态引擎
- other: 无

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_avoidance + goal_approach + soft_landing | -214.16 | -214.16 | 0.00 | 182.50 | crash_avoidance=-0.001 goal_approach=0.001 soft_landing=-0.009 | new_best |
| 2 | crash_avoidance + goal_approach + soft_landing | -109.58 | -109.58 | 0.00 | 68.45 | crash_avoidance=-0.002 goal_approach=0.016 soft_landing=-0.282 | new_best |
