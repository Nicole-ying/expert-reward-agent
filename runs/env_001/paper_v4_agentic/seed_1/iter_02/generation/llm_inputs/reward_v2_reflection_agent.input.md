# 1. Search objective
- target_score: 200.000000
- current_score: 226.436359
- gap_to_target: -26.436359
- target_achievement_ratio: 113.218%

# 2. 上一轮奖励函数代码（该轮得分: 226.436359）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ================== main progress：减少到目标垫的欧氏距离 ==================
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_old - dist_new                    # 距离缩小为正奖励
    w_progress = 1.0

    # ================== 姿态/角速度稳定约束（hinge 软惩罚）===================
    body_angle = abs(obs[4])
    ang_vel = abs(obs[5]) if abs(obs[5]) > 1e-6 else 0.0

    angle_penalty = max(0.0, body_angle - 0.3)        # 角度超过0.3 rad 才惩罚
    ang_vel_penalty = max(0.0, ang_vel - 1.0)         # 角速度超过1.0 rad/s 才惩罚

    w_angle = 0.05
    w_ang_vel = 0.02

    # ================== 登陆完成软代理（joint_condition_proxy）=================
    # 用 next_obs 判断着陆条件
    proximity = 1.0 / (1.0 + 5.0 * dist_new)           # 越近越接近1
    leg_contact = next_obs[6] * next_obs[7]            # 双腿都接触=1.0，否则0
    speed = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * speed)           # 低速接近1

    # 几何平均缓解乘积塌缩
    proxy = (proximity * leg_contact * speed_factor) ** (1.0/3.0)
    w_proxy = 0.3

    # ================== 总奖励 ==================
    total_reward = (w_progress * progress
                    - w_angle * angle_penalty
                    - w_ang_vel * ang_vel_penalty
                    + w_proxy * proxy)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": -w_angle * angle_penalty,
        "ang_vel_penalty": -w_ang_vel * ang_vel_penalty,
        "landing_proxy": w_proxy * proxy
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=226.436359, len=599.100000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[129.817242, 315.811652]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 101.160778 | 98.5% | 98.5% | 58.3% |
| progress | 1.384166 | 1.3% | 1.4% | 98.6% |
| angle_penalty | -0.070187 | -0.1% | 0.1% | 2.2% |
| ang_vel_penalty | -0.000054 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标

任务目标：控制一个2D飞行器从顶部初始位置尽快且节省燃料地降落到中央目标垫上，并稳定停靠（settle）。  
次目标：使用尽可能少的引擎推力（省燃料），同时保持飞行器姿态稳定（小角度）、降低相对速度、让两条支撑腿都安全接触目标垫。  
不应混淆的目标：不应追求原地悬停、单纯加速、或与垫子发生刚性碰撞。

## 3. 观察空间 observation_space

- type: Box
- shape: [8]
- dtype: float32（推断）
- obs[0]: x_position，水平坐标相对目标垫（目标垫为原点），reward_usable: true
- obs[1]: y_position，垂直坐标相对垫高度，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（1.0/0.0），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（1.0/0.0），reward_usable: true

## 4. 动作空间 action_space

- type: Discrete
- n: 4
- action 0: no_engine — 不点火，仅靠惯性运动
- action 1: left_orientation_engine — 点燃左侧姿态引擎（产生旋转力矩）
- action 2: main_engine — 点燃主引擎（产生推力，主要向上或向前）
- action 3: right_orientation_engine — 点燃右侧姿态引擎（产生反向旋转力矩）

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` — 当飞行器稳定停靠在目标垫上且处于休眠/静止状态时触发，这可能对应成功settle。
- **failure-like termination**: `crash_or_body_contact`（机体与障碍物或危险接触导致坠落/碰撞），`horizontal_position_outside_viewport`（水平位置超出视野范围，出界）。
- **ambiguous termination**: `body_not_awake_or_settled` 也可能是由于空中静止（hover）造成的，需结合位置、速度、接触信号区分成功与悬停。
- **truncation**: 未明确提及最大步数截断，但通常存在环境上限（不可用于奖励）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: `{}`（空字典，无任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为 step 返回空 `{}`。

> **间接推断路径**（derived_possible）：
> - 成功着陆：episode 被 `body_not_awake_or_settled` 终止，且此时 obs 满足 `|x_position|很小，|y_position|很小，|body_angle|小，left_contact==1 且 right_contact==1`。
> - 坠毁：`crash_or_body_contact` 触发，或 x_position 骤变伴随异常接触。
> - 出界：`horizontal_position_outside_viewport` 触发，或 x_position 绝对值超过阈值。
> 以上推断可在 reward 中利用 obs 信号构建成功/失败的密集奖励代理，但**不可**直接使用 done 标志或 info 中的显式 flag。

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
| 1 | ang_vel_penalty + angle_penalty + landing_proxy + progress | 226.44 | 226.44 | 0.00 | 599.10 | ang_vel_penalty=-0.000 angle_penalty=-0.000 landing_proxy=0.130 progress=0.006 | target_solved_new_best |
