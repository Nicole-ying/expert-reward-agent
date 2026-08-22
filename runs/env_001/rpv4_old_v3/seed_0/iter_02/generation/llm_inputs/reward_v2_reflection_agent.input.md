# 1. Search objective
- target_score: 200.000000
- current_score: -40.037266
- gap_to_target: 240.037266
- target_achievement_ratio: -20.019%

# 2. 你的修改自传
（这是你第一次反思，没有历史。专注于当前反馈。）

# 3. 上一轮奖励函数代码（该轮得分: -40.037266）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v1: 主进展信号 + 软着陆约束 + 碰撞预防 + 时间惩罚
    """
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]

    # 距离计算（欧几里得距离）
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（基于距离改善的差分奖励）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)  # 距离减小得正奖励

    # 2. 软着陆约束：接近垫子时惩罚高速和倾斜姿态
    w_soft = 0.5
    proximity_threshold = 0.3  # 当距离 < 0.3 时激活
    gate = max(0.0, 1.0 - next_dist / proximity_threshold)  # 线性衰减门控
    # 速度用绝对值，姿态用平方（抑制大幅倾斜）
    soft_penalty_intensity = abs(next_vx) + abs(next_vy) + body_angle ** 2
    penalty_soft = -w_soft * gate * soft_penalty_intensity

    # 3. 碰撞预防：水平越界惩罚
    x_bound = 0.9  # 视野假设边界
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    # 4. 碰撞预防：高速垂直坠落惩罚（低于垫子且下落过快）
    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:  # 低于垫子且速度向下超过阈值
        # next_vy 为负值，差值平方惩罚强度随速度增大而增加
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 5. 时间惩罚：推动快速完成任务
    living_penalty = -0.002

    total_reward = (reward_prox + penalty_soft +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "soft_landing_penalty": penalty_soft,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 训练数据报告

## 1. 行为事实
- 得分：score=-40.04, range=[-133.28, 15.32]
- 终止方式：terminated 20/20（成功/失败），truncated 0/20（时间上限）
- 回合长度：len=154.05
- Checkpoint 轨迹：官方得分从 cp_40%: 0.00 → cp_80%: 0.00（持平）

## 2. 骨架演进表
| iter | score | len | term | crash_prevention_x_ep_sum | crash_prevention_x_active | crash_prevention_y_ep_sum | crash_prevention_y_active | goal_proximity_ep_sum | goal_proximity_active | living_penalty_ep_sum | living_penalty_active | soft_landing_penalty_ep_sum | soft_landing_penalty_active |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -40.04 | 154.05 | 20/0 | 0.000 | 0.0% | 0.000 | 0.0% | 2.139 | 100.0% | -0.308 | 100.0% | -0.053 | 1.5% |

## 3. 数字标记
- **主导组件**：goal_proximity (magnitude_share = 86.4%)
- **死组件**：
  - soft_landing_penalty (active_rate = 1.5%)
  - crash_prevention_x (active_rate = 0.0%)
  - crash_prevention_y (active_rate = 0.0%)
- **全覆盖惩罚**：living_penalty (active_rate = 100.0%, signed_share = -11.6%)
- **量级对比**（每步均值，ep_sum / len）：
  - 最大两个：
    1. goal_proximity: 0.0139
    2. living_penalty: -0.0020 (绝对值 0.0020)
  - 最小两个：
    1. crash_prevention_x: 0.0000
    2. crash_prevention_y: 0.0000
- **修改变化**：无上一轮，Δ 不适用

# 5. 组件逐项对比
（第一轮反思，无历史对比）

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-40.037266, len=154.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-133.278295, 15.315072]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 2.138831 | 80.7% | 86.4% | 100.0% |
| living_penalty | -0.308100 | -11.6% | 11.6% | 100.0% |
| soft_landing_penalty | -0.052633 | -2.0% | 2.0% | 1.5% |
| crash_prevention_x | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_prevention_y | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：从画面顶部中央附近的随机初态出发，控制二维飞行器尽快到达中心目标垫并稳定着陆（速度接近零、姿态接近水平、支撑腿接触）。  
次目标：在完成主目标的前提下，尽可能少用引擎推力（节能）。  
混淆澄清：该任务不是纯粹的速度竞赛，也不是单纯的节能悬停，核心是实现“成功、快速、节能”的精确着陆。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: `float32`（推断）
- obs[0]: `x_position`，水平坐标（相对目标垫），可用于奖励接近目标，reward_usable: true
- obs[1]: `y_position`，垂直坐标（相对垫高），可用于奖励接近目标和防越界，reward_usable: true
- obs[2]: `x_velocity`，水平线速度，可用于奖励稳定着陆和禁止硬着陆，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，同上，reward_usable: true
- obs[4]: `body_angle`，机体俯仰角，可用于奖励姿态稳定，reward_usable: true
- obs[5]: `angular_velocity`，角速度，可用于奖励姿态平滑，reward_usable: true
- obs[6]: `left_support_contact`，左支撑腿接触标志（0/1），用于判断着陆成功，reward_usable: true
- obs[7]: `right_support_contact`，右支撑腿接触标志，同上，reward_usable: true

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- action 0: `no_engine`，不做任何推力/旋转，完全靠惯性
- action 1: `left_orientation_engine`，激发侧向方向引擎（改变机体转角，可能逆时针）
- action 2: `main_engine`，激发主引擎（沿机体朝向产生推力）
- action 3: `right_orientation_engine`，激发另一侧方向引擎（改变机体转角，可能顺时针）

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（当飞行器停稳、身体不再移动时触发，可能在正确着陆后发生）
- failure-like termination:  
  - `crash_or_body_contact`（主体非支撑部分碰地或坠毁）  
  - `horizontal_position_outside_viewport`（水平方向飞出视野）
- ambiguous termination: `body_not_awake_or_settled`在未到达目标垫时也可能触发（如坠落后停在边界外），但这时的终止不应视为成功；需要通过观察是否在目标垫上和接触确认。
- truncation: 代码中 `truncated` 恒为 `False`，无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `{}`（空字典，无任何 info 字段）
- forbidden_or_uncertain_info_fields: 不存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]`，任何推断成功/失败的逻辑必须从 `obs` / `next_obs` 的状态条件进行判定。

## 7. 可用于奖励函数的信号
- **位置**：`x_position`(obs[0]), `y_position`(obs[1]) — 水平与垂直相对目标垫，可直接用于奖励接近和判断边界。
- **速度**：`x_velocity`(obs[2]), `y_velocity`(obs[3]) — 线速度，用于奖励减速、软着陆。
- **姿态**：`body_angle`(obs[4]) — 角度（可能是弧度或归一化角），用于奖励水平姿态；`angular_velocity`(obs[5]) — 用于奖励姿态平稳。
- **接触**：`left_support_contact`(obs[6]), `right_support_contact`(obs[7]) — 布尔值，标志支撑腿是否接触地面，可奖励着陆成功。
- **动作/引擎**：`action` — 离散索引，可对主引擎和方向引擎使用进行惩罚，实现节能。

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_prevention_x + crash_prevention_y + goal_proximity + living_penalty + soft_landing_penalty | -40.04 | -40.04 | 0.00 | 154.05 | crash_prevention_x=-0.000 crash_prevention_y=-0.007 goal_proximity=0.021 living_penalty=-0.002 soft_landing_penalty=-0.012 | new_best |
