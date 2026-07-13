# Search objective
- target_score: 200.000000
- current_score: 114.499268
- gap_to_target: 85.500732
- target_achievement_ratio: 57.250%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 114.499268）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_left = next_obs[6]
    next_right = next_obs[7]

    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward ------------------
    w_progress = 2.0
    progress = dist - next_dist
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated) ------------------
    gate = 1.0 / (1.0 + dist)
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe contact bonus (transition-based) ------------------
    w_contact = 10.0
    prox_factor = 1.0 / (1.0 + 10.0 * next_dist)
    contact_before = obs[6] * obs[7]         # 1.0 if both legs already touching
    contact_now = next_left * next_right      # 1.0 if both legs touching after step
    contact_transition = (1.0 - contact_before) * contact_now  # 1.0 only on newly achieved dual contact
    safe_contact_bonus = w_contact * prox_factor * contact_transition

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=114.499268, len=907.150000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[38.473905, 267.678322]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_bonus | 521.688581 | 95.6% | 95.6% | 8.3% |
| fuel_penalty | -13.953000 | -2.6% | 2.6% | 76.9% |
| stability_penalty | -7.026677 | -1.3% | 1.3% | 100.0% |
| progress_reward | 2.672819 | 0.5% | 0.6% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 载具式着陆任务。主体从视口上方中央附近开始，带有随机初速扰动。**核心目标**是在尽可能短的时间内安全抵达画面中央的目标着陆垫并稳定停驻，同时尽可能减少引擎推力消耗。智能体必须学会精确控制位置与速度，在接近着陆垫时减速、保持竖直姿态，并实现两条支撑腿的平稳接触。

**次要目标**：节约引擎燃料；快速完成任务。  
**不应混淆的目标**：不存在与到达目标同等权重的冲突目标（如“保持高速”或“探索未知区域”），燃料节省仅为附属要求。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（来自 Box 观察）
- 各维度含义：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | x_position | 相对于目标垫的水平坐标 | true |
| 1 | y_position | 相对于目标垫高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 主体姿态角（绝对值或从竖直偏移） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑腿是否触地（1.0 或 0.0） | true（谨慎） |
| 7 | right_support_contact | 右支撑腿是否触地（1.0 或 0.0） | true（谨慎） |

**注意**：左/右支撑接触标志可用于奖励，但需考虑它们可能与导致终止的“crash_or_body_contact”混淆风险（见下文分析）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 各动作含义：

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 不点火，仅靠惯性漂移 |
| 1 | left_orientation_engine | 点燃左侧姿态调节引擎，产生一个方向扭矩/推力 |
| 2 | main_engine | 点燃主引擎（推测向上推力，对抗重力） |
| 3 | right_orientation_engine | 点燃右侧姿态调节引擎，产生相反方向扭矩/推力 |

动作空间为离散选择，每次步只能执行四个动作之一。无连续控制。

## 5. step 与终止条件分析
### 5.1 终止模式
源码中的终止判断为：
```python
terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled
```
其中 **crash_or_body_contact**、**horizontal_position_outside_viewport** 和 **body_not_awake_or_settled** 均为复合条件（具体实现未暴露）。

- **success-like termination**：`body_not_awake_or_settled` 可能表示主体已安定在着陆垫上（无运动或规定时间内稳定），此类终止可视为成功。
- **failure-like termination**：`crash_or_body_contact`（如翻滚、身体直接碰撞）与 `horizontal_position_outside_viewport`（飞出边界）明显是失败。
- **ambiguous termination**：`body_not_awake_or_settled` 若发生在着陆垫之外，可能不算成功，但由于无额外信息，无法在 step 返回值中分辨。环境没有提供显式的“成功 / 失败”标志。
- **truncation**：源码中 `step` 未返回 `truncated`（第四个返回值为 `False`），因此无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: `info` 固定为空字典 `{}`，无任何可用字段。
- forbidden_or_uncertain_info_fields: 不可假设存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等。`info` 完全不可用。

## 7. 可用于奖励函数的信号
基于观察和动作的可用信号分类如下：

- **位置**：`x_position`（0）、`y_position`（1）—— 相对目标垫坐标，目标为 (0, 0)。
- **速度**：`x_velocity`（2）、`y_velocity`（3）—— 目标为零。
- **方向/姿态**：`body_angle`（4）、`angular_velocity`（5）—— 理想为零（竖直稳定）。
- **接触/着陆**：`left_support_contact`（6）、`right_support_contact`（7）—— 标志双腿是否触地，可用于安全着陆奖励，但需注意避免与 crash 混淆（见下节）。
- **动作/引擎**：采取的动作 `action`（0~3）用于推力惩罚（如约束不点火为主，或限制除主引擎外的浪费）。可设计动作惩罚。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_penalty + progress_reward + safe_contact_bonus + stability_penalty | 139.53 | 139.53 | 0.00 | 1000.00 | fuel_penalty=-0.013 progress_reward=0.006 safe_contact_bonus=0.225 stability_penalty=-0.021 | new_best |
| 2 | fuel_penalty + progress_reward + safe_contact_bonus + stability_penalty | 114.50 | 139.53 | -25.03 | 907.15 | fuel_penalty=-0.016 progress_reward=0.006 safe_contact_bonus=0.490 stability_penalty=-0.021 | no_meaningful_improvement |
