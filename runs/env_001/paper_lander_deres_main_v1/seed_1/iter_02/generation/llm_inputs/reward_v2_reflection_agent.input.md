# 上一轮奖励函数代码（该轮得分: 241.414498）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === Progress reward (main learning signal) ===
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress_reward = dist_prev - dist_next  # positive when getting closer

    # === Landing quality bonus (soft proxy for successful landing) ===
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    landing_quality_bonus = 0.0
    if both_contact:
        # position error from center of the pad
        pos_err = dist_next
        # velocity magnitude
        vx, vy = next_obs[2], next_obs[3]
        vel = (vx ** 2 + vy ** 2) ** 0.5
        # absolute body angle
        angle = abs(next_obs[4])

        # exponential-based quality score (max ~1.0 for perfect landing)
        # temperatures control how quickly the score decays with imperfection
        temp_pos = 0.2
        temp_vel = 0.5
        temp_angle = 0.1
        quality = (2.718281828 ** (-pos_err / temp_pos)) * \
                  (2.718281828 ** (-vel / temp_vel)) * \
                  (2.718281828 ** (-angle / temp_angle))
        landing_quality_bonus = quality

    total_reward = progress_reward + landing_quality_bonus

    components = {
        "progress_reward": progress_reward,
        "landing_quality_bonus": landing_quality_bonus
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=241.414498, len=394.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_quality_bonus | 0.369599 | 0.369599 | 0.541875 | 0.682074 | 89.453967 |
| progress_reward | 0.003847 | 0.004132 | 0.994395 | 0.003868 | 0.931035 |
| total_reward | 0.373445 | 0.373656 | 0.999976 | 0.373454 | 90.385002 |
| original_env_reward | -0.128733 | 1.590839 | 1.000000 | -0.128733 | -31.157284 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_quality_bonus | 121.322329 | 121.322329 | 0.000000 | 817.357410 | 3047 |
| progress_reward | 1.265536 | 1.265536 | 0.329691 | 1.420877 | 3047 |
| total_reward | 122.587865 | 122.587865 | 0.329691 | 818.757685 | 3047 |

## Distribution
- score: mean=241.414498, min=81.923051, max=298.037395
- episode_length: mean=394.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个2D飞行器/着陆器轨迹优化任务。飞行器初始位于画面顶部中央附近，带有随机初始作用力。  
**核心目标**：飞行器应尽快且平稳地降落在场地中央的“目标垫”上，并保持稳定的姿态与相对静止。  
**附属约束**：尽可能少地使用引擎推力，但这不是与核心目标冲突的多目标优化任务，而是围绕“高效着陆”的自然偏好。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推断为 float (位置、速度、角度、角速度为连续值，接触标志用 1.0/0.0 表示)  
- obs[0]: x_position — 飞行器相对于目标垫中心的水平距离 (接近0表示水平对齐)  
- obs[1]: y_position — 飞行器相对于目标垫高度的垂直距离 (接近0表示正好在垫面高度)  
- obs[2]: x_velocity — 水平线速度  
- obs[3]: y_velocity — 垂直线速度  
- obs[4]: body_angle — 机体倾斜角 (接近0表示竖直)  
- obs[5]: angular_velocity — 角速度  
- obs[6]: left_support_contact — 左支撑脚触地标志 (1.0 触碰, 0.0 未触碰)  
- obs[7]: right_support_contact — 右支撑脚触地标志 (1.0 触碰, 0.0 未触碰)

## 4. 动作空间 action_space
- type: Discrete (4)  
- action 0: no_engine — 不点火，无推力输出  
- action 1: left_orientation_engine — 侧向引擎喷火，产生侧向推力（用于调整机头朝向/水平速度）  
- action 2: main_engine — 主引擎喷火，产生向上的主要推力（用于减速/悬停）  
- action 3: right_orientation_engine — 与左侧引擎对称的侧向喷火

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` 触发时，若飞行器双支撑脚均触地 (`left_support_contact == 1.0` 且 `right_support_contact == 1.0`)、水平/垂直位置接近 0、速度很小、机体角度近乎竖直，则很可能是一次成功着陆。  
- failure-like termination:  
  - `crash_or_body_contact` — 机体与地面非目标区域剧烈接触（如翻倒、撞击地面），通常视为失败。  
  - `horizontal_position_outside_viewport` — 飞出水平边界，视为失败。  
- ambiguous termination:  
  `body_not_awake_or_settled` 也可能出现在非成功场景（例如飞行器靠某种方式静止在错误位置），因此单靠终止信号无法完全确定成功与否，需要结合观察信号自行判定。  
- truncation: 源信息中未提及 episode 步长上限，因此无显式 truncation；但实际使用中可能存在外部限制（如最大步数），非环境内建。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: (空) — 该环境 `info` 返回空字典 `{}`，因此 `info` 不提供任何额外字段。  
- forbidden_or_uncertain_info_fields: `info` 所有字段均不可用，`info["success"]`、`info["termination_reason"]` 等均不存在。

## 7. 可用于奖励函数的信号
从 `next_obs` 中可直接提取的、可量化的信号包括：
- **位置**：`next_obs[0]` (x), `next_obs[1]` (y) — 表示与目标垫的相对距离，可用于塑造接近/居中的奖励。  
- **速度**：`next_obs[2]` (vx), `next_obs[3]` (vy) — 可用于惩罚过高触地速度。  
- **姿态**：`next_obs[4]` (角度) — 可用于奖励竖直姿态（接近0）。  
- **角速度**：`next_obs[5]` — 可用于惩罚快速旋转。  
- **接触状态**：`next_obs[6]`, `next_obs[7]` — 双脚同时触地可作为成功着陆的有力判据，也可用于塑造中间奖励。  
- **动作/引擎使用**：`action` 为非零（使用引擎）时，可考虑付出燃油成本。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_quality_bonus + progress_reward | 241.41 | 241.41 | 0.00 | 394.00 | landing_quality_bonus=0.370 progress_reward=0.004 | target_solved_new_best |
