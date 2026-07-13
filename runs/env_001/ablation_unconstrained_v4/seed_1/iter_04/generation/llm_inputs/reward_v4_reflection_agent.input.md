# Duplicate reward retry
The previous generation duplicated iter 3 (runs\env_001\ablation_unconstrained_v4\seed_1\iter_03\generation\reward_v3.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- distances ----
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ---- distance progress (strengthened) ----
    progress = prev_dist - next_dist
    c_progress = 5.0 * progress

    # ---- step penalty to discourage lingering ----
    c_step_penalty = -0.05

    # ---- contact reward (without proximity gate) ----
    contact_count = n_left_contact + n_right_contact
    c_contact = 5.0 * contact_count

    # ---- soft landing bonus (one large lump, will only trigger at termination) ----
    landed = (
        n_left_contact > 0.5 and n_right_contact > 0.5 and
        abs(nx) < 0.15 and abs(ny) < 0.15 and
        nvx**2 + nvy**2 < 0.04 and
        abs(n_angle) < 0.2 and
        abs(n_angvel) < 0.2
    )
    c_landing = 200.0 if landed else 0.0

    # ---- speed penalty (global) ----
    speed_sq = nvx**2 + nvy**2
    c_speed = -0.1 * speed_sq

    # ---- posture penalties (global) ----
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.005 * (n_angvel**2)

    # ---- total reward ----
    total_reward = (
        c_progress
        + c_step_penalty
        + c_contact
        + c_landing
        + c_speed
        + c_angle
        + c_angvel
    )

    components = {
        'distance_progress': c_progress,
        'step_penalty': c_step_penalty,
        'contact_reward': c_contact,
        'landing_bonus': c_landing,
        'speed_penalty': c_speed,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
    }

    return (float(total_reward), components)
```

# Search objective
- target_score: 200.000000
- current_score: -62.672049
- gap_to_target: 262.672049
- target_achievement_ratio: -31.336%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -62.672049）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- distances ----
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ---- distance progress (strengthened) ----
    progress = prev_dist - next_dist
    c_progress = 5.0 * progress

    # ---- step penalty to discourage lingering ----
    c_step_penalty = -0.05

    # ---- contact reward (without proximity gate) ----
    contact_count = n_left_contact + n_right_contact
    c_contact = 5.0 * contact_count

    # ---- soft landing bonus (one large lump, will only trigger at termination) ----
    landed = (
        n_left_contact > 0.5 and n_right_contact > 0.5 and
        abs(nx) < 0.15 and abs(ny) < 0.15 and
        nvx**2 + nvy**2 < 0.04 and
        abs(n_angle) < 0.2 and
        abs(n_angvel) < 0.2
    )
    c_landing = 200.0 if landed else 0.0

    # ---- speed penalty (global) ----
    speed_sq = nvx**2 + nvy**2
    c_speed = -0.1 * speed_sq

    # ---- posture penalties (global) ----
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.005 * (n_angvel**2)

    # ---- total reward ----
    total_reward = (
        c_progress
        + c_step_penalty
        + c_contact
        + c_landing
        + c_speed
        + c_angle
        + c_angvel
    )

    components = {
        'distance_progress': c_progress,
        'step_penalty': c_step_penalty,
        'contact_reward': c_contact,
        'landing_bonus': c_landing,
        'speed_penalty': c_speed,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
    }

    return (float(total_reward), components)
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-62.672049, len=979.900000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-91.889328, -15.928720]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| step_penalty | -48.995000 | -61.3% | 61.3% | 100.0% |
| contact_reward | 20.250000 | 25.4% | 25.4% | 0.3% |
| distance_progress | 4.138454 | 5.2% | 9.1% | 100.0% |
| speed_penalty | -3.276417 | -4.1% | 4.1% | 100.0% |
| angle_penalty | -0.103067 | -0.1% | 0.1% | 100.0% |
| angvel_penalty | -0.008897 | -0.0% | 0.0% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个二维飞行器轨迹优化任务。智能体控制一个带有主引擎和左右姿态引擎的飞行器，从视口顶部中心附近出发。主目标是**以尽可能快的速度到达并稳定降落到中央目标垫上**，同时**尽可能少地使用引擎推力**（节省燃料）。次要目标包括保持姿态稳定、在接触时速度尽可能小（软着陆）、避免主体碰撞或飞出边界。  
**不应混淆的目标**：单纯追求速度而不计燃料，或不关心姿态稳定只求触碰目标垫。

## 3. 观察空间 observation_space
- **type:** `Box`
- **shape:** `[8]`
- **dtype:** `float32`（推测，因为标志位也以浮点数 1.0 / 0.0 表示）
- **各维度含义与奖励可用性：**

| 索引 | 名称 | 含义 | 奖励可用 |
|------|------|------|----------|
| 0 | `x_position` | 相对于目标垫中心的水平坐标，目标值为 0 | true |
| 1 | `y_position` | 相对于垫子高度的垂直坐标，目标值为 0 | true |
| 2 | `x_velocity` | 水平线速度 | true |
| 3 | `y_velocity` | 垂直线速度（正值向上，负值向下） | true |
| 4 | `body_angle` | 机体朝向角度 | true |
| 5 | `angular_velocity` | 角速度 | true |
| 6 | `left_support_contact` | 左支撑腿接触标志（1.0 或 0.0） | true |
| 7 | `right_support_contact` | 右支撑腿接触标志（1.0 或 0.0） | true |

## 4. 动作空间 action_space
- **type:** `Discrete`
- **n:** 4
- **动作含义：**

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | `no_engine` | 什么都不做（无推力） |
| 1 | `left_orientation_engine` | 启动左姿态引擎（产生逆时针旋转力矩） |
| 2 | `main_engine` | 启动主引擎（产生向上的推力，抵消重力） |
| 3 | `right_orientation_engine` | 启动右姿态引擎（产生顺时针旋转力矩） |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:**  
  由 `body_not_awake_or_settled` 触发，表示机体已稳定并且不再活跃。若此时满足着陆条件（x≈0, y≈0, 两腿接触, 速度和角速度极低），则可判定为成功着陆。  
- **failure-like termination:**  
  - `crash_or_body_contact`：机体主体或非腿部部分触地/碰到障碍物，视为坠毁。  
  - `horizontal_position_outside_viewport`：水平位置超出视口边界，视为飞出任务区域。  
- **ambiguous termination:**  
  - 理论上 `body_not_awake_or_settled` 若发生在非目标位置（如悬停不动但并未着陆），虽然罕见，但被视作失败（因为未完成到达目标垫的任务）。  
- **truncation:**  
  未提及，此处未使用。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** `false`（`info` 为空，无明确成功标志）
- **explicit_failure_flag_available:** `false`
- **allowed_info_fields:** 无（`info = {}`，任何字段均不可用）
- **forbidden_or_uncertain_info_fields:** `success`, `failure`, `termination_reason` 等任何未在 step 源码中出现的字段均禁止使用。

## 7. 可用于奖励函数的信号
- **位置信号:** `x_position` (goal=0), `y_position` (goal=0)  
- **速度信号:** `x_velocity`, `y_velocity`（希望着陆时接近 0）  
- **姿态与角速度:** `body_angle`, `angular_velocity`（着陆时应接近 0）  
- **接触信号:** `left_support_contact`, `right_support_contact`（两条腿都接触才满足软着陆）  
- **动作信号:** 离散动作编号，可用于推力消耗惩罚  
- **其他:** 可以从 `next_obs` 推断终止时的着陆成功（见 12 节）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_progress + posture_penalty + soft_landing_bonus | -107.52 | -107.52 | 0.00 | 91.20 | distance_progress=0.015 posture_penalty=-0.002 soft_landing_bonus=0.035 | new_best |
| 2 | angle_global + angle_near + angvel_global + angvel_near + contact_reward + distance_progress | -40.47 | -40.47 | 0.00 | 983.30 | angle_global=-0.001 angle_near=-0.009 angvel_global=-0.000 angvel_near=-0.001 contact_reward=2.532 | new_best |
| 3 | angle_penalty + angvel_penalty + contact_reward + distance_progress + landing_bonus + speed_penalty | -62.67 | -40.47 | -22.20 | 979.90 | angle_penalty=-0.004 angvel_penalty=-0.000 contact_reward=3.991 distance_progress=0.026 landing_bonus=34.783 | no_meaningful_improvement |
