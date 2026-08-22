# 1. Search objective
- target_score: 200.000000
- current_score: -211.111008
- gap_to_target: 411.111008

# 2. Current reward program (score: -211.111008)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Descent Incentive (strong, per-step downward progress) ----------
    descent = y_prev - y
    descent_reward = 10.0 * max(0.0, descent)

    # ---------- B: Target Proximity (unified position gradient toward (0,0)) ----------
    dist = (x * x + y * y) ** 0.5
    proximity_reward = 2.0 / (1.0 + dist)

    # ---------- C: Landing Quality (product of soft-landing conditions) ----------
    near_ground = 1.0 / (1.0 + 5.0 * abs(y))
    speed_quality = 1.0 / (1.0 + 3.0 * (vx * vx + vy * vy))
    attitude_quality = 1.0 / (1.0 + 2.0 * angle * angle)
    landing_quality = near_ground * speed_quality * attitude_quality

    # ---------- D: Contact Completion Bonus (large, one-time, quality-scaled) ----------
    contact = left_contact * right_contact
    contact_bonus = 60.0 * contact * speed_quality * attitude_quality

    # ---------- E: Engine Efficiency (mild) ----------
    if action == 2:
        engine_cost = -0.08
    elif action == 1 or action == 3:
        engine_cost = -0.03
    else:
        engine_cost = 0.0

    total_reward = (
        descent_reward
        + proximity_reward
        + landing_quality
        + contact_bonus
        + engine_cost
    )

    components = {
        'descent_reward': descent_reward,
        'proximity_reward': proximity_reward,
        'landing_quality': landing_quality,
        'contact_bonus': contact_bonus,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-211.111008, len=659.450000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-270.754536, -109.435024]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 540.673710 | 82.5% | 82.5% | 100.0% |
| landing_quality | 76.275135 | 11.6% | 11.6% | 100.0% |
| engine_cost | -37.086500 | -5.7% | 5.7% | 91.0% |
| descent_reward | 1.497447 | 0.2% | 0.2% | 50.5% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
### Overall score
| pct | score_mean | score_std |
|---|---|---|
| 40% | -383.46 | 0.00 |
| 80% | -172.24 | 0.00 |

### Per-component ep_sum_mean at each checkpoint
| pct | contact_bonus (ep_sum) | descent_reward (ep_sum) | engine_cost (ep_sum) | landing_quality (ep_sum) | proximity_reward (ep_sum) |
|---|---||---||---||---||---|
| 40% | 0.000 | 0.348 | -36.008 | 44.965 | 388.444 |
| 80% | 0.000 | 2.568 | -52.186 | 112.435 | 749.339 |

### Per-component active_rate at each checkpoint
| pct | contact_bonus (active%) | descent_reward (active%) | engine_cost (active%) | landing_quality (active%) | proximity_reward (active%) |
|---|---||---||---||---||---|
| 40% | 0.0% | 1.8% | 100.0% | 100.0% | 100.0% |
| 80% | 0.0% | 63.5% | 100.0% | 100.0% | 100.0% |

# 4. Environment facts
## 1. 任务目标
该环境是一个 2D 飞行器着陆任务。飞行器从目标着陆平台上方区域出发，受到随机初始扰动力。核心目标是引导飞行器尽快接近并稳定地停在中央着陆平台上，并最大限度减少引擎使用。成功的着陆应满足：位置靠近平台中心、相对垂直速度很小、身体姿态平稳、且两个支撑部件都与平台接触。附属目标是降低燃料消耗（即减少引擎推力动作的使用）并缩短耗时。任务**不**是简单的平衡或存活，而是以到达指定位置为根本驱动。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: 默认 `float32`（其中 contact 标志为 0.0 / 1.0）
- 每一维含义及奖励可用性：

| 维度索引 | 名称 | 含义 | reward_usable |
|-----------|------|------|---------------|
| 0 | `x_position` | 飞行器相对于着陆平台的水平坐标 | true |
| 1 | `y_position` | 飞行器相对于平台高度的垂直坐标 | true |
| 2 | `x_velocity` | 水平线速度 | true |
| 3 | `y_velocity` | 垂直线速度 | true |
| 4 | `body_angle` | 机体倾斜角 | true |
| 5 | `angular_velocity` | 角速度 | true |
| 6 | `left_support_contact` | 左支撑腿是否接触地面（0/1） | true |
| 7 | `right_support_contact` | 右支撑腿是否接触地面（0/1） | true |

所有维度均可安全用于奖励函数计算，不存在需禁用的字段。

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- 动作含义：

| 动作索引 | 名称 | 含义 |
|----------|------|------|
| 0 | `no_engine` | 无任何推力输出 |
| 1 | `left_orientation_engine` | 开启左侧转向引擎（产生逆时针旋转力矩及少量推力） |
| 2 | `main_engine` | 开启主引擎（产生强大的向下推力，抵消重力） |
| 3 | `right_orientation_engine` | 开启右侧转向引擎（产生顺时针旋转力矩及少量推力） |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 当飞行器身体不再活跃（进入休眠）且已基本静止时触发。通常由稳定着陆引发。
- **failure-like termination**: `crash_or_body_contact` —— 飞行器与地面/平台发生不当碰撞（如侧面触地、强烈冲击），视为坠毁；`horizontal_position_outside_viewport` —— 水平位置超出视觉窗口，表示偏离过大。
- **ambiguous termination**: 无。所有终止条件均可明确区分为成功或失败。
- **truncation**: 环境描述未提及最大步数截断，但实际部署时可能存在；truncation 不能被直接当作成功/失败信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （`info` 字典为空，无直接标志位）
- explicit_failure_flag_available: **false** （同上）
- allowed_info_fields: **无** 额外允许字段（`info = {}`）
- forbidden_or_uncertain_info_fields: 所有未在 step source 中出现的 `info` 键均不可用，包括但不限于 `success`、`failure`、`termination_reason`。

因此，奖励函数**必须**仅依赖 `obs`、`action`、`next_obs` 以及终止状态来隐式推断任务成果，不能依赖外部成功/失败标记。

## 7. 可用于奖励函数的信号
- **位置信号**: `x_position`, `y_position` （相对目标，可直接计算距离）
- **速度信号**: `x_velocity`, `y_velocity` （衡量平缓着陆的关键）
- **姿态信号**: `body_angle` （水平姿态维稳）
- **角速度**: `angular_velocity`
- **接触信号**: `left_support_contact`, `right_support_contact` （软着陆的最后确认）
- **动作/引擎信号**: 动作索引可识别是否使用主引擎或转向引擎，用于燃料惩罚。
- **其他潜在信号**: 可从 `obs` 和 `next_obs` 差分得到加速度等，但需谨慎引入。