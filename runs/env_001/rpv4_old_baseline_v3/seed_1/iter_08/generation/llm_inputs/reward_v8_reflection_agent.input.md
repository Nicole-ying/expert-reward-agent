# 1. Search objective
- target_score: 200.000000
- current_score: 59.583215
- gap_to_target: 140.416785

# 2. Current reward program (score: 59.583215)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]

    v7 design:
    - Restore v5's proven 30x Euclidean progress as the approach engine
    - Add descent_bonus for dense vertical progress (no hovering local optimum)
    - Replace sparse contact/landing with continuous soft_landing gradient
    - Drop fuel_cost entirely (v6's 63.7% magnitude killer)
    - Keep mild attitude penalty and crash guard
    """
    ox, oy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ── 1. Progress: Euclidean distance delta (v5's proven backbone) ──
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    progress_reward = 30.0 * (old_dist - new_dist)

    # ── 2. Descent bonus: reward downward movement ──
    # Dense per-step signal. Hovering or ascending gives zero — no exploit.
    dy = oy - ny
    descent_bonus = 5.0 * max(0.0, dy)

    # ── 3. Attitude penalty: keep it level ──
    attitude_penalty = 0.5 * (n_angle**2) + 0.1 * (n_av**2)

    # ── 4. Soft landing: continuous gradient for contact + low speed + upright ──
    # Uses multiplication: each factor (contact, speed, angle) gates the reward,
    # providing gradient even for partial fulfillment
    both_legs = n_left * n_right
    speed = (nvx**2 + nvy**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(n_angle))
    soft_landing = 25.0 * both_legs * speed_factor * angle_factor

    # ── 5. Crash guard: penalize high-speed ground contact ──
    any_contact = (n_left + n_right) > 0.0
    crash_penalty = speed if (any_contact and speed > 2.0) else 0.0

    total_reward = (
        progress_reward
        + descent_bonus
        - attitude_penalty
        + soft_landing
        - crash_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "descent_bonus": descent_bonus,
        "attitude_penalty": -attitude_penalty,
        "soft_landing": soft_landing,
        "crash_penalty": -crash_penalty,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=59.583215, len=417.600000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-140.494405, 231.250382]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 770.756891 | 94.2% | 94.2% | 12.5% |
| progress_reward | 26.861687 | 3.3% | 4.7% | 100.0% |
| descent_bonus | 7.156978 | 0.9% | 0.9% | 69.3% |
| attitude_penalty | -2.497972 | -0.3% | 0.3% | 100.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
# Checkpoint 评估（训练过程中各组件的变化趋势）

每个 checkpoint 在训练到 N% 时用当前模型评估 20 集，记录每个组件的 ep_sum_mean。
**趋势比单点数值更重要：**
- 组件全程为 0 → 死组件，agent 在训练中忽略了它
- 组件从 0 逐渐增长 → 正在引导学习
- 组件骤升 → 可能被 exploit
- 组件骤降 → agent 在学习抑制它

## Overall score trend
| cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -114.7 | -108.3 | -56.9 | -61.6 | -88.1 | -92.7 | -47.1 | -49.5 | 93.0 | 71.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -1.737 | -0.912 | -0.860 | -1.029 | -0.948 | -1.265 | -1.802 | -1.507 | -1.688 | -1.779 |
| crash_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| descent_bonus | 7.180 | 7.183 | 7.172 | 7.040 | 6.437 | 7.172 | 7.014 | 6.602 | 7.056 | 7.206 |
| progress_reward | 33.649 | 33.659 | 33.426 | 21.493 | 16.445 | 20.406 | 20.073 | 17.989 | 28.212 | 27.099 |
| soft_landing | 8.105 | 10.120 | 10.615 | 188.058 | 161.380 | 159.104 | 725.401 | 243.211 | 823.842 | 764.138 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| crash_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| descent_bonus | 94.3% | 94.3% | 94.5% | 65.7% | 56.8% | 62.7% | 65.4% | 82.3% | 69.7% | 75.6% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% | 99.8% |
| soft_landing | 1.8% | 1.9% | 1.9% | 6.6% | 4.0% | 4.0% | 11.7% | 5.2% | 13.4% | 14.6% |

# 4. Environment facts
## 1. 任务目标
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断)
- **obs[0]**: x_position – 相对于目标垫中心的水平坐标，reward_usable: true
- **obs[1]**: y_position – 相对于垫着陆面的垂直坐标（高度），reward_usable: true
- **obs[2]**: x_velocity – 水平线速度，reward_usable: true
- **obs[3]**: y_velocity – 垂直线速度，reward_usable: true
- **obs[4]**: body_angle – 车体倾角，reward_usable: true
- **obs[5]**: angular_velocity – 角速度，reward_usable: true
- **obs[6]**: left_support_contact – 左支撑腿接触标志（连续 1.0 或 0.0），reward_usable: true
- **obs[7]**: right_support_contact – 右支撑腿接触标志，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- **action 0**: no_engine – 不点火，自由运动  
- **action 1**: left_orientation_engine – 左姿态引擎点火（产生逆时针力矩）  
- **action 2**: main_engine – 主引擎点火（产生向上推力，同时可能改变姿态）  
- **action 3**: right_orientation_engine – 右姿态引擎点火（产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**: `body_not_awake_or_settled` 中的 “settled” 情况（身体静止且可能已着陆在垫上），通常意味着成功。但不能仅靠该标记，因为也可能包含觉醒失败等。
- **failure‑like termination**: `crash_or_body_contact`（与危险物体碰撞或身体部分接触非目标表面）、`horizontal_position_outside_viewport`（飞出水平边界）
- **ambiguous termination**: `body_not_awake_or_settled` 在没有额外信息时可能是成功也可能是长时间静止在空中，需要结合位置和接触信号判断。
- **truncation**: 未提及（环境中无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields:（info 返回 `{}`，没有任何可用字段）
- forbidden_or_uncertain_info_fields: 没有 info 字段，不能依赖任何隐藏终止原因；成功/失败需通过状态和终止组合推断。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` (x), `next_obs[1]` (y) – 接近目标水平的水平和垂直距离
- **velocity**: `next_obs[2]` (vx), `next_obs[3]` (vy) – 抑制高速，鼓励着陆时低速
- **orientation**: `next_obs[4]` (angle), `next_obs[5]` (angular velocity) – 姿态稳定
- **contact**: `next_obs[6]`, `next_obs[7]` – 接触状态，可用于奖励正确着垫
- **action/engine**: `action` – 可用于惩罚不必要的引擎使用，且动作类型影响姿态和推力
- **other**: 可结合上述信号构建复合指标（如距离惩罚、摇晃惩罚）