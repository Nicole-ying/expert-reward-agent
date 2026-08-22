# 1. Search objective
- target_score: 200.000000
- current_score: -121.737397
- gap_to_target: 321.737397

# 2. Current reward program (score: -121.737397)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]

    v8: Fix soft_landing exploit by converting it into a one-shot transition bonus.
    - Keep v5's 30x Euclidean progress backbone
    - Keep descent_bonus for dense vertical signal
    - Replace per-step soft_landing (25.0/step exploit) with:
      * landing_bonus: one-shot transition reward (~50 max) when both legs first touch
      * stability_reward: tiny per-step posture nudge (0.3 max, not exploitable)
    - Drop dead crash_penalty
    - Drop fuel_cost (v6's magnitude killer)
    """
    ox, oy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    o_left, o_right = obs[6], obs[7]
    n_left, n_right = next_obs[6], next_obs[7]

    # ── 1. Progress: Euclidean distance delta (v5's proven backbone) ──
    old_dist = (ox**2 + oy**2) ** 0.5
    new_dist = (nx**2 + ny**2) ** 0.5
    progress_reward = 30.0 * (old_dist - new_dist)

    # ── 2. Descent bonus: reward downward movement, zero for hover/ascent ──
    dy = oy - ny
    descent_bonus = 5.0 * max(0.0, dy)

    # ── 3. Attitude penalty: keep level, suppress spin ──
    attitude_penalty = 0.5 * (n_angle**2) + 0.1 * (n_av**2)

    # ── 4. Landing transition bonus: ONE-SHOT reward for first achieving contact ──
    # Uses obs→next_obs transition: only fires when contact is NEW, not maintained.
    # This eliminates the "sit on pad for 750 points" exploit.
    old_both = o_left * o_right
    new_both = n_left * n_right

    landing_bonus = 0.0
    if new_both > old_both:
        speed = (nvx**2 + nvy**2) ** 0.5
        speed_factor = 1.0 / (1.0 + 5.0 * speed)
        angle_factor = 1.0 / (1.0 + 5.0 * abs(n_angle))
        landing_bonus = 50.0 * speed_factor * angle_factor

    # ── 5. Stability: tiny per-step reward for good contact posture ──
    # At 0.3 max/step, even 30 settled steps = 9 points (not 750).
    stability_reward = 0.0
    if new_both > 0.0:
        speed = (nvx**2 + nvy**2) ** 0.5
        speed_factor = 1.0 / (1.0 + 5.0 * speed)
        angle_factor = 1.0 / (1.0 + 5.0 * abs(n_angle))
        stability_reward = 0.3 * speed_factor * angle_factor

    total_reward = (
        progress_reward
        + descent_bonus
        - attitude_penalty
        + landing_bonus
        + stability_reward
    )

    components = {
        "progress_reward": progress_reward,
        "descent_bonus": descent_bonus,
        "attitude_penalty": -attitude_penalty,
        "landing_bonus": landing_bonus,
        "stability_reward": stability_reward,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-121.737397, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.421811, -99.908123]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 33.779022 | 64.8% | 66.9% | 100.0% |
| landing_bonus | 8.457278 | 16.2% | 16.2% | 1.1% |
| descent_bonus | 7.173807 | 13.8% | 13.8% | 94.3% |
| attitude_penalty | -1.527282 | -2.9% | 2.9% | 100.0% |
| stability_reward | 0.091095 | 0.2% | 0.2% | 1.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| -116.2 | -117.4 | -115.5 | -115.0 | -114.7 | -124.7 | -110.7 | -120.5 | -123.2 | -117.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -1.341 | -2.239 | -1.883 | -1.330 | -1.235 | -2.379 | -1.214 | -1.398 | -1.769 | -1.166 |
| descent_bonus | 7.179 | 7.170 | 7.175 | 7.180 | 7.178 | 7.170 | 7.175 | 7.210 | 7.183 | 7.177 |
| landing_bonus | 10.220 | 7.314 | 4.193 | 5.853 | 9.879 | 6.320 | 5.529 | 11.116 | 6.350 | 7.939 |
| progress_reward | 33.651 | 33.519 | 33.593 | 33.703 | 33.657 | 33.617 | 33.643 | 33.556 | 33.702 | 33.748 |
| stability_reward | 0.116 | 0.089 | 0.083 | 0.126 | 0.114 | 0.082 | 0.143 | 0.121 | 0.083 | 0.111 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| descent_bonus | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% | 94.3% |
| landing_bonus | 1.2% | 1.1% | 1.1% | 1.2% | 1.4% | 1.2% | 1.2% | 1.2% | 1.2% | 1.3% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| stability_reward | 1.8% | 1.5% | 1.6% | 1.8% | 1.9% | 1.6% | 2.0% | 1.9% | 1.8% | 2.0% |

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