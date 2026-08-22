# 1. Search objective
- target_score: 200.000000
- current_score: -61.359974
- gap_to_target: 261.359974

# 2. Current reward program (score: -61.359974)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3: back to v1's proven skeleton + descent pressure + stronger landing.

    Core changes from v2:
    - Remove: flat speed_penalty, step_cost, fuel_cost, hard_landing, position_quality
    - Restore: strong proximity_progress (15x, between v1's 30x and v2's 5x)
    - Add: altitude_reward for continuous descent incentive
    - Add: partial_contact for easier landing discovery
    - Strengthen: soft_landing 3x vs v1 (15.0 vs 5.0)
    - Add: proactive ground_speed_penalty (fires before crash, not after)
    - Revert: attitude_penalty to v1's mild coefficients
    """
    # ── state extraction ──
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_x, old_y = obs[0], obs[1]

    # ── 1. proximity_progress: weighted delta-distance ──
    # Vertical weighted 2x — descending is the critical action.
    # 15x coefficient is between v1 (30x) and v2 (5x).
    old_wdist = (old_x ** 2 + 2.0 * old_y ** 2) ** 0.5
    new_wdist = (x ** 2 + 2.0 * y ** 2) ** 0.5
    proximity_progress = 15.0 * (old_wdist - new_wdist)

    # ── 2. altitude_reward: dense incentive to descend ──
    # Max 0.1 at y=0, decays smoothly with height.
    # Gives the agent a reason to commit to going down instead of hovering.
    altitude_reward = 0.1 / (1.0 + 3.0 * y)

    # ── 3. soft_landing: contact-based rewards ──
    speed = (vx ** 2 + vy ** 2) ** 0.5
    both_contact = left_contact * right_contact
    any_contact = max(left_contact, right_contact)

    # Quality gates: landing is only rewarded when gentle and upright
    speed_factor = 1.0 / (1.0 + 5.0 * speed)
    angle_factor = 1.0 / (1.0 + 3.0 * abs(angle))

    # Full bonus for perfect two-leg landing (3x stronger than v1)
    soft_landing = 15.0 * both_contact * speed_factor * angle_factor
    # Partial credit for any leg contact — easier to discover
    partial_contact = 3.0 * any_contact * speed_factor * angle_factor

    # ── 4. attitude_penalty: mild, same as v1 ──
    attitude_penalty = -0.5 * (angle ** 2) - 0.2 * (ang_vel ** 2)

    # ── 5. ground_speed_penalty: proactive crash prevention ──
    # Only activates when near ground AND moving fast — warns before impact.
    # Scales linearly with proximity to ground: stronger as y→0.
    ground_speed_penalty = 0.0
    if y < 0.5 and speed > 0.5:
        ground_speed_penalty = -3.0 * speed * (0.5 - y) / 0.5

    # ── total ──
    total_reward = (
        proximity_progress
        + altitude_reward
        + soft_landing
        + partial_contact
        + attitude_penalty
        + ground_speed_penalty
    )

    components = {
        "proximity_progress": proximity_progress,
        "altitude_reward": altitude_reward,
        "soft_landing": soft_landing,
        "partial_contact": partial_contact,
        "attitude_penalty": attitude_penalty,
        "ground_speed_penalty": ground_speed_penalty,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-61.359974, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-83.392232, -16.155499]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| altitude_reward | 31.594156 | 50.8% | 50.8% | 100.0% |
| proximity_progress | 15.989999 | 25.7% | 42.0% | 100.0% |
| attitude_penalty | -4.468155 | -7.2% | 7.2% | 100.0% |
| ground_speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| partial_contact | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -117.4 | -66.7 | -356.1 | -507.6 | -306.7 | -226.5 | -220.5 | -108.3 | -80.6 | -77.2 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| altitude_reward | 2.239 | 5.126 | 4.741 | 4.057 | 12.704 | 14.923 | 5.368 | 26.012 | 26.950 | 29.488 |
| attitude_penalty | -4.359 | -9.772 | -3.919 | -3.460 | -6.066 | -5.390 | -3.745 | -7.521 | -5.860 | -4.823 |
| ground_speed_penalty | -37.091 | -39.143 | -12.789 | 0.000 | 0.000 | -0.278 | 0.000 | 0.000 | 0.000 | 0.000 |
| partial_contact | 1.544 | 1.781 | 0.593 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| proximity_progress | 25.443 | 25.011 | -35.753 | -62.519 | -16.430 | -4.627 | -12.364 | 9.980 | 12.645 | 13.648 |
| soft_landing | 5.783 | 4.944 | 0.290 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| altitude_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ground_speed_penalty | 20.6% | 14.2% | 3.4% | 0.0% | 0.0% | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| partial_contact | 3.4% | 2.0% | 0.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| proximity_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_landing | 1.8% | 0.9% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

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