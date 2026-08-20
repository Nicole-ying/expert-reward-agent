# Prompt Record

## System Prompt

```text
你是奖励函数修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- 你将看到：最终评估的组件表（episode_sum_mean, magnitude_share, active_rate），以及训练过程中的组件轨迹（每个 checkpoint 的标量值，含 max/mean/min）。轨迹能告诉你组件是在学习还是停滞了。
- 组件统计是观察证据，不是因果贡献。必须结合 score、episode_length、terminated/truncated 判断。

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。

# 代码约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的 obs、next_obs、action 和 info 字段，不得发明字段、切片维度或新输入。
- 第一个 Python code block 只能包含一个完整的 `compute_reward` 函数；不要写 import、class、try/except 或额外函数，不要使用 self。
- 禁止 eval/exec/open，禁止使用 original_reward 或原始环境 reward。
- 需要平方根时使用 `** 0.5`，禁止 import numpy。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`；components 只放总公式中直接出现的奖励组件。

# 输出

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: -53.292398
- gap_to_target: 253.292398

# 2. Current reward program (score: -53.292398)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]

    Design rationale:
    - Keep v5's proven potential-based progress as the approach engine
    - Add contact_reward to teach the agent to touch the pad (was missing!)
    - Add landing_bonus for the full "stable landing" configuration
    - Mild fuel_cost and crash_penalty as guardrails
    """
    # Current state
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    # Next state
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ── 1. Potential-based progress (kept from v5) ──
    # Rewards moving closer AND slowing down simultaneously
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5
    speed_sq = vx**2 + vy**2
    nspeed_sq = nvx**2 + nvy**2

    pot = -ndist - 0.2 * nspeed_sq
    prev_pot = -dist - 0.2 * speed_sq
    progress_reward = 15.0 * (pot - prev_pot)

    # ── 2. Attitude penalty (slightly strengthened angular velocity weight) ──
    attitude_penalty = 0.5 * (n_angle ** 2) + 0.15 * (n_av ** 2)

    # ── 3. Contact reward: per-leg, dense, additive ──
    # Each leg in contact gives +0.5 per step. Easy to discover.
    # Does NOT require both legs simultaneously — builds a gradient.
    contact_reward = 0.5 * (n_left + n_right)

    # ── 4. Stable landing bonus: sparse but strong ──
    # All four success conditions met simultaneously:
    # both legs down, nearly stopped, nearly upright
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    low_speed = (abs(nvx) < 0.3 and abs(nvy) < 0.3)
    upright = (abs(n_angle) < 0.2)
    landing_bonus = 3.0 if (both_contacts and low_speed and upright) else 0.0

    # ── 5. Fuel efficiency: mild penalty for any engine fire ──
    fuel_cost = 0.05 if action != 0 else 0.0

    # ── 6. Crash penalty: high-speed contact with ground ──
    contact_any = (n_left + n_right) > 0.0
    speed_mag = (nvx**2 + nvy**2) ** 0.5
    crash_penalty = speed_mag if (contact_any and speed_mag > 1.0) else 0.0

    total_reward = (
        progress_reward
        - attitude_penalty
        + contact_reward
        + landing_bonus
        - fuel_cost
        - crash_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": -attitude_penalty,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "fuel_cost": -fuel_cost,
        "crash_penalty": -crash_penalty,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-53.292398, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-90.412879, -9.533744]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_cost | -42.720000 | -63.7% | 63.7% | 85.4% |
| progress_reward | 13.278782 | 19.8% | 34.5% | 100.0% |
| attitude_penalty | -1.205911 | -1.8% | 1.8% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -67.2 | -23.9 | -15.2 | -10.6 | 10.0 | -11.6 | -13.1 | -22.8 | 55.2 | -45.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -2.616 | -2.539 | -2.173 | -2.813 | -2.115 | -1.993 | -1.945 | -1.656 | -1.679 | -1.232 |
| contact_reward | 2.200 | 39.875 | 16.275 | 11.575 | 11.150 | 21.925 | 7.225 | 6.400 | 69.250 | 0.000 |
| crash_penalty | -1.546 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fuel_cost | -1.293 | -7.408 | -14.923 | -24.203 | -27.050 | -36.517 | -41.777 | -46.102 | -37.722 | -43.665 |
| landing_bonus | 0.450 | 21.300 | 10.050 | 8.400 | 0.000 | 0.450 | 0.000 | 0.750 | 166.050 | 0.000 |
| progress_reward | 15.579 | 12.167 | 13.979 | 16.704 | 18.939 | 18.948 | 18.725 | 18.488 | 19.040 | 14.674 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_reward | 4.3% | 28.3% | 5.2% | 1.7% | 1.6% | 4.8% | 1.0% | 0.8% | 9.9% | 0.0% |
| crash_penalty | 1.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fuel_cost | 34.4% | 64.3% | 62.6% | 62.1% | 64.8% | 81.1% | 95.2% | 96.0% | 91.7% | 87.3% |
| landing_bonus | 0.2% | 3.1% | 0.7% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% | 6.7% | 0.0% |
| progress_reward | 100.0% | 100.0% | 99.9% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 99.7% | 100.0% |

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
```
