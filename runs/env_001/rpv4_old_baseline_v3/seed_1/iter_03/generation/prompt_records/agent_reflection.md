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
- current_score: -113.739688
- gap_to_target: 313.739688

# 2. Current reward program (score: -113.739688)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2: balanced approach shaping + tiered contact + efficiency pressure.

    Core changes from v1:
    - approach: vertically-weighted delta-distance (reduced from 30x to 5x)
    - position_quality: dense absolute reward for being near (0,0) with low vel/angle
    - contact_reward: tiered — partial credit for any leg contact, bonus for both
    - hard_landing: penalty for high-speed contact
    - fuel + speed + step costs: create pressure to land efficiently, not hover
    - attitude: strengthened 3x
    """
    # ── state extraction ──
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_x, old_y = obs[0], obs[1]

    # ── 1. approach_shaping: reward moving toward landing zone ──
    # Vertical progress is weighted 2x because descending is the critical action;
    # horizontal centering matters but the agent must commit to going down.
    old_wdist = (old_x ** 2 + 2.0 * old_y ** 2) ** 0.5
    new_wdist = (x ** 2 + 2.0 * y ** 2) ** 0.5
    approach = 5.0 * (old_wdist - new_wdist)

    # ── 2. position_quality: dense reward for being in the landing sweet spot ──
    # Gives the agent a reason to stay near the target, not just approach it.
    # Each factor is in [0, 1]; overall max is 1.0 at perfect landing state.
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    pos_factor = 1.0 / (1.0 + 3.0 * dist)
    vel_factor = 1.0 / (1.0 + 2.0 * speed)
    ang_factor = 1.0 / (1.0 + 2.0 * abs(angle))
    position_quality = 1.5 * pos_factor * vel_factor * ang_factor

    # ── 3. contact_reward: tiered — any contact helps, both legs best ──
    any_contact = max(left_contact, right_contact)          # 1 if either leg touches
    both_contact = left_contact * right_contact             # 1 only if both touch

    contact_softness = 1.0 / (1.0 + 3.0 * speed)
    contact_upright = 1.0 / (1.0 + 2.0 * abs(angle))

    # Partial credit for any leg contact (easier to discover)
    partial_contact = 2.0 * any_contact * contact_softness * contact_upright
    # Full bonus for both legs
    full_contact = 6.0 * both_contact * contact_softness * contact_upright
    contact_reward = partial_contact + full_contact

    # ── 4. hard_landing: penalize high-speed contact ──
    hard_landing = -4.0 * both_contact * speed

    # ── 5. attitude_stabilization: strengthened 3x from v1 ──
    attitude_penalty = -1.5 * (angle ** 2) - 0.5 * (ang_vel ** 2)

    # ── 6. speed_penalty: general disincentive for high velocity ──
    speed_penalty = -0.15 * speed

    # ── 7. fuel_cost: penalize engine use to discourage hovering ──
    if action == 2:          # main engine
        fuel_cost = -0.3
    elif action in (1, 3):   # side orientation engines
        fuel_cost = -0.03
    else:                    # no_engine
        fuel_cost = 0.0

    # ── 8. step_cost: small per-step penalty to create urgency to land ──
    step_cost = -0.1

    # ── total ──
    total_reward = (
        approach
        + position_quality
        + contact_reward
        + hard_landing
        + attitude_penalty
        + speed_penalty
        + fuel_cost
        + step_cost
    )

    components = {
        "approach": approach,
        "position_quality": position_quality,
        "contact_reward": contact_reward,
        "hard_landing": hard_landing,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "fuel_cost": fuel_cost,
        "step_cost": step_cost,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-113.739688, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-136.258679, -93.458690]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -10.987555 | -20.8% | 20.8% | 100.0% |
| attitude_penalty | -9.715681 | -18.4% | 18.4% | 100.0% |
| position_quality | 8.862232 | 16.7% | 16.7% | 100.0% |
| approach | 8.487203 | 16.0% | 16.6% | 100.0% |
| step_cost | -6.840000 | -12.9% | 12.9% | 100.0% |
| contact_reward | 4.055390 | 7.7% | 7.7% | 3.1% |
| hard_landing | -3.589825 | -6.8% | 6.8% | 1.5% |
| fuel_cost | -0.109500 | -0.2% | 0.2% | 5.3% |

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
| -126.5 | -110.0 | -118.2 | -114.6 | -118.9 | -118.0 | -118.2 | -110.4 | -116.4 | -111.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach | 8.458 | 8.488 | 8.479 | 8.487 | 8.480 | 8.479 | 8.480 | 8.496 | 8.487 | 8.490 |
| attitude_penalty | -13.930 | -9.587 | -10.799 | -10.178 | -10.866 | -9.858 | -11.178 | -7.071 | -7.269 | -9.874 |
| contact_reward | 2.616 | 4.113 | 3.592 | 3.978 | 3.817 | 3.683 | 3.522 | 4.898 | 4.583 | 4.179 |
| fuel_cost | -0.138 | -0.542 | -0.105 | -0.251 | -0.114 | -0.091 | -0.091 | -0.126 | -0.102 | -0.108 |
| hard_landing | -2.223 | -4.410 | -3.412 | -4.441 | -3.290 | -3.346 | -3.576 | -4.813 | -3.404 | -4.976 |
| position_quality | 7.671 | 8.856 | 8.534 | 8.820 | 8.561 | 8.653 | 8.699 | 9.407 | 8.905 | 9.009 |
| speed_penalty | -11.047 | -10.989 | -10.981 | -10.992 | -10.983 | -10.985 | -10.985 | -10.958 | -10.953 | -10.987 |
| step_cost | -6.835 | -6.845 | -6.830 | -6.840 | -6.830 | -6.835 | -6.835 | -6.845 | -6.835 | -6.840 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_reward | 3.6% | 2.9% | 3.0% | 2.9% | 3.0% | 3.0% | 3.1% | 3.0% | 3.1% | 3.0% |
| fuel_cost | 6.7% | 16.5% | 5.1% | 8.3% | 4.9% | 4.5% | 4.5% | 6.1% | 5.0% | 5.3% |
| hard_landing | 1.0% | 1.7% | 1.4% | 1.7% | 1.5% | 1.5% | 1.5% | 1.9% | 1.5% | 1.8% |
| position_quality | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| step_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

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
