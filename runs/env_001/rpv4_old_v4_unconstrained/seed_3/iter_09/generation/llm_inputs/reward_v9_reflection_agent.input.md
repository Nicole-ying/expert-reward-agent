下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_8, score=229.814493)
# Training Feedback

## Final-policy outcome
score=229.814493, len=278.000000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-16.883920, 302.212101]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settled_reward | 222.001056 | 76.0% | 76.0% | 32.6% |
| contact_bonus | 50.203315 | 17.2% | 17.2% | 0.6% |
| goal_proximity | -10.080279 | -3.4% | 3.4% | 100.0% |
| velocity_penalty | -3.773904 | -1.3% | 1.3% | 96.4% |
| landing_approach | 3.580217 | 1.2% | 1.2% | 100.0% |
| fuel_penalty | -1.162250 | -0.4% | 0.4% | 83.6% |
| orientation_penalty | -0.957769 | -0.3% | 0.3% | 100.0% |
| contact_speed_penalty | -0.447208 | -0.2% | 0.2% | 29.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


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
| -368.9 | -148.5 | -32.9 | -231.8 | -183.2 | 119.4 | 253.6 | 246.6 | 216.8 | 224.2 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.000 | 5.498 | 440.652 | 1.333 | 0.000 | 18.322 | 37.267 | 46.721 | 31.945 | 32.318 |
| contact_speed_penalty | -1.519 | -1.285 | -1.394 | -1.449 | -3.028 | -6.102 | -0.153 | -0.418 | -0.390 | -0.456 |
| fuel_penalty | -0.358 | -0.886 | -2.379 | -0.654 | -0.501 | -0.810 | -1.356 | -1.228 | -1.809 | -1.664 |
| goal_proximity | -10.455 | -13.275 | -11.507 | -11.761 | -11.389 | -9.795 | -10.309 | -10.077 | -10.436 | -10.283 |
| landing_approach | 0.027 | 1.072 | 11.106 | 0.677 | 0.090 | 1.808 | 4.696 | 4.069 | 7.192 | 6.753 |
| orientation_penalty | -2.325 | -3.470 | -3.090 | -2.398 | -1.812 | -2.024 | -1.205 | -2.118 | -1.640 | -1.167 |
| settled_reward | 0.000 | 3.020 | 568.553 | 46.890 | 0.000 | 56.428 | 130.139 | 141.196 | 290.850 | 409.554 |
| velocity_penalty | -9.783 | -12.780 | -8.617 | -11.618 | -10.123 | -10.911 | -4.087 | -4.914 | -4.550 | -3.777 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.0% | 0.3% | 2.1% | 0.0% | 0.2% | 0.6% | 0.3% | 0.4% | 0.3% | 0.3% |
| contact_speed_penalty | 0.3% | 3.9% | 33.0% | 11.7% | 1.0% | 14.7% | 10.4% | 31.1% | 34.3% | 40.3% |
| fuel_penalty | 65.1% | 71.4% | 85.3% | 67.7% | 64.3% | 67.3% | 80.7% | 71.8% | 79.1% | 82.1% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_approach | 99.6% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| settled_reward | 0.2% | 3.9% | 33.0% | 11.6% | 0.9% | 16.5% | 13.2% | 33.5% | 35.7% | 41.8% |
| velocity_penalty | 100.0% | 100.0% | 99.7% | 100.0% | 100.0% | 97.7% | 96.7% | 97.1% | 98.4% | 98.2% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: return to iter_5 contact_bonus architecture (proven, score=175.7)
        with modest anti-bounce hardening.

    Diagnosis (iter_5→7 regression):
    - iter_5 contact_bonus: w=50, σ²_spatial=0.06, σ²_speed=0.08 → ep_sum=1177,
      score=175.7. Bounce phase at cp80-90% self-resolved by cp100%.
    - iter_7 over-corrected: tightened gates (σ²→0.04 each) + approach_scale →
      contact_bonus collapsed to 24.4 ep_sum, agent never learned to land.
    - The bounce exploit in iter_5 was self-limiting: agent naturally converged
      to settled_reward dominance by cp100%. No heavy anti-bounce needed.

    Changes from iter_7 (→ iter_5):
    1. contact_bonus: RESTORE iter_5 gates (σ² 0.06/0.08), REMOVE approach_scale.
       w=50 (same). This is the single most critical fix.
    2. settled_reward: w 6→3.5. iter_7's w=6 was compensating for dead contact_bonus
       but created extreme volatility. Moderate boost over iter_5's w=2.
    3. contact_speed_penalty: w 2.0→1.8 (between iter_5 1.2 and iter_7 2.0).
    4. goal_proximity: iter_5-style squared distance for strong far-field gradient.
       No approach_bonus (avoids oscillation cycles).
    5. velocity_penalty: iter_5-style gated + anti-slam, slightly wider slam gate.
    6. landing_approach: w 0.03→0.04, whisper only.
    7. orientation_penalty & fuel_penalty: same as iter_5.
    """

    # ── unpack ──────────────────────────────────────────────────────
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_c_cur, right_c_cur = obs[6], obs[7]
    left_c_nxt, right_c_nxt = next_obs[6], next_obs[7]

    # ── derived ─────────────────────────────────────────────────────
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    had_contact = left_c_cur * right_c_cur          # 0.0 or 1.0
    has_contact = left_c_nxt * right_c_nxt           # 0.0 or 1.0

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — squared-distance penalty
    #    Strong far-field gradient (at dist=5: -2.5), gentle near pad.
    #    No approach_bonus: avoids oscillation (approach→overshoot→re-approach).
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.10
    goal_proximity = -w_dist * dist_nxt_sq            # -2.5 at dist=5, -0.1 at dist=1

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed cost + anti-slam near ground
    #    Gate allows free motion far from pad.
    #    Anti-slam penalises downward velocity when very close to ground.
    # ══════════════════════════════════════════════════════════════════
    k_gate = 5.0
    proximity_gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 near pad, ~0.17 at dist=1

    w_v = 0.30
    gated_speed_penalty = -w_v * proximity_gate * speed_sq

    w_slam = 2.0
    slam_gate = 2.718281828 ** (-dist_nxt_sq / 0.04)    # tight: only dist < ~0.25 matters
    downward_v = max(0.0, -vy_nxt)
    anti_slam = -w_slam * slam_gate * (downward_v ** 2)

    velocity_penalty = gated_speed_penalty + anti_slam

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — whisper-level shaping
    #    Tiny weight provides soft gradient toward slow+near+upright
    #    without creating a profitable hover steady-state.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.04 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. contact_bonus — ONE-SHOT transition bonus (iter_5 RESTORED)
    #
    #    This is THE essential landing driver. Fires ONCE on first touchdown.
    #    Gates at iter_5 levels (σ² 0.06/0.08) — tight enough to require
    #    a reasonable landing but loose enough to be discoverable.
    #
    #    NO approach_scale: the iter_5 evidence shows bounce exploit is
    #    self-limiting. Agent converged from bounce (cp80-90%) to settled
    #    (cp100%) naturally. Over-engineering kills the signal.
    #
    #    Anti-bounce now comes from: boosted settled_reward (w=3.5 vs 2)
    #    + stronger contact_speed_penalty (w=1.8 vs 1.2).
    # ══════════════════════════════════════════════════════════════════
    w_contact = 50.0
    new_contact = has_contact * (1.0 - had_contact)       # 1 ONLY on transition step

    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.06)   # σ²=0.06: iter_5 level
    speed_gate   = 2.718281828 ** (-speed_sq / 0.08)      # σ²=0.08: iter_5 level

    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate

    # ══════════════════════════════════════════════════════════════════
    # 6. settled_reward — per-step reward for STAYING landed
    #
    #    w=3.5 (iter_5 was 2.0, iter_7 was 6.0). Moderate boost makes
    #    sustained settling clearly dominate bouncing without creating
    #    the extreme volatility seen in iter_7.
    #
    #    Gates: intermediate between iter_5 and iter_7. Wide enough to
    #    reward imperfect landings, tight enough to incentivise improvement.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 3.5
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.12)    # σ²=0.12: moderate
    s_slow     = 2.718281828 ** (-speed_sq / 0.10)       # σ²=0.10: moderate
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.20) # σ²=0.20: tolerant
    settled_reward = w_settled * has_contact * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_speed_penalty — anti-bounce friction
    #    w=1.8 (iter_5 was 1.2, iter_7 was 2.0). Any speed while in contact
    #    incurs penalty, making bounce cycles less profitable.
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 1.8
    contact_speed_penalty = -w_contact_spd * has_contact * speed_sq

    # ══════════════════════════════════════════════════════════════════
    # 8. fuel_penalty — small regularisation
    # ══════════════════════════════════════════════════════════════════
    w_fuel = 0.005
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ══════════════════════════════════════════════════════════════════
    # assemble
    # ══════════════════════════════════════════════════════════════════
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components
```

---

以上是本轮数据。以下信息可通过工具自主查阅：
- `read_memory()` — 全局记忆表，简单记录每一轮迭代iteration的socre，len长度，奖励函数组件组成
- `read_training_feedback(N)` — 任意迭代的详细训练反馈，了解以前的iteration有过什么好的或者坏的行为，可以避免重蹈覆辙
- `read_reward_code(N)` — 任意迭代的奖励函数代码，方便你深入查看比较好的奖励函数的具体设计
- `read_past_reflection(N)` — 任意迭代的诊断和干预记录，可以查看由好到坏或者由坏到好的记录，吸收学习经验
- `get_component_history(name)` — 组件在所有迭代中的演化，能看清楚某一组件的好的或者坏的演化过程
- `read_environment_card()` — 环境的观察空间、动作空间、终止条件，用来不确定的时候理解每个输入的观测和动作空间的维度信息
- `read_checkpoint_trend(N)` — 训练过程中各组件的趋势，可以看到从训练过程中看出组件是否发挥作用，这样就不仅只看最终反馈，更全面

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

现在开始调查，分析后输出诊断和代码。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。

**2. 完整 Python 代码**（```python ... ``` 代码块）。