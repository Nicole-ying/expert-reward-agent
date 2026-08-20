# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的调查指引，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。快速掌握全貌。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。动手前查询以避免重复失败的修改。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。对比代码差异时使用。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。

# 输出

输出两个部分：

**1. 诊断报告**（在代码之前）：

- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的失败行为
- `signal_completeness`：必要职责是否完备，缺什么信号
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修改应改善策略
- `expected_next_round`：下一轮哪些指标应如何变化
- `main_risk`：最可能引入的新问题

**2. 完整 Python 代码**（```python ... ``` 代码块）。

```

## User Prompt

```markdown
第 9 轮迭代训练已经结束，得分 3.039331（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(9)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

3. **查全局历史避免打转**——用 `read_memory()` 看所有迭代。**关键判断：你是不是在同一个语义功能上反复修改但无突破？**（注意：组件名字可能变了，但语义功能相同——比如 success_landing_bonus 和 success_bonus 都是"着陆成功信号"。）如果某个语义已经改了 2-3 轮都不好，不要再改它了——换一个语义方向。

4. **参考 best code 但要验证**——用 `read_reward_code(best_iter)` 看历史最佳奖励函数的代码。它的哪些组件可能促成了当时的最高分？但不要盲目照搬——结合 best 那轮的 feedback（`read_training_feedback(best_iter)`）看：它的得分分布、终止率、组件 activation 是否健康。best 只是记录最高分，不一定代表正确的引导方式。

5. **Checkpoint 是训练过程的证据**——如果某个组件在 checkpoint 40%→80% 期间 ep_sum 持续为 0 或不变，说明它没有提供有效梯度，agent 在训练中忽略了它。

### 修改原则

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态（稀疏→连续、状态值→改善量、均匀→高度感知等）。适用于组件的引导逻辑有问题（如均匀惩罚导致无近地减速梯度）。
- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。
- **⚠️ 组件被 exploit 时必须干预**：如果一个组件 magnitude_share > 90%，即使它刚被激活，也必须修改它——加触发条件限制、加每 episode 次数上限、或收紧阈值。"刚激活所以只做 L1"不适用于 exploit 场景。
- **避免打转**：如果 `read_past_reflection` 显示你之前已经改过同一语义且结果不好，换方向。

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v4/seed_1/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_9, score=3.039331)
# Training Feedback

## Final-policy outcome
score=3.039331, len=965.950000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-51.586939, 201.807159]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_envelope | 1670.925091 | 98.8% | 98.8% | 100.0% |
| velocity_penalty | -9.010700 | -0.5% | 0.5% | 100.0% |
| goal_proximity | 5.029909 | 0.3% | 0.4% | 100.0% |
| upright_penalty | -5.312046 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
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
| -209.1 | -153.6 | -88.0 | -32.7 | -56.6 | -13.2 | -42.1 | -18.2 | -23.5 | 13.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| goal_proximity | -1.361 | -0.909 | 1.603 | 4.219 | 3.190 | 4.940 | 3.865 | 4.833 | 4.713 | 4.955 |
| soft_landing_envelope | 24.016 | 11.115 | 57.902 | 709.915 | 250.128 | 1019.595 | 382.114 | 1222.565 | 1035.587 | 1580.540 |
| upright_penalty | -4.318 | -8.247 | -5.811 | -4.240 | -5.361 | -4.771 | -5.534 | -5.521 | -5.798 | -6.276 |
| velocity_penalty | -8.124 | -3.805 | -3.524 | -6.392 | -5.387 | -5.748 | -4.645 | -6.944 | -7.250 | -8.723 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_landing_envelope | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| upright_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 9): contact_bonus (sparse binary) replaced by
    soft_landing_envelope — a continuous multiplicative reward that peaks
    at the ideal landing state (h=0, x=0, v=0). Unlike the sparse bonus
    which oscillated between dead (0%) and dominating (98% magnitude share)
    across 7 of 8 iterations, the envelope provides gradient everywhere:
      - exp(-h/0.3): sharp height decay — negligible above h=1, ramps near ground
      - exp(-5x²): centering — same gate proven effective in iters 2/7/8
      - exp(-(vx²+vy²)): slows the agent as it approaches the pad

    Velocity penalty gate softened from exp(-h/1.0) to exp(-h/1.5):
    the iter 8 gate was too sharp, creating a "penalty cliff" near the
    surface that the agent learned to avoid entirely. exp(-h/1.5) provides
    a smoother transition while keeping the penalty concentrated near ground.

    The multiplicative envelope ensures all landing criteria must be
    simultaneously satisfied for high reward — partial satisfaction
    (e.g., centered but fast) yields proportionally less.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 at ground, decays smoothly with height (k=1.5)
    # At h=1.5: gate=0.368, h=1.0: gate=0.513, h=0.5: gate=0.717, h=0: gate=1.0
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.5)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- soft landing envelope: continuous reward peaking at ideal landing ---
    # Three multiplicative factors, each in (0, 1], product peaks at 1.0
    # at the perfect landing state (h=0, x=0, vx=vy=0).
    # Peak value = 3.0 per step at ideal state.
    # At h=0.5, x=0.1, v²=0.1: ~3.0 * 0.189 * 0.951 * 0.905 ≈ 0.49
    # At h=1.0, x=0.2, v²=0.3: ~3.0 * 0.036 * 0.819 * 0.741 ≈ 0.065
    # At h=2.0, x=0.5, v²=1.0: ~3.0 * 0.0013 * 0.287 * 0.368 ≈ 0.0004
    h = max(next_obs[1], 0.0)
    h_factor = 2.718281828 ** (-h / 0.3)
    x_factor = 2.718281828 ** (-5.0 * next_obs[0]**2)
    v_factor = 2.718281828 ** (-(next_obs[2]**2 + next_obs[3]**2))
    soft_landing_envelope = 3.0 * h_factor * x_factor * v_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + soft_landing_envelope

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "soft_landing_envelope": soft_landing_envelope
    }
    return float(total_reward), components
```

# Subagent 数据报告
# 数据报告

## 1. 行为事实
- 得分：score=3.04，range=[-51.59, 201.81]
- 终止方式：terminated 2/20（成功），truncated 18/20（时间上限）
- 回合长度：len=965.95
- Checkpoint 轨迹：官方得分从 cp1=-209.1 → cp2=-153.6 → cp3=-88.0 → cp4=-32.7 → cp5=-56.6 → cp6=-13.2 → cp7=-42.1 → cp8=-18.2 → cp9=-23.5 → cp10=13.9（整体上升，有波动）

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | soft_landing_envelope_ep_sum | soft_landing_envelope_active | contact_bonus_ep_sum | contact_bonus_active | velocity_penalty_ep_sum | velocity_penalty_active | upright_penalty_ep_sum | upright_penalty_active | landing_approach_ep_sum | landing_approach_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -76.56 | 975.15 | 1/20 | 0.64 | 100.0% | — | — | 0.00 | 0.0% | -12.56 | 100.0% | -3.71 | 100.0% | — | — |
| 2 | 63.72 | 668.05 | 17/20 | 1.84 | 98.8% | — | — | 1075.63 | 5.2% | -9.52 | 99.0% | -7.30 | 100.0% | — | — |
| 3 | 21.45 | 864.55 | 9/20 | 1.77 | 99.9% | — | — | 1011.47 | 8.2% | -9.10 | 99.9% | -11.45 | 100.0% | — | — |
| 4 | -10.11 | 1000.00 | 0/20 | 2.92 | 100.0% | — | — | 0.00 | 0.0% | -7.84 | 100.0% | -5.11 | 100.0% | — | — |
| 5 | -33.04 | 1000.00 | 0/20 | 4.29 | 100.0% | — | — | 0.00 | 0.0% | -10.12 | 100.0% | -3.60 | 100.0% | — | — |
| 6 | -81.80 | 1000.00 | 0/20 | 1.77 | 100.0% | — | — | — | — | -2.39 | 100.0% | -3.08 | 100.0% | 0.97 | 98.3% |
| 7 | -5.37 | 964.10 | 2/20 | 4.75 | 100.0% | — | — | 95.90 | 0.7% | -4.62 | 100.0% | -5.45 | 100.0% | — | — |
| 8 | -76.20 | 1000.00 | 0/20 | 2.73 | 100.0% | — | — | 0.00 | 0.0% | -5.31 | 100.0% | -4.19 | 100.0% | — | — |
| 9 | 3.04 | 965.95 | 2/20 | 5.03 | 100.0% | 1670.93 | 100.0% | — | — | -9.01 | 100.0% | -5.31 | 100.0% | — | — |

## 3. 数字标记

- **主导**：soft_landing_envelope magnitude_share=98.8%（>80%）

- **死组件**：无。所有组件 active_rate 均为 100.0%

- **淹没**：goal_proximity active_rate=100% 但 magnitude_share=0.4%（<3%）；upright_penalty active_rate=100% 但 magnitude_share=0.3%（<3%）；velocity_penalty active_rate=100% 但 magnitude_share=0.5%（<3%）

- **全覆盖惩罚**：无。无组件 active_rate > 95% 且 signed_share 为负

- **量级对比**：
  - 每步均值（ep_sum ÷ len）最大：soft_landing_envelope=1.730
  - 每步均值（ep_sum ÷ len）第二大：goal_proximity=0.005
  - 每步均值（ep_sum ÷ len）最小：velocity_penalty=-0.009
  - 每步均值（ep_sum ÷ len）第二小：upright_penalty=-0.006

- **修改变化（iter 9 vs iter 8）**：
  | component | Δ ep_sum | Δ active_rate |
  ||---|---:|---:|
  | goal_proximity | +2.30 | 0pp |
  | velocity_penalty | -3.70 | 0pp |
  | upright_penalty | -1.12 | 0pp |
  | contact_bonus（iter 8）→ soft_landing_envelope（iter 9） | +1670.93 | +100.0pp |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆/接触信号 | v1_contact_bonus(死) → v2_contact_bonus → v3_contact_bonus → v4_contact_bonus(死) → v5_contact_bonus(死) → v6_landing_approach → v7_contact_bonus → v8_contact_bonus(死) → v9_soft_landing_envelope | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 稀疏bonus↔连续奖励：3次切换（v6, v7, v9） |
| 接近/进度信号 | v1_goal_proximity → v2_goal_proximity → v3_goal_proximity → v4_goal_proximity(淹没) → v5_goal_proximity(淹没) → v6_goal_proximity → v7_goal_proximity → v8_goal_proximity(淹没) → v9_goal_proximity(淹没) | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 线性delta→逆距离delta→线性delta：2次切换（v4, v5）；淹没状态：v4/v5/v8/v9 |
| 稳定性/约束信号（速度） | v1_velocity_penalty → v2_velocity_penalty → v3_velocity_penalty → v4_velocity_penalty → v5_velocity_penalty → v6_velocity_penalty → v7_velocity_penalty → v8_velocity_penalty → v9_velocity_penalty | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 门控宽度切换：k=∞→k=2.0→k=1.0→k=1.5（3次切换） |
| 稳定性/约束信号（姿态） | v1_upright_penalty → v2_upright_penalty → v3_upright_penalty → v4_upright_penalty → v5_upright_penalty → v6_upright_penalty → v7_upright_penalty → v8_upright_penalty → v9_upright_penalty | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 无切换（始终为二次型姿态角+角速度惩罚） |

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 2): velocity penalty is now height-gated via
    exp(-height / 2.0). At high altitude, velocity penalty is relaxed,
    allowing the agent to move freely toward the target. Near the ground,
    the penalty ramps up to full strength for soft landing.
    Goal proximity weight increased to 2.0 to strengthen approach incentive.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 2.0 * (prev_dist - next_dist)   # delta reward (weight raised from 1.0)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)  # clamp to avoid amplification below pad
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: reward simultaneous leg contact, modulated by horizontal centering ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian-like decay with horizontal offset
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 50.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
该任务要求控制一个具有两条支撑腿的 2D 刚体飞行器，从视口顶部中央附近开始（带有随机初始力），**快速、低燃料地到达并稳定停靠在中央目标垫上**。核心是到达指定目标位置并实现安全软着陆，附属目标是最小化发动机使用次数（燃料）与总时间。代理人必须学会平滑接近目标、减速、保持竖直姿态并产生双腿的稳定接触。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64
- obs[0]: `x_position` — 飞行器相对于目标垫中心的水平坐标。可用于奖励对靶心的逼近。reward_usable: true
- obs[1]: `y_position` — 飞行器相对于目标垫表面的垂直坐标。可用于奖励下降与接近。reward_usable: true
- obs[2]: `x_velocity` — 水平线速度。可用于惩罚大水平漂移。reward_usable: true
- obs[3]: `y_velocity` — 垂直线速度。控制软着陆冲击，极其重要。reward_usable: true
- obs[4]: `body_angle` — 刚体倾角。竖直向上时接近 0。奖励维持竖直。reward_usable: true
- obs[5]: `angular_velocity` — 角速度。鼓励姿态稳定。reward_usable: true
- obs[6]: `left_support_contact` — 左腿接触标志（0/1）。表示左腿已与地面接触。reward_usable: true
- obs[7]: `right_support_contact` — 右腿接触标志（0/1）。表示右腿已与地面接触。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不开启任何引擎，仅受重力/惯性影响。
- action 1: `left_orientation_engine` — 启动左侧姿态引擎，主要产生逆时针力矩（可能附带微小侧向力），用于校正右倾。
- action 2: `main_engine` — 启动主引擎，产生向上的推力，是主要的减速和高度控制手段。
- action 3: `right_orientation_engine` — 启动右侧姿态引擎，主要产生顺时针力矩，校正左倾。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 当 agent 以低速度、接近竖直的姿态通过双腿接触目标垫，且不再运动（`body_not_awake_or_settled`）时触发。这通常意味着安全着陆成功。
- **failure-like termination**:
  - `crash_or_body_contact`：飞行器任意部位（特别是非腿部）与地面或其它物体发生硬碰撞，导致坠毁。
  - `horizontal_position_outside_viewport`：飞行器水平漂出窗口，视为失控。
- **ambiguous termination**: 若仅因 `body_not_awake_or_settled` 终止，但未满足成功条件（例如悬浮在空中静止，或偏离目标垫静止），则不属于成功；但这类情形在合理动力学下罕见。
- **truncation**: 任务描述中未出现时间截断条件，但实际训练中可能存在未声明的最大步数截断；当前 step 源无 `truncated` 信息，奖励函数中不应依赖该假设。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
  step 返回的 `info` 固定为 `{}`，没有任何成功/失败标志或终止原因字段。奖励函数必须完全从 `next_obs` 的物理状态（位置、速度、接触）中推断是否成功着陆或发生硬碰撞。
- **allowed_info_fields**: 无（info 为空）。
- **forbidden_or_uncertain_info_fields**: `info["success"]`, `info["failure"]`, `info["termination_reason"]` 等未声明字段均禁止使用。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]`（水平差距）, `next_obs[1]`（垂直高度）。两者可直接用于构建距离/接近奖励。
- **velocity**: `next_obs[2]`, `next_obs[3]`。尤其垂向速度对软着陆至关重要。
- **orientation**: `next_obs[4]`（倾角）, `next_obs[5]`（角速度）。可用于姿态稳定性奖励。
- **contact**: `next_obs[6]`, `next_obs[7]`。表示双腿是否触地，是判断着陆成功的关键二值特征。
- **action/engine**: 可利用 `action` 实现燃料惩罚（例如对 1,2,3 动作添加负奖励）。
- **other**: 无其他活跃信号。
```
