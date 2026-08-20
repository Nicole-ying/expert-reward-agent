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
第 6 轮迭代训练已经结束，得分 -81.797020（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(6)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

# 上一轮训练反馈 (iter_6, score=-81.797020)
# Training Feedback

## Final-policy outcome
score=-81.797020, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-149.216081, 84.983972]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_penalty | -3.075439 | -29.4% | 29.4% | 100.0% |
| goal_proximity | 1.774610 | 17.0% | 25.7% | 100.0% |
| velocity_penalty | -2.385383 | -22.8% | 22.8% | 100.0% |
| landing_approach | 0.974175 | 9.3% | 22.1% | 98.3% |

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
| -172.8 | 71.4 | -39.0 | 34.8 | 62.3 | 61.1 | 138.5 | 56.2 | 46.2 | -67.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| goal_proximity | 1.202 | 4.761 | 3.533 | 5.459 | 5.386 | 5.355 | 5.411 | 5.288 | 5.450 | 2.237 |
| landing_approach | 2.814 | 2.583 | 1.780 | 2.815 | 2.814 | 2.818 | 2.815 | 2.817 | 2.820 | 1.187 |
| upright_penalty | -3.310 | -4.073 | -5.312 | -4.958 | -4.097 | -4.874 | -4.026 | -3.755 | -3.743 | -3.008 |
| velocity_penalty | -18.477 | -3.620 | -3.210 | -3.529 | -3.409 | -3.349 | -3.337 | -3.454 | -3.223 | -2.404 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| goal_proximity | 100.0% | 99.7% | 100.0% | 100.0% | 99.7% | 100.0% | 98.6% | 99.8% | 100.0% | 100.0% |
| landing_approach | 100.0% | 91.9% | 100.0% | 94.1% | 89.8% | 83.5% | 96.3% | 94.7% | 94.6% | 99.8% |
| upright_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 99.7% | 100.0% | 100.0% | 99.8% | 100.0% | 98.7% | 99.8% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 6): contact_bonus replaced with landing_approach —
    a continuous potential-based reward that provides gradient all the way
    to the surface. Unlike the sparse binary contact_bonus (which oscillated
    between dead and 98% magnitude-share across iters 1-5), this signal:
      - rewards descending when horizontally centered (x≈0)
      - suppresses descent reward when far from center (via exp(-5x²))
      - provides uniform per-step gradient at all altitudes (no dead zones)
      - cannot explode: total per-episode reward is bounded by the potential
        difference between start and goal states.

    Velocity penalty weight reduced 0.5→0.3 to soften the "near-ground
    penalty wall" and allow the new landing_approach signal to effectively
    guide the final descent phase.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.3 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- landing approach: continuous surface-proximity potential ---
    # phi(x, y) = -y * exp(-alpha * x²)
    #   High potential when low (small y) AND centered (small |x|).
    #   The exp(-5x²) centering gate suppresses descent reward when the
    #   agent is far from the pad, preventing it from diving at wrong spots.
    # Reward = phi(next) - phi(prev):
    #   + when descending while centered (both factors improve)
    #   + when centering while low (x-improvement matters more near surface)
    #   - when rising or moving away from center
    prev_h = max(obs[1], 0.0)
    next_h = max(next_obs[1], 0.0)
    alpha = 5.0
    prev_phi = -prev_h * (2.718281828 ** (-alpha * obs[0]**2))
    next_phi = -next_h * (2.718281828 ** (-alpha * next_obs[0]**2))
    landing_approach = 2.0 * (next_phi - prev_phi)

    total_reward = goal_proximity + vel_penalty + upright_penalty + landing_approach

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "landing_approach": landing_approach
    }
    return total_reward, components
```

# Subagent 数据报告
## 1. 行为事实

- 得分：score=-81.80，range=[-149.22, 84.98]
- 终止方式：terminated 0/20（无成功着陆），truncated 20/20（全部时间上限截断）
- 回合长度：len=1000.00（全部跑满1000步）
- 最高分84.98出现在某评估集，远低于历史最佳265.83（iter_02）
- Checkpoint 轨迹：cp1=-172.8 → cp2=71.4 → cp3=-39.0 → cp4=34.8 → cp5=62.3 → cp6=61.1 → cp7=138.5 → cp8=56.2 → cp9=46.2 → cp10=-67.1
  - 最高官方得分出现在cp7（138.5），随后持续下降至cp10的-67.1
  - cp10得分（-67.1）与cp2得分（71.4）相差138.5，说明训练后期性能大幅退化
  - 与iter_05对比（score=-33.04）：得分下降-48.8，回合长度持平（均为1000）

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | upright_penalty_ep_sum | upright_penalty_active | contact_bonus_ep_sum | contact_bonus_active | landing_approach_ep_sum | landing_approach_active |
|------|-------|-----|------|----------------------|----------------------|------------------------|------------------------|-----------------------|-----------------------|---------------------|---------------------|------------------------|------------------------|
| 1 | -76.56 | 975.15 | 1/20 | 0.635 | 100.0% | -12.561 | 100.0% | -3.714 | 100.0% | 0.000 | 0.0% | — | — |
| 2 | 63.72 | 668.05 | 17/20 | 1.835 | 98.8% | -9.517 | 99.0% | -7.301 | 100.0% | 1075.630 | 5.2% | — | — |
| 3 | 21.45 | 864.55 | 9/20 | 1.769 | 99.9% | -9.103 | 99.9% | -11.452 | 100.0% | 1011.468 | 8.2% | — | — |
| 4 | -10.11 | 1000.00 | 0/20 | 2.922 | 100.0% | -7.837 | 100.0% | -5.113 | 100.0% | 0.000 | 0.0% | — | — |
| 5 | -33.04 | 1000.00 | 0/20 | 4.286 | 100.0% | -10.122 | 100.0% | -3.598 | 100.0% | 0.000 | 0.0% | — | — |
| 6 | -81.80 | 1000.00 | 0/20 | 1.775 | 100.0% | -2.385 | 100.0% | -3.075 | 100.0% | — | — | 0.974 | 98.3% |

## 3. 数字标记

- **主导**：无。最大magnitude_share为upright_penalty的29.4%，远低于80%阈值。

- **死组件**：无。所有组件active_rate均≥98.3%。

- **淹没**：无。无组件同时满足active_rate>90%且magnitude_share<3%。

- **全覆盖惩罚**：无。无组件满足active_rate>95%且signed_share为负——upright_penalty虽为负且active_rate=100%，但magnitude_share=29.4%，不是完全主导；velocity_penalty虽为负且active_rate=100%，但magnitude_share=22.8%，同样不构成全覆盖。

- **量级对比**：
  - 每步均值（ep_sum ÷ len=1000）：
    - goal_proximity: 0.00177
    - velocity_penalty: -0.00239
    - upright_penalty: -0.00308
    - landing_approach: 0.00097
  - 最大两个：upright_penalty (-0.00308，绝对值最大)，velocity_penalty (-0.00239)
  - 最小两个：landing_approach (0.00097)，goal_proximity (0.00177)

- **修改变化**（iter_05 → iter_06）：
  - goal_proximity: ep_sum 4.286→1.775（Δ=-2.511），active_rate 100.0%→100.0%（Δ=0.0pp）
  - velocity_penalty: ep_sum -10.122→-2.385（Δ=+7.737），active_rate 100.0%→100.0%（Δ=0.0pp）
  - upright_penalty: ep_sum -3.598→-3.075（Δ=+0.523），active_rate 100.0%→100.0%（Δ=0.0pp）
  - contact_bonus→landing_approach: ep_sum 0.000→0.974（新组件），active_rate 0.0%→98.3%（Δ=+98.3pp）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_goal_proximity→v2_goal_proximity→v3_goal_proximity→v4_goal_proximity→v5_goal_proximity→v6_goal_proximity | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | delta距离(iter_01)→delta距离×2(iter_02)→逆势能(iter_04)→delta距离×4(iter_05-06): 1次切换 |
| 着陆/接触信号 | v1_contact_bonus(死)→v2_contact_bonus(5.2%)→v3_contact_bonus(8.2%)→v4_contact_bonus(死)→v5_contact_bonus(死)→v6_landing_approach | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | 稀疏bonus(iter_01-05)↔连续势能(iter_06): 1次切换 |
| 稳定性/约束（速度） | v1_velocity_penalty→v2_velocity_penalty→v3_velocity_penalty→v4_velocity_penalty→v5_velocity_penalty→v6_velocity_penalty | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | 全高度(iter_01)→门控(iter_02-06): 1次切换 |
| 稳定性/约束（姿态） | v1_upright_penalty→v2_upright_penalty→v3_upright_penalty→v4_upright_penalty→v5_upright_penalty→v6_upright_penalty | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | 无切换(持续active) |

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
