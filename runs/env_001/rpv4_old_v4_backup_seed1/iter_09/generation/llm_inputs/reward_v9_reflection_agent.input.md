第 8 轮迭代训练已经结束，得分 -76.197309（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(8)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

# 上一轮训练反馈 (iter_8, score=-76.197309)
# Training Feedback

## Final-policy outcome
score=-76.197309, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-117.545539, -12.319018]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 2.728712 | 18.3% | 36.5% | 100.0% |
| velocity_penalty | -5.308256 | -35.5% | 35.5% | 100.0% |
| upright_penalty | -4.189605 | -28.0% | 28.0% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -92.3 | -177.2 | -133.5 | -126.7 | -174.3 | -123.8 | -120.2 | -82.0 | -65.2 | -71.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 75.195 | 0.000 | 0.000 | 0.000 | 0.000 | 88.965 | 0.000 | 0.000 | 0.000 | 0.000 |
| goal_proximity | 2.354 | 0.141 | 1.387 | 1.781 | 1.280 | 2.053 | 2.015 | 2.543 | 3.143 | 2.908 |
| upright_penalty | -10.866 | -3.725 | -4.044 | -3.607 | -3.904 | -9.536 | -3.920 | -4.319 | -4.061 | -4.203 |
| velocity_penalty | -27.416 | -6.856 | -5.869 | -5.090 | -7.401 | -9.433 | -7.582 | -5.513 | -6.115 | -5.288 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 2.9% | 0.0% | 0.0% | 0.0% | 0.0% | 1.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| upright_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 8): velocity penalty gate sharpened from exp(-h/2.0)
    to exp(-h/1.0) and weight raised from 0.3 to 0.5. This concentrates
    the soft-landing penalty near the ground (h < 1.0) while giving the
    agent more freedom at altitude. The sharper near-ground gradient forces
    deceleration in the final descent phase, bridging the gap between
    "approaching the pad" and "actually landing."

    Evidence: iter 7 had only 2/20 landings despite active goal_proximity.
    The velocity penalty at -0.3 with the wide gate (exp(-h/2)) was too
    diffuse to create a clear "slow down now" signal near the surface.
    Iter 2 (best, 17/20) used w=0.5, and the sharper gate retains altitude
    freedom while making the near-ground braking signal even stronger than
    iter 2's configuration.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: sharply height-gated ---
    # gate = 1.0 at ground, decays rapidly with height (k=1.0 vs old k=2.0)
    # At h=1.0: gate=0.368 (old: 0.607) — weaker, more freedom at altitude
    # At h=0.3: gate=0.741 (old: 0.861) — comparable
    # At h=0.0: gate=1.000 — full penalty strength
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: moderate terminal reward for dual-leg landing ---
    # Value 20 kept from iter 7 — large enough to incentivize landing,
    # small enough not to dominate when continuous signals are healthy.
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 20.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return float(total_reward), components
```

# Subagent 数据报告
## 1. 行为事实
- 得分：score=-76.20，range=[-117.55, -12.32]
- 终止方式：terminated 0/20（成功着陆），truncated 20/20（时间上限截断）
- 回合长度：len=1000.00（全部达到最大步数）
- Checkpoint轨迹：官方得分从 cp10%=-92.3 → cp20%=-177.2 → cp30%=-133.5 → cp40%=-126.7 → cp50%=-174.3 → cp60%=-123.8 → cp70%=-120.2 → cp80%=-82.0 → cp90%=-65.2 → cp100%=-71.4（先剧烈下降后逐步回升，最终收敛于-71.4）

## 2. 骨架演进表
| iter | score | len | term | goal_prox_ep_sum | goal_prox_active | vel_pen_ep_sum | vel_pen_active | upr_pen_ep_sum | upr_pen_active | contact_bonus_ep_sum | contact_bonus_active | land_ap_ep_sum | land_ap_active |
|------|-------|------|------|------------------|------------------|----------------|----------------|----------------|----------------|----------------------|-----------------------|----------------|----------------|
| 1 | -76.56 | 975.15 | 1/20 | 0.64 | 100.0% | -12.56 | 100.0% | -3.71 | 100.0% | 0.00 | 0.0% | — | — |
| 2 | 63.72 | 668.05 | 17/20 | 1.84 | 98.8% | -9.52 | 99.0% | -7.30 | 100.0% | 1075.63 | 5.2% | — | — |
| 3 | 21.45 | 864.55 | 9/20 | 1.77 | 99.9% | -9.10 | 99.9% | -11.45 | 100.0% | 1011.47 | 8.2% | — | — |
| 4 | -10.11 | 1000.00 | 0/20 | 2.92 | 100.0% | -7.84 | 100.0% | -5.11 | 100.0% | 0.00 | 0.0% | — | — |
| 5 | -33.04 | 1000.00 | 0/20 | 4.29 | 100.0% | -10.12 | 100.0% | -3.60 | 100.0% | 0.00 | 0.0% | — | — |
| 6 | -81.80 | 1000.00 | 0/20 | 1.77 | 100.0% | -2.39 | 100.0% | -3.08 | 100.0% | — | — | 0.97 | 98.3% |
| 7 | -5.37 | 964.10 | 2/20 | 4.75 | 100.0% | -4.62 | 100.0% | -5.45 | 100.0% | 95.90 | 0.7% | — | — |
| 8 | -76.20 | 1000.00 | 0/20 | 2.73 | 100.0% | -5.31 | 100.0% | -4.19 | 100.0% | 0.00 | 0.0% | — | — |

## 3. 数字标记
- **死组件**：`contact_bonus`，active_rate = 0.0%（连续三轮处于0%）
- **全覆盖惩罚**：
  - `velocity_penalty`：active_rate=100.0%，signed_share=-35.5%（负贡献全覆盖）
  - `upright_penalty`：active_rate=100.0%，signed_share=-28.0%（负贡献全覆盖）
- **量级对比**（每步均值 = ep_sum / len）：
  - 最大两个：`goal_proximity`（+0.00273），`contact_bonus`（0.00000，死组件）
  - 最小两个：`velocity_penalty`（-0.00531），`upright_penalty`（-0.00419）
- **修改变化**（iter 7 → iter 8 的 Δ）：
  - `goal_proximity`：ep_sum -2.03，active_rate 不变
  - `velocity_penalty`：ep_sum -0.69（更负），active_rate 不变
  - `upright_penalty`：ep_sum +1.26（惩罚减轻），active_rate 不变
  - `contact_bonus`：ep_sum -95.90（完全消失），active_rate -0.7%（从0.7%降至0%）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分（ep_sum） | 方向切换 |
|---------|----------|------------------|---------|
| 接近/进度信号 | goal_proximity（所有轮次） | iter1:0.64 → 2:1.84 → 3:1.77 → 4:2.92 → 5:4.29 → 6:1.77 → 7:4.75 → 8:2.73 | 改善量→状态值→改善量：2次切换 |
| 着陆/接触信号 | contact_bonus（iter1-5,7-8） → landing_approach（iter6） → contact_bonus（iter7-8） | iter1:0.0 → 2:1075.6 → 3:1011.5 → 4:0.0 → 5:0.0 → 6:0.97（land_ap）→ 7:95.9 → 8:0.0 | 稀疏bonus→连续奖励→稀疏bonus：2次切换 |
| 稳定性/约束信号 | velocity_penalty + upright_penalty（所有轮次） | vel_pen: -12.6,-9.5,-9.1,-7.8,-10.1,-2.4,-4.6,-5.3；up_pen: -3.7,-7.3,-11.5,-5.1,-3.6,-3.1,-5.5,-4.2 | 无对立设计切换（始终为连续惩罚，velocity一直带高度门控） |

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