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
第 7 轮迭代训练已经结束，得分 146.319931（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(7)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

实验记录位于 `rpv4_old_v4/seed_2/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_7, score=146.319931)
# Training Feedback

## Final-policy outcome
score=146.319931, len=673.350000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[0.897306, 252.418300]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 47.039200 | 70.5% | 70.5% | 19.0% |
| distance_progress | 5.729716 | 8.6% | 12.9% | 99.5% |
| fuel_penalty | -4.571500 | -6.8% | 6.8% | 67.9% |
| stability | -3.284203 | -4.9% | 4.9% | 100.0% |
| descent_bonus | 3.276273 | 4.9% | 4.9% | 62.8% |

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
| -99.0 | -36.6 | 2.6 | 13.4 | 2.2 | -14.4 | 30.9 | 148.8 | 105.3 | 179.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 2.621 | 2.519 | 2.662 | 2.738 | 2.942 | 3.121 | 3.180 | 3.340 | 3.412 | 3.224 |
| distance_progress | 5.532 | 4.882 | 5.325 | 5.763 | 6.068 | 6.233 | 6.429 | 5.953 | 5.800 | 5.901 |
| fuel_penalty | -0.131 | -0.576 | -0.644 | -0.638 | -0.683 | -0.697 | -1.034 | -4.935 | -4.856 | -3.868 |
| landing_completion | 0.000 | 0.000 | 0.449 | 0.074 | 0.013 | 0.000 | 0.434 | 24.320 | 24.697 | 24.303 |
| stability | -3.293 | -3.195 | -2.335 | -2.094 | -2.008 | -2.240 | -2.436 | -2.815 | -3.251 | -3.408 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 92.9% | 76.0% | 85.5% | 95.6% | 96.3% | 96.5% | 97.6% | 65.3% | 75.5% | 71.8% |
| distance_progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.4% | 99.5% | 98.9% |
| fuel_penalty | 18.9% | 56.1% | 59.1% | 59.2% | 58.0% | 56.5% | 63.6% | 68.7% | 65.2% | 66.2% |
| landing_completion | 0.0% | 0.0% | 2.6% | 1.3% | 0.6% | 0.0% | 2.0% | 9.9% | 12.1% | 11.9% |
| stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar lander-like navigation task.
    Components:
      A. distance_progress   – reduction in Euclidean distance to landing pad
      B. stability           – penalise high velocity and large attitude when close to pad
      C. descent_bonus       – reward vertical descent when horizontally aligned with pad
      D. landing_completion  – small soft success bonus for two-leg contact with low speed
      E. fuel_penalty        – tiny penalty for using any engine
    """
    # --- unpack observations ---
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos = next_obs[0]
    ny_pos = next_obs[1]

    # --- distances ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    velocity_norm = (x_vel**2 + y_vel**2) ** 0.5

    # =====================================================================
    # A. distance_progress: reward reduction in Euclidean distance to target
    # =====================================================================
    progress = (current_distance - next_distance) * 5.0   # scaling factor

    # =====================================================================
    # B. stability: smooth penalty for speed and attitude when near the pad
    #    gate: exp(-2 * distance)  →  close to 1 when distance small
    # =====================================================================
    gate = 2.718281828 ** (-2.0 * current_distance)     # approximate exp(-2d)
    w_vel = 0.1
    w_angle = 0.5
    w_angv = 0.5
    stability_penalty = - (w_vel  * velocity_norm**2 +
                           w_angle * angle**2 +
                           w_angv  * ang_vel**2) * gate

    # =====================================================================
    # C. descent_bonus: reward vertical descent when horizontally aligned
    #    with the landing pad.  This provides continuous gradient from
    #    altitude to ground, bridging the gap between distance_progress
    #    (which is isotropic) and landing_completion (which is sparse).
    #
    #    Key design: rewards the *change* in y (delta-y), NOT the absolute
    #    y position.  Hovering at low altitude gives zero reward → no exploit.
    #    Gate ensures reward only activates when agent is above the pad.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.5)          # 1 at x=0, 0 at |x|>=0.5
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 3.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: small soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
    #    Reduced weight (0.5 instead of 2.0) to avoid reward-spike instability.
    # =====================================================================
    both_feet = left_contact * right_contact            # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 0.5 * speed_quality * angle_quality * angvel_quality
    else:
        landing_reward = 0.0

    # =====================================================================
    # E. fuel_penalty: minuscule cost for any engine use
    # =====================================================================
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = (progress + stability_penalty + descent_bonus
                    + landing_reward + fuel_penalty)

    components = {
        "distance_progress": progress,
        "stability": stability_penalty,
        "descent_bonus": descent_bonus,
        "landing_completion": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# Subagent 数据报告
# 数据报告 — iter_07

## 1. 行为事实
- **得分**：score=146.32，range=[0.90, 252.42]
- **终止方式**：terminated 14/20（着陆/坠毁），truncated 6/20（达到时间上限）
- **回合长度**：len=673.35（截断集均约 1000，终止集均约 533）
- **Checkpoint 轨迹**：官方得分 cp10%=-99.0 → cp20%=-36.6 → cp30%=2.6 → cp40%=13.4 → cp50%=2.2 → cp60%=-14.4 → cp70%=30.9 → cp80%=148.8 → cp90%=105.3 → cp100%=179.9（先升后降再升，后期突破正值）
- **组件组成**（ep_sum_mean）：landing_completion=47.04（70.5% signed share），distance_progress=5.73（8.6%），descent_bonus=3.28（4.9%），stability=-3.28（-4.9%），fuel_penalty=-4.57（-6.8%）
- **活跃率**：distance_progress 99.5%，stability 100.0%，fuel_penalty 67.9%，descent_bonus 62.8%，landing_completion 19.0%
- **步均贡献**（ep_sum ÷ len）：landing_completion=0.0699，distance_progress=0.0085，descent_bonus=0.0049，stability=-0.0049，fuel_penalty=-0.0068

## 2. 骨架演进表

| iter | score | len | term | landing_ep_sum | landing_active | progress_ep_sum | progress_active | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | velocity_guidance_ep_sum | velocity_guidance_active | stability_ep_sum | stability_active | descent_bonus_ep_sum | descent_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | landing_completion_ep_sum | landing_completion_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -203.53 | 185.8 | 20/0/0 | 0.000 | 0.0% | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 2 | -37.03 | 962.6 | 4/16/0 | 697.436 | 99.6% | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 3 | -222.85 | 209.6 | 20/0/0 | 0.000 | 0.0% | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 4 | -295.33 | 138.9 | 20/0/0 | — | — | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | -1.596 | 100.0% | — | — | — | — | — | — | — | — |
| 5 | -197.46 | 185.5 | 20/0/0 | 0.000 | 0.0% | -0.183 | 100.0% | -0.717 | 15.8% | -0.343 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 6 | -13.57 | 1000.0 | 0/20/0 | — | — | 6.331 | 100.0% | — | — | — | — | — | — | -1.416 | 100.0% | 0.000 | 0.0% | -9.889 | 98.9% | 0.000 | 0.0% |
| 7 | 146.32 | 673.4 | 14/6/0 | — | — | 5.730 | 99.5% | — | — | — | — | — | — | -3.284 | 100.0% | 3.276 | 62.8% | -4.572 | 67.9% | 47.039 | 19.0% |

注：progress 在 iter_1-5 中为"progress"，在 iter_6-7 中为"distance_progress"（相同功能，列合并展示）。landing 在 iter_1-5 中为"landing"，在 iter_6-7 中为"landing_completion"（相同功能，列合并展示）。stability 仅在 iter_6-7 出现。

## 3. 数字标记

- **主导**：landing_completion magnitude_share=70.5%（>80%），在 ep_sum 中占绝对主导
- **死组件**：无（所有组件 active_rate ≥ 19.0%）
- **淹没**：无（无组件同时满足 active_rate > 90% 且 magnitude_share < 3%）
- **全覆盖惩罚**：无（无组件满足 active_rate > 95% 且 signed_share 为负；stability active_rate=100% 但 signed_share=-4.9%，未达<-10% 极端负值标准）
- **量级对比**：
  - 步均最大（ep_sum/len）：landing_completion=0.0699，distance_progress=0.0085
  - 步均最小：stability=-0.0049（绝对值排第三），fuel_penalty=-0.0068（绝对值排第二）
- **修改变化（vs iter_06）**：
  - distance_progress：ep_sum 6.33→5.73（Δ=-0.60），active_rate 100.0%→99.5%（Δ=-0.5%）
  - stability：ep_sum -1.42→-3.28（Δ=-1.87），active_rate 100.0%→100.0%（Δ=0）
  - descent_bonus：ep_sum 0.00→3.28（Δ=+3.28），active_rate 0.0%→62.8%（Δ=+62.8%）
  - fuel_penalty：ep_sum -9.89→-4.57（Δ=+5.32），active_rate 98.9%→67.9%（Δ=-31.0%）
  - landing_completion：ep_sum 0.00→47.04（Δ=+47.04），active_rate 0.0%→19.0%（Δ=+19.0%）

## 4. 语义演化

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_progress → v4_progress → v6_distance_progress | -203.53 → -295.33 → 146.32 | 无切换：始终为状态改善量（delta距离），全高度激活 |
| 着陆/接触信号 | v1_landing(死) → v2_landing(主导) → v3_landing(死) → v5_landing(死) → v6_landing_completion(死) → v7_landing_completion | -203.53 → -37.03 → -222.85 → -197.46 → -13.57 → 146.32 | 稀疏bonus↔连续奖励：5次切换（v1乘积稀疏→v2 proximity连续→v3 contact稀疏→v5 proximity连续→v6 双足乘积稀疏→v7 双足乘积稀疏） |
| 稳定性/约束信号 | v1_speed_penalty + v1_angular_vel_penalty → v6_stability | -203.53 → -13.57 | 门控→全高度：1次切换（v1距离门控→v6 exp门控但参数不同，本质仍为门控） |
| 完成/成功代理 | v2_landing(主导) → v7_landing_completion(主导) | -37.03 → 146.32 | 连续奖励→稀疏bonus：1次切换（v2 proximity连续→v7 双足接触稀疏） |

**方向切换汇总**：
- 着陆/接触信号：5次切换（在该功能上的设计振荡最显著）
- 连续奖励 vs 稀疏bonus 在这 5 次中交替出现，v2(连续)→v3(稀疏)→v5(连续)→v6(稀疏)→v7(稀疏)

**注**：v4 的 velocity_guidance 未归入上述任何语义功能——它属于"速度方向引导"这一独立功能，仅在 v4 出现一次后即被移除。

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
主体是一个2D类车辆刚体，从视口上方中央附近出发，带有随机初始作用力。任务是**尽快到达并稳定停靠在中央的目标垫上，同时尽可能减少发动机推力消耗**。理想行为是：平滑接近目标、降低线速度、保持垂直姿态（零角度）、以低速和正确朝向使左右两个支撑点同时接触目标垫并固化。次要目标包含节省燃料和/或快速完成，但不应混淆为主目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (默认) 或 float32
- obs[0]: x_position — 相对目标垫的水平坐标 (reward_usable: true)
- obs[1]: y_position — 相对目标垫垫面高度的垂直坐标 (reward_usable: true)
- obs[2]: x_velocity — 水平线速度 (reward_usable: true)
- obs[3]: y_velocity — 垂直线速度 (reward_usable: true)
- obs[4]: body_angle — 车身朝向角 (reward_usable: true)
- obs[5]: angular_velocity — 角速度 (reward_usable: true)
- obs[6]: left_support_contact — 左支撑脚接触标志，接触为1.0 (reward_usable: true)
- obs[7]: right_support_contact — 右支撑脚接触标志，接触为1.0 (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，靠惯性滑行
- action 1: left_orientation_engine — 点燃左姿态发动机（提供转向力矩）
- action 2: main_engine — 点燃主发动机（提供向上/向前的推力）
- action 3: right_orientation_engine — 点燃右姿态发动机（提供相反力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact: 主体发生碰撞或非预期的身体接触（很可能为失败，但不排除目标垫接触也属于body_contact，需结合后续稳定信号区分）
- horizontal_position_outside_viewport: 水平位置超出视口边界（明确失败）
- body_not_awake_or_settled: 主体不再活动或已稳定（speed/angular接近零且可能接触垫）；此条件可能是成功（settled on pad）或失败（settled elsewhere or falling asleep）
- 没有显式截断（truncation）条件提及，但实际环境可能有最大步数限制，目前未在信息中给出。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info为空字典)
- explicit_failure_flag_available: false
- allowed_info_fields: （当前step返回的info = {}，无可用字段）
- forbidden_or_uncertain_info_fields: 无任何显式success/failure/termination_reason字段；不可使用info进行reward或终止判定

## 7. 可用于奖励函数的信号
- position: x_position, y_position (相对目标垫)  
- velocity: x_velocity, y_velocity  
- orientation: body_angle  
- angular_velocity: angular_velocity  
- contact: left_support_contact, right_support_contact (二值浮点，1.0表示接触)  
- action/engine: action index (可用于动作惩罚，或结合下一步状态推断推力效果)
- other: 无额外传感器
```
