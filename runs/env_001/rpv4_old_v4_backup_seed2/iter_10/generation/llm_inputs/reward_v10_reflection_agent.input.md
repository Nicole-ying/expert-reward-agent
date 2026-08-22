第 9 轮迭代训练已经结束，得分 186.537439（目标: 200）。

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

实验记录位于 `rpv4_old_v4/seed_2/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_9, score=186.537439)
# Training Feedback

## Final-policy outcome
score=186.537439, len=260.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[14.384539, 288.210746]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 19.171905 | 53.2% | 53.2% | 19.4% |
| descent_bonus | 6.855863 | 19.0% | 19.0% | 84.5% |
| distance_progress | 6.257078 | 17.4% | 18.6% | 96.8% |
| stability | -1.922777 | -5.3% | 5.3% | 100.0% |
| fuel_penalty | -1.383500 | -3.8% | 3.8% | 53.0% |

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
| -106.3 | 55.0 | 22.1 | -1.3 | -4.8 | -23.9 | -22.9 | 39.4 | 125.8 | 188.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 6.499 | 6.415 | 6.472 | 6.823 | 7.193 | 7.296 | 7.340 | 7.241 | 7.018 | 6.826 |
| distance_progress | 5.561 | 5.397 | 5.473 | 5.910 | 6.117 | 6.151 | 6.210 | 6.219 | 6.161 | 6.338 |
| fuel_penalty | -0.098 | -1.144 | -0.652 | -0.585 | -0.662 | -0.582 | -0.582 | -0.788 | -1.568 | -1.359 |
| landing_completion | 0.000 | 4.060 | 0.101 | 0.037 | 0.008 | 0.000 | 0.000 | 3.212 | 48.055 | 19.827 |
| stability | -3.071 | -2.864 | -2.723 | -2.426 | -2.197 | -2.284 | -2.345 | -1.983 | -1.797 | -1.862 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 94.3% | 69.8% | 92.0% | 96.2% | 96.7% | 96.6% | 96.5% | 92.5% | 66.7% | 83.9% |
| distance_progress | 100.0% | 98.5% | 99.3% | 100.0% | 100.0% | 100.0% | 100.0% | 99.0% | 98.0% | 96.5% |
| fuel_penalty | 14.3% | 58.5% | 57.2% | 56.5% | 55.3% | 51.3% | 51.6% | 55.9% | 55.7% | 53.0% |
| landing_completion | 0.0% | 7.2% | 1.0% | 1.0% | 0.1% | 0.0% | 0.0% | 7.2% | 37.7% | 20.4% |
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
    #
    #    v9 change: x_gate widened from /0.5 to /0.8 to give partial descent
    #    incentive to agents moderately offset horizontally (0.5<|x|<0.8),
    #    helping them descend while centering instead of requiring perfect
    #    alignment first.  This targets the 3/20 truncated episodes from v8.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.8)          # 1 at x=0, 0 at |x|>=0.8
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 6.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: small soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
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
# 训练数据报告 — 本轮 (iter=9)

---

## 1. 行为事实

- **得分**：score=186.54，range=[14.38, 288.21]
- **终止方式**：terminated=20/20（成功），truncated=0/20（时间截断）
- **回合长度**：len=260.90
- **Checkpoint 轨迹**（官方得分）：cp10%=-106.3 → cp20%=55.0 → cp30%=22.1 → cp40%=-1.3 → cp50%=-4.8 → cp60%=-23.9 → cp70%=-22.9 → cp80%=39.4 → cp90%=125.8 → cp100%=188.1（总体上升，cp30-70 经历性能低谷）

---

## 2. 骨架演进表

（注：iter 1-5 骨架结构与 iter 6-9 不同，分别列出）

| iter | score | len | term | progress_ep_sum | progress_active | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | landing_ep_sum | landing_active | velocity_guidance_ep_sum | velocity_guidance_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -203.53 | 185.80 | 20T/0Tr | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | 0.000 | 0.0% | — | — |
| 2 | -37.03 | 962.55 | 4T/16Tr | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | 697.436 | 99.6% | — | — |
| 3 | -222.85 | 209.60 | 20T/0Tr | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | 0.000 | 0.0% | — | — |
| 4 | -295.33 | 138.90 | 20T/0Tr | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | — | — | -1.596 | 100.0% |
| 5 | -197.46 | 185.50 | 20T/0Tr | -0.183 | 100.0% | -0.717 | 15.8% | -0.343 | 100.0% | 0.000 | 0.0% | — | — |

| iter | score | len | term | distance_progress_ep_sum | distance_progress_active | stability_ep_sum | stability_active | descent_bonus_ep_sum | descent_bonus_active | landing_completion_ep_sum | landing_completion_active | fuel_penalty_ep_sum | fuel_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | -13.57 | 1000.00 | 0T/20Tr | 6.331 | 100.0% | -1.416 | 100.0% | — | — | 0.000 | 0.0% | -9.889 | 98.9% |
| 7 | 146.32 | 673.35 | 14T/6Tr | 5.730 | 99.5% | -3.284 | 100.0% | 3.276 | 62.8% | 47.039 | 19.0% | -4.572 | 67.9% |
| 8 | 199.42 | 477.65 | 17T/3Tr | 6.304 | 98.5% | -1.871 | 100.0% | 6.377 | 69.7% | 70.983 | 32.3% | -3.051 | 63.9% |
| **9** | **186.54** | **260.90** | **20T/0Tr** | **6.257** | **96.8%** | **-1.923** | **100.0%** | **6.856** | **84.5%** | **19.172** | **19.4%** | **-1.384** | **53.0%** |

*T=terminated, Tr=truncated*

---

## 3. 数字标记

- **主导**：无。landing_completion magnitude_share=53.2%，未超过 80%。
- **死组件**：无。所有组件 active_rate ≥ 19.4%（最低为 landing_completion=19.4%）。
- **淹没**：无。stability active_rate=100% 但 magnitude_share=5.3%，未低于 3%；distance_progress active_rate=96.8%，magnitude_share=18.6%，未低于 3%。
- **全覆盖惩罚**：stability signed_share 为负（-5.3%），active_rate=100%。符合条件（active_rate>95% 且 signed_share 为负）。
- **量级对比（每步均值 = ep_sum ÷ len）**：
  - 最大两个：landing_completion=0.0735，descent_bonus=0.0263
  - 最小两个：fuel_penalty=-0.0053，stability=-0.0074
- **修改变化（iter 8 → iter 9 Δ）**：
  - distance_progress: ep_sum Δ=-0.047, active_rate Δ=-1.7%
  - stability: ep_sum Δ=-0.052, active_rate Δ=0.0%
  - descent_bonus: ep_sum Δ=+0.479, active_rate Δ=+14.8%
  - landing_completion: ep_sum Δ=-51.811, active_rate Δ=-12.9%
  - fuel_penalty: ep_sum Δ=+1.667, active_rate Δ=-10.9%

---

## 4. 语义演化（跨轮组件功能追踪）

### 接近/进度信号

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向/设计切换 |
|---|---|---|---|---|
| 距离/进度引导 | iter1: progress → iter2: progress → iter3: progress → iter4: progress → iter5: progress → iter6: distance_progress → iter7: distance_progress → iter8: distance_progress → iter9: distance_progress | -0.200 → 0.665 → -0.299 → -0.026 → -0.183 → 6.331 → 5.730 → 6.304 → 6.257 | 100% → 100% → 100% → 100% → 100% → 100% → 99.5% → 98.5% → 96.8% | iter6 重构：原始距离差 → 5×缩放；iter4 追加 velocity_guidance（速度方向引导，iter5 移除）|
| 垂直下降激励 | iter6: descent_bonus（无）→ iter7: descent_bonus → iter8: descent_bonus → iter9: descent_bonus | — → 3.276 → 6.377 → 6.856 | — → 62.8% → 69.7% → 84.5% | iter7 引入（系数 6.0, x_gate /0.5）；iter9 x_gate 从 /0.5 扩至 /0.8 |
| 速度方向引导 | iter4: velocity_guidance → iter5: 移除 | -1.596 → — | 100% → — | 1 次引入→移除 |

### 着陆/接触信号

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向切换 |
|---|---|---|---|---|
| 着陆/接触奖励 | iter1: landing → iter2: landing → iter3: landing → iter4: 移除 → iter5: landing → iter6: landing_completion → iter7: landing_completion → iter8: landing_completion → iter9: landing_completion | 0.000 → 697.436 → 0.000 → — → 0.000 → 0.000 → 47.039 → 70.983 → 19.172 | 0% → 99.6% → 0% → — → 0% → 0% → 19.0% → 32.3% → 19.4% | 稀疏bonus↔连续奖励：4次切换（iter1 接触门控相乘 → iter2 连续 proximity-gated sum-of-factors → iter3 接触门控 quality-scaled → iter5 proximity-gated contact-multiplier → iter6 硬条件 sparse bonus）；iter7-9 稳定为硬条件 sparse bonus |

### 稳定性/约束信号

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向切换 |
|---|---|---|---|---|
| 速度约束 | iter1: speed_penalty → iter2: speed_penalty → iter3: speed_penalty → iter4: speed_penalty → iter5: speed_penalty → iter6: stability → iter7: stability → iter8: stability → iter9: stability | -0.669 → -42.130 → -0.735 → -2.097 → -0.717 → -1.416 → -3.284 → -1.871 → -1.923 | 15.5% → 100% → 14.8% → 63.2% → 15.8% → 100% → 100% → 100% → 100% | iter6 重构：线性 closeness-gated speed 惩罚 → exponential-gated 平方惩罚（含 angle, ang_vel）；iter1-5 为距离门控线性惩罚，iter6-9 为 exp(-2d) 门控平方惩罚 |
| 姿态约束 | iter1: angular_vel_penalty → iter2: angular_vel_penalty → iter3: angular_vel_penalty → iter4: angular_vel_penalty → iter5: angular_vel_penalty → iter6: (合并入 stability) | -0.434 → -2.457 → -0.368 → -1.266 → -0.343 → 合并 | 100% → 100% → 100% → 100% → 100% → 合并 | iter6 将角速度惩罚合并入 stability 组件（angle + ang_vel 平方） |
| 燃料约束 | iter6: fuel_penalty → iter7: fuel_penalty → iter8: fuel_penalty → iter9: fuel_penalty | -9.889 → -4.572 → -3.051 → -1.384 | 98.9% → 67.9% → 63.9% → 53.0% | 无切换，稳定为恒定 -0.01 per action |

### 完成/成功代理

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向切换 |
|---|---|---|---|---|
| 成功代理 | iter1-5: landing（同上）→ iter6: landing_completion → iter7: landing_completion → iter8: landing_completion → iter9: landing_completion | 见着陆/接触信号行 | 见着陆/接触信号行 | iter1-5 landing 在 0 和非零之间振荡；iter6-9 landing_completion 为硬条件乘积 sparse bonus |

# 历史最佳奖励函数代码
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
    #
    #    v8 change: coefficient 3.0 → 6.0 to strengthen descent gradient.
    #    The per-step descent amount is naturally small (~0.01–0.05 per step),
    #    and at 3.0 the total per-episode signal (~3.3) was too weak to
    #    compete with other components and actively shape descent behavior.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.5)          # 1 at x=0, 0 at |x|>=0.5
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 6.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: small soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
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