第 11 轮迭代训练已经结束，得分 -7.587057（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(11)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

实验记录位于 `rpv4_old_v3/seed_2/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_11, score=-7.587057)
# Training Feedback

## Final-policy outcome
score=-7.587057, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-40.994947, 28.904234]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 26.426158 | 63.1% | 71.9% | 100.0% |
| fuel_penalty | -10.000000 | -23.9% | 23.9% | 100.0% |
| vel_damping | -1.478922 | -3.5% | 3.5% | 100.0% |
| angle_stab | -0.290908 | -0.7% | 0.7% | 100.0% |
| landing_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
### Overall score
| pct | score_mean | score_std |
|---|---|---|
| 40% | -33.07 | 0.00 |
| 80% | -6.78 | 0.00 |

### Per-component ep_sum_mean at each checkpoint
| pct | angle_stab (ep_sum) | fuel_penalty (ep_sum) | landing_quality (ep_sum) | progress (ep_sum) | vel_damping (ep_sum) |
|---|---||---||---||---||---|
| 40% | -0.299 | -10.000 | 0.000 | 21.169 | -0.896 |
| 80% | -0.265 | -9.984 | 4.348 | 26.193 | -1.086 |

### Per-component active_rate at each checkpoint
| pct | angle_stab (active%) | fuel_penalty (active%) | landing_quality (active%) | progress (active%) | vel_damping (active%) |
|---|---||---||---||---||---|
| 40% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| 80% | 100.0% | 99.8% | 0.2% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    
    Components:
        A: progress (distance reduction towards target)       -- main learning signal (STRENGTHENED)
        B: velocity damping (next_obs squared velocity)       -- stability constraint
        C: orientation stabilization (next_obs angle/angvel)  -- stability constraint
        D: fuel penalty (per action cost)                     -- efficiency / mild action cost
        E: landing_quality (per-step reward while landed)     -- continuous soft-landing signal (L2: was sparse bonus)
    
    iter_11 change (L2): Replaced sparse success_landing_bonus (transition-triggered, 
    w=100 one-time) with continuous landing_quality reward (per-step while both feet 
    on ground, scaled by proximity/speed/tilt quality). This prevents exploitation 
    via bouncing (iter_10: 61 triggers/episode at w=100 → ep_sum=6100, 99.3% share)
    and restores gradient to dense shaping signals.
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 20.0  # iter_10: verified boostable, kept at 20
    component_A = w_progress * progress

    # Component B: velocity damping – penalise large speeds in next step
    vx_n, vy_n = next_obs[2], next_obs[3]
    vel_penalty = vx_n**2 + vy_n**2
    w_vel = 0.05
    component_B = -w_vel * vel_penalty

    # Component C: orientation stabilization – penalise tilt and rotation
    angle_n, angvel_n = next_obs[4], next_obs[5]
    ang_penalty = angle_n**2 + angvel_n**2
    w_ang = 0.05
    component_C = -w_ang * ang_penalty

    # Component D: fuel penalty – light cost for any engine use
    fuel_cost = 0.0 if action == 0 else 1.0
    w_fuel = 0.01
    component_D = -w_fuel * fuel_cost

    # Component E: landing quality – per-step reward while both feet are on ground
    # Replaces the sparse transition-based bonus that was exploited via bouncing.
    # Reward is continuous: every step with both feet down earns quality-scaled reward.
    # Quality drops to 0 for bad landings (high speed, tilt, or far from center).
    # Bouncing interrupts reward → no incentive to bounce.
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]
    curr_both_contact = (n_left_contact > 0.5) and (n_right_contact > 0.5)

    # Landing quality metrics (0=perfect, threshold=zero-quality)
    landing_dist = next_dist  # already computed above
    landing_speed = (vx_n**2 + vy_n**2) ** 0.5
    landing_tilt = abs(angle_n)

    # Quality scores: 1.0 at perfect, linear decay to 0 at threshold
    prox_score = max(0.0, 1.0 - landing_dist / 0.4)      # 0 at dist≥0.4
    speed_score = max(0.0, 1.0 - landing_speed / 1.0)    # 0 at speed≥1.0
    tilt_score = max(0.0, 1.0 - landing_tilt / 0.5)      # 0 at tilt≥0.5

    landing_quality = prox_score * speed_score * tilt_score  # 0.0 to 1.0

    w_landing = 5.0  # per-step reward when landed; ~100 total over 20 stable steps
    component_E = w_landing * landing_quality if curr_both_contact else 0.0

    total_reward = component_A + component_B + component_C + component_D + component_E

    components = {
        "progress": component_A,
        "vel_damping": component_B,
        "angle_stab": component_C,
        "fuel_penalty": component_D,
        "landing_quality": component_E
    }

    return float(total_reward), components
```

# Subagent 数据报告
# 训练数据报告

## 1. 行为事实

- **得分**（官方环境）：score_mean=-7.59，range=[-40.99, 28.90]
- **终止方式**：terminated=0/20（零成功着陆），truncated=20/20（全部达到时间上限 1000 步）
- **回合长度**：len=1000.00（全部跑满）
- **Checkpoint 轨迹**（10%步数间隔官方得分）：cp40=-33.07 → cp80=-6.78（上升但仍为负）。两个点均无成功着陆，landing_quality 仅在 cp80 处被记录到 4.348 ep_sum（active_rate=0.2%），说明脚触地行为在训练后期偶尔出现，但在最终策略中完全消失（active_rate=0.0%）。

---

## 2. 骨架演进表

```
| iter |  score  |  len   | term(T/t) | progress(es)  | progress(ar) | fuel_penalty(es) | fuel_penalty(ar) | vel_damping(es) | vel_damping(ar) | angle_stab(es) | angle_stab(ar) | succ_bonus(es) | succ_bonus(ar) | landing_q(es) | landing_q(ar) |
|------|---------|--------|-----------|---------------|--------------|---------------------|---------------------|-----------------|-----------------|----------------|----------------|----------------|----------------|---------------|---------------|
|  1   | -155.98 | 154.40 |  20/0     | 1.58(p_shp)   | 100.0%       | -1.80(act_cost)     | 58.4%(act_cost)     | —               | —               | -1.14(ang_pen) | 100.0%         | —              | —              | —             | —             |
|  2   | -119.51 |  68.30 |  20/0     | 5.58(p_shp)   | 100.0%       | -0.07(act_cost)     | 5.2%(act_cost)      | —               | —               | -0.20(ang_pen) | 100.0%         | 5.00           | 0.1%           | —             | —             |
|  3   | -203.45 | 154.15 |  20/0     | 1.49(p_shp)   | 100.0%       | -1.47(act_cost)     | 47.7%(act_cost)     | —               | —               | -2.46(ang_pen) | 100.0%         | 0.00           | 0.0%           | —             | —             |
|  4   | -114.38 |  68.45 |  20/0     | 5.60(p_shp)   | 100.0%       | -0.10(act_cost)     | 7.1%(act_cost)      | —               | —               | -0.09(ang_pen) | 100.0%         | 30.00          | 0.4%           | —             | —             |
|  5   |  -46.51 | 740.15 |   7/13    | 5.10(p_shp)   | 99.9%        | -13.47(act_cost)    | 91.0%(act_cost)     | —               | —               | -6.83(ang_pen) | 100.0%         | 1195.00        | 1.6%           | —             | —             |
|  6   | -111.18 |  68.45 |  20/0     | 5.60(p_shp)   | 100.0%       | -0.07(act_cost)     | 4.9%(act_cost)      | —               | —               | -0.02(ang_pen) | 100.0%         | 0.00           | 0.0%           | —             | —             |
|  7   |  -98.93 |  69.60 |  20/0     | 5.55(p_shp)   | 100.0%       | -0.18(act_cost)     | 12.8%(act_cost)     | —               | —               | -0.22(ang_pen) | 100.0%         | 5.00           | 0.1%           | —             | —             |
|  8   |   26.45 | 206.50 |  20/0     | 2.67(progress)| 100.0%       | -1.23(fuel_pen)     | 59.6%(fuel_pen)     | -1.58           | 99.9%           | -0.26          | 100.0%         | —              | —              | —             | —             |
|  9   |  145.17 | 718.80 |   9/11    | 2.64(progress)| 99.2%        | -4.27(fuel_pen)     | 59.4%(fuel_pen)     | -2.02           | 99.2%           | -0.75          | 100.0%         | 5330.00        | 7.4%           | —             | —             |
| 10   |  131.35 | 831.25 |  11/9     | 24.70(progress)| 100.0%      | -7.93(fuel_pen)     | 95.4%(fuel_pen)     | -1.53           | 99.9%           | -0.56          | 100.0%         | 6100.00        | 7.3%           | —             | —             |
| 11   |   -7.59 |1000.00 |   0/20    | 26.43(progress)| 100.0%      | -10.00(fuel_pen)    | 100.0%(fuel_pen)    | -1.48           | 100.0%          | -0.29          | 100.0%         | —              | —              | 0.00          | 0.0%          |
```

注：iter_01-07 的 `action_cost` 是 `fuel_penalty` 的前身，`angle_penalty` 是 `angle_stab` 的前身（iter_01 无 vel_damping）。`progress_shaping` 在 iter_08 后改名 `progress`，数学形式一致（均为 dist - next_dist × w）。es = episode_sum_mean，ar = active_rate。

---

## 3. 数字标记

- **死组件**：`landing_quality` active_rate = 0.0%（ep_sum=0.00），20 个评估回合中完全没有触发。代码条件为双足同时接触地面（`curr_both_contact`），意味着策略从未使两足同时落地。
- **主导**：无单一组件 magnitude_share > 80%。`progress` 占比最高，为 71.9%。
- **淹没**：无。
- **全覆盖惩罚**：无。`fuel_penalty` active_rate=100%，signed_share=-23.9%，不满足"全覆盖惩罚"定义（需 >95% 且 signed_share 为负——虽然满足，但 magnitude_share 仅 23.9%，远非全覆盖）。
- **量级对比**：
  - 每步均值（ep_sum ÷ len）最大的两个组件：`progress`（0.0264/步）、`fuel_penalty`（-0.0100/步，按绝对值0.0100）。
  - 每步均值最小的两个组件：`angle_stab`（-0.000291/步）、`landing_quality`（0.000/步）。
- **修改变化（iter_10 → iter_11）**：
  - `progress` ep_sum: 24.704 → 26.426 (Δ=+1.722), active_rate: 100.0% → 100.0% (Δ=0.0pp)
  - `fuel_penalty` ep_sum: -7.934 → -10.000 (Δ=-2.066), active_rate: 95.4% → 100.0% (Δ=+4.6pp)
  - `vel_damping` ep_sum: -1.529 → -1.479 (Δ=+0.050), active_rate: 99.9% → 100.0% (Δ=+0.1pp)
  - `angle_stab` ep_sum: -0.564 → -0.291 (Δ=+0.273), active_rate: 100.0% → 100.0% (Δ=0.0pp)
  - `success_landing_bonus` → 被 `landing_quality` 替换：无法直接对比，但新组件 active_rate 从 7.3% 骤降至 0.0%，ep_sum 从 +6100.00 骤降至 0.00。足见策略失去了所有"着陆相关"的奖励获取能力。

---

## 4. 语义演化

### 接近/进度信号
| 语义功能 | 各轮组件名 | iter_01-11 得分趋势 | 方向切换 |
|---------|----------|---------|---------|
| 距离缩减正奖励 | progress_shaping → progress（iter_08改名） | 1.58→5.58→1.49→5.60→5.10→5.60→5.55→2.67→2.64→24.70→26.43 | 权重切换：5.0→2.0→20.0（2次切换）。始终为连续delta形式（dist - next_dist），无状态值↔改善量的切换 |

- 此功能自 iter_01 至今保持同一数学范式：Δ势能。iter_08 权重从 5.0 降为 2.0，iter_10 权重跳升至 20.0。在 iter_10-11（w=20.0）下 ep_sum 跃升至 24-26 量级，成为唯一的正信号支柱。

### 着陆/接触信号
| 语义功能 | 各轮组件名 | iter_01-11 得分趋势 | 方向切换 |
|---------|----------|---------|---------|
| 着陆成功/接触奖励 | landing_velocity_penalty → success_landing_bonus → crash_penalty + success_landing_bonus → success_landing_bonus → landing_quality | -0.07(v1)→5.00(v2)→0.00(v3)→30.00(v4)→1195.00(v5)→0.00(v6)→5.00(v7)→—(v8)→5330.00(v9)→6100.00(v10)→0.00(v11) | 连续惩罚→稀疏bonus→稀疏bonus+对称惩罚→稀疏bonus（放宽阈值4次）→连续per-step质量奖励。方向切换："连续/稀疏/连续/稀疏/连续"共5次切换。v11 归零。 |

- 此功能从 v2 起成为"得分波动轴心"。v5 通过极度放宽阈值使 active_rate 达 1.6%、ep_sum 达 1195，但触发不足导致 agent 学会"反复弹跳触发"（v9-v10 每 episode 触发 53-61 次）。v11 从稀疏转变切换为连续每步接触奖励，但 active_rate 直接归零——策略不再使双足同时着地。这是 v11 得分崩溃的直接原因：失去了历史上唯一的大正信号。

### 稳定性/约束信号
| 语义功能 | 各轮组件名 | 各轮 ep_sum 趋势 | 方向切换 |
|---------|----------|---------|---------|
| 速度阻尼 | vel_damping（iter_08新增） | —→-1.58→-2.02→-1.53→-1.48 | 始终为连续负惩罚。无切换。 |
| 角度/角速度约束 | angle_penalty → angle_stab（iter_08改名，并加入角速度项） | -1.14→-0.20→-2.46→-0.09→-6.83→-0.02→-0.22→-0.26→-0.75→-0.56→-0.29 | iter_08 从仅角度→角度+角速度，惩罚形式保持二次。无"门控/全高度"切换。权重保持 0.05（iter_01 为 0.5）。 |
| 能量效率 | action_cost → fuel_penalty（iter_08改名） | -1.80→-0.07→-1.47→-0.10→-13.47→-0.07→-0.18→-1.23→-4.27→-7.93→-10.00 | 始终为连续负惩罚。iter_08 后权重统一为 0.01 per action，ep_sum 增长主要由 active_rate 和 episode 长度驱动（len: 1000→fuel_cost 必然达 -10.00 天花板）。无切换。 |

### 完成/成功代理
此语义功能与"着陆/接触信号"高度重叠，不再单独列出。历史上 `success_landing_bonus` 充当了这一角色，v11 替换为 `landing_quality` 但已完全失活。

---

**v11 关键事实**：`landing_quality` 替换 `success_landing_bonus` 后完全失效（active_rate=0.0%）。策略在 1000 步中未使双足同时着地一次。所有 20 回合被 truncate。历史上唯一的正信号源被切断，整个 reward 函数只剩 progress（+26.4）对抗 fuel_penalty（-10.0）+ vel_damping（-1.48）+ angle_stab（-0.29），导致得分 -7.59。

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    Components:
        A: progress (distance reduction towards target)  -- main learning signal
        B: velocity damping (next_obs squared velocity)  -- stability constraint
        C: orientation stabilization (next_obs angle/angular velocity) -- stability constraint
        D: fuel penalty (per action cost)                -- efficiency / mild action cost
        E: success_landing_bonus (transition-based)      -- terminal success signal, one-time
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 2.0
    component_A = w_progress * progress

    # Component B: velocity damping – penalise large speeds in next step
    vx_n, vy_n = next_obs[2], next_obs[3]
    vel_penalty = vx_n**2 + vy_n**2
    w_vel = 0.05
    component_B = -w_vel * vel_penalty

    # Component C: orientation stabilization – penalise tilt and rotation
    angle_n, angvel_n = next_obs[4], next_obs[5]
    ang_penalty = angle_n**2 + angvel_n**2
    w_ang = 0.05
    component_C = -w_ang * ang_penalty

    # Component D: fuel penalty – light cost for any engine use
    fuel_cost = 0.0 if action == 0 else 1.0
    w_fuel = 0.01
    component_D = -w_fuel * fuel_cost

    # Component E: success landing bonus – one-time bonus on foot contact transition
    # Trigger: both support legs transition from OFF to ON in a single step
    # AND soft-landing conditions are met (moderate thresholds to avoid dead component)
    left_contact = obs[6]
    right_contact = obs[7]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    prev_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    curr_both_contact = (n_left_contact > 0.5) and (n_right_contact > 0.5)
    contact_made = curr_both_contact and (not prev_both_contact)

    # Soft landing conditions: near target, slow, upright
    landing_dist = (nx**2 + ny**2) ** 0.5
    landing_speed = (vx_n**2 + vy_n**2) ** 0.5
    landing_tilt = abs(angle_n)

    is_soft_landing = (landing_dist < 0.4) and (landing_speed < 1.0) and (landing_tilt < 0.5)

    w_success = 100.0
    component_E = w_success if (contact_made and is_soft_landing) else 0.0

    total_reward = component_A + component_B + component_C + component_D + component_E

    components = {
        "progress": component_A,
        "vel_damping": component_B,
        "angle_stab": component_C,
        "fuel_penalty": component_D,
        "success_landing_bonus": component_E
    }

    return float(total_reward), components
```

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
这是一个 2D 的飞行器/着陆器轨迹优化任务。智能体从一个靠近顶部中央的位置出发，带有随机初始作用力。  
**主目标**：尽快到达画面中央的目标平台，并安全停靠（soft landing），即要求位置收敛到平台上、速度很低、姿态竖直、支撑脚安全触地。  
**次目标**：在完成任务的前提下，尽量少使用引擎推力（节省燃料），同时保持姿态稳定。  
**不应混淆的目标**：该任务不是纯生存或平衡任务——到达目标平台是刚性的成功条件，能量节省只是辅助约束，不是多目标的平等折衷。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: [8]
- **dtype**: float32 (推断)
- **obs[0]**: `x_position`，水平坐标（相对于目标平台中心），用于衡量横向偏移，奖励可用 ✅
- **obs[1]**: `y_position`，垂直坐标（相对于平台高度），用于衡量高度/下降进程，奖励可用 ✅
- **obs[2]**: `x_velocity`，水平线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[3]**: `y_velocity`，垂直线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[4]**: `body_angle`，机体角度（可能以弧度为单位，0 表示竖直），奖励可用 ✅
- **obs[5]**: `angular_velocity`，角速度，用于稳定姿态，奖励可用 ✅
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `0`: `no_engine` – 不激活任何引擎，仅靠惯性飞行。
  - `1`: `left_orientation_engine` – 点火左定向引擎，产生一个使机体逆时针（或某一方向）旋转的力矩。
  - `2`: `main_engine` – 点火主引擎，产生向上的推力（对抗重力并控制下降速度）。
  - `3`: `right_orientation_engine` – 点火右定向引擎，产生与动作 1 相反的旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `MASKED_STEP_SOURCE` 的 `terminated` 计算逻辑：
- **crash_or_body_contact**：躯体发生碰撞或身体支撑点接触地面。此情景可能包括：
  - 成功的软着陆（接触时速度极低、姿态正）。
  - 失败的高速撞击/翻倒等。
- **horizontal_position_outside_viewport**：x 坐标超出视区边界，明确失败。
- **body_not_awake_or_settled**：躯体进入睡眠状态或已稳定（settled）。这可能是：
  - 成功着陆后自然睡眠。
  - 坠毁后物理引擎判定身体安定，属于失败。

**总结终止模式**：
- **success-like termination**：`body_not_awake_or_settled` 在速度接近于 0、位置接近目标平台、接触标志为 1 且角度接近 0 时发生。
- **failure-like termination**：`horizontal_position_outside_viewport`；或因高速、大角度导致的 `crash_or_body_contact`/`body_not_awake_or_settled`。
- **ambiguous termination**：`crash_or_body_contact`/`body_not_awake_or_settled` 需结合 next_obs 判定成败。
- **truncation**：未提及，但有最大 episode 长度（通常存在），应在 reward 中注意时间步惩罚。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无任何显式标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因为没有提供）

> 因此，任何成功/失败的判断必须从 `next_obs` 与 `terminated` 的组合中推断，而不能依赖 `info`。

## 7. 可用于奖励函数的信号
- **position**：`x_position`, `y_position`（相对目标），可计算到目标的欧氏距离或分别控制。
- **velocity**：`x_velocity`, `y_velocity`，可设计近地减速、稳定约束。
- **orientation**：`body_angle`，鼓励竖直。
- **angular**：`angular_velocity`，鼓励旋转抑制。
- **contact**：`left_support_contact`, `right_support_contact`，标志着陆接触已发生，可配合速度用于判定软着陆。
- **action/engine**：`action` 选择，可用于惩罚引擎使用（燃料节省）。
- **other**：`terminated` 标志（作为环境返回，但 `compute_reward` 接口未直接提供，可通过在调用 `compute_reward` 时传入 `done` 标志获取；我们可以在训练循环中以额外参数传递，但接口中只有 `next_obs` 和 `info`；`terminated` 通常可在调用时通过另一个变量获得，若必须在 `compute_reward` 中获取，需要从外部传入——在此假设可以在调用时提供 `done` 标志，否则可通过 `next_obs` 的特征推断终止状态。由于题干未明确禁止，但谨慎起见，直接使用 `done` 需要额外参数，在标准接口中不鼓励，所以这里只强调可从 `next_obs` 和 `info` 推断。但 info 空，所以成功稀疏奖励只能基于 `next_obs` 的状态是否符合成功条件给出，而不依赖外部 `done`。所以我们仍可以将“是否处于终止且状态符合成功条件”作为奖励计算的一部分，但必须在训练循环的每个 step 结束时调用，需要能够判断 `done`，这部分由训练框架负责。我们此处声明信号可用，但不具体到实现细节。）