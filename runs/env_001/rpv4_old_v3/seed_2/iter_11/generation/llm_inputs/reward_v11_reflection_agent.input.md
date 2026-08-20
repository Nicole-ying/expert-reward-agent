第 10 轮迭代训练已经结束，得分 131.350093（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(10)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

# 上一轮训练反馈 (iter_10, score=131.350093)
# Training Feedback

## Final-policy outcome
score=131.350093, len=831.250000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[12.949437, 257.366190]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 6100.000000 | 99.3% | 99.3% | 7.3% |
| progress | 24.704294 | 0.4% | 0.5% | 100.0% |
| fuel_penalty | -7.933500 | -0.1% | 0.1% | 95.4% |
| vel_damping | -1.529314 | -0.0% | 0.0% | 99.9% |
| angle_stab | -0.563688 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
### Overall score
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    Components:
        A: progress (distance reduction towards target)  -- main learning signal (STRENGTHENED)
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
    w_progress = 20.0  # iter_10: 2.0 → 20.0 (10x) to counter success_landing_bonus drowning
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

# Subagent 数据报告
# 训练数据报告 — iter_10

---

## 1. 行为事实

- 得分：score=131.35，range=[12.95, 257.37]
- 终止方式：terminated=11/20（成功着陆），truncated=9/20（超时）
- 回合长度：len=831.25
- Checkpoint 轨迹：
  - cp_40%: official_score=0.00（std=0.00）
  - cp_80%: official_score=0.00（std=0.00）
  - 两个checkpoint得分均为0，说明训练过程中间阶段官方得分无变化

---

## 2. 骨架演进表

仅列出有完整组件分解数据的轮次（iter_01 — iter_10）。

> 组件命名已统一为最近两轮使用的名称。`progress_shaping` 与 `progress` 为同一语义功能的不同名称，表中以 `progress` 统一标记。

| iter | score | len | term | progress_ep_sum | progress_active | vel_damping_ep_sum | vel_damping_active | angle_stab_ep_sum | angle_stab_active | fuel_penalty_ep_sum | fuel_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | crash_penalty_ep_sum | crash_penalty_active |
|------|-------|-----|------|----------------|-----------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-----------------------------|---------------------------|-------------------------------|-------------------------------|----------------------|---------------------|
| 1 | -155.98 | 154.40 | 20/0/0 | 1.582 | 100.0% | — | — | -1.141* | 100.0% | -1.804* | 58.4% | — | — | -0.067 | 5.3% | — | — |
| 2 | -119.51 | 68.30 | 20/0/0 | 5.583 | 100.0% | — | — | -0.205 | 100.0% | -0.071* | 5.2% | 5.000 | 0.1% | — | — | — | — |
| 3 | -203.45 | 154.15 | 20/0/0 | 1.488 | 100.0% | — | — | -2.460 | 100.0% | -1.472* | 47.7% | 0.000 | 0.0% | — | — | 0.000 | 0.0% |
| 4 | -114.38 | 68.45 | 20/0/0 | 5.597 | 100.0% | — | — | -0.087 | 100.0% | -0.097* | 7.1% | 30.000 | 0.4% | — | — | — | — |
| 5 | -46.51 | 740.15 | 7/13/0 | 5.105 | 99.9% | — | — | -6.828 | 100.0% | -13.469* | 91.0% | 1195.000 | 1.6% | — | — | — | — |
| 6 | -111.18 | 68.45 | 20/0/0 | 5.603 | 100.0% | — | — | -0.020 | 100.0% | -0.067* | 4.9% | 0.000 | 0.0% | — | — | — | — |
| 7 | -98.93 | 69.60 | 20/0/0 | 5.553 | 100.0% | — | — | -0.221 | 100.0% | -0.178* | 12.8% | 5.000 | 0.1% | — | — | — | — |
| 8 | 26.45 | 206.50 | 20/0/0 | 2.667 | 100.0% | -1.577 | 99.9% | -0.263 | 100.0% | -1.232 | 59.6% | — | — | — | — | — | — |
| 9 | 145.17 | 718.80 | 9/11/0 | 2.638 | 99.2% | -2.022 | 99.2% | -0.748 | 100.0% | -4.268 | 59.4% | 5330.000 | 7.4% | — | — | — | — |
| **10** | **131.35** | **831.25** | **11/9/0** | **24.704** | **100.0%** | **-1.529** | **99.9%** | **-0.564** | **100.0%** | **-7.934** | **95.4%** | **6100.000** | **7.3%** | **—** | **—** | **—** | **—** |

> `term` 列格式：`terminated/truncated/early_terminal`。
> `*` 标记：iter_01 中的 `angle_penalty` 和 `action_cost` 在概念上对应后续的 `angle_stab` 和 `fuel_penalty`，但计算方式不同，仅供参考。
> `action_cost` 在 iter_01—07 中作为独立组件存在，iter_08 起改为 `fuel_penalty`（语义上等价）。

---

## 3. 数字标记

### 主导
- **success_landing_bonus**：magnitude_share=99.3%，为绝对主导组件（ep_sum=6100.00）

### 死组件
- 无。最低 active_rate 为 success_landing_bonus (7.3%)，但这是稀疏事件奖励的预期行为，实际非零。

### 淹没
- **angle_stab**：active_rate=100.0%，magnitude_share≈0.0%（ep_sum_mean=-0.564，被 success_landing_bonus 的 6100.00 完全淹没）
- **vel_damping**：active_rate=99.9%，magnitude_share≈0.0%（ep_sum_mean=-1.529，同上）
- **progress**：active_rate=100.0%，magnitude_share=0.5%（ep_sum_mean=24.704 vs 6100.00）

### 全覆盖惩罚
- 无。所有惩罚组件（vel_damping、angle_stab、fuel_penalty）的 signed_share 均为负但 magnitude_share 极低（0.0%—0.1%），不满足 active_rate>95% 的正 magnitude_share 条件。

### 量级对比（每步均值 = ep_sum ÷ len）
- 最大：
  1. success_landing_bonus: 6100.00 / 831.25 ≈ 7.338
  2. progress: 24.704 / 831.25 ≈ 0.030
- 最小：
  1. angle_stab: -0.564 / 831.25 ≈ -0.00068
  2. vel_damping: -1.529 / 831.25 ≈ -0.00184

所有 shaping/constraint 组件（progress、vel_damping、angle_stab、fuel_penalty）的每步均值均在 0.03 以下，而 success_landing_bonus 约为 7.34，相差约 245 倍。

### 修改变化（iter_09 → iter_10）
| component | Δ ep_sum | Δ active_rate |
|-----------|----------|---------------|
| progress | +22.066 | +0.8 pp |
| vel_damping | +0.492 | +0.7 pp |
| angle_stab | +0.184 | 0.0 pp |
| fuel_penalty | -3.666 | +36.0 pp |
| success_landing_bonus | +770.0 | -0.1 pp |

唯一显著变化：progress 的 ep_sum 从 2.638 升至 24.704（约 9.4 倍），对应 w_progress 从 2.0 提升至 20.0。fuel_penalty 的 active_rate 从 59.4% 跳升至 95.4%，ep_sum 翻倍。success_landing_bonus ep_sum 增加 770（≈14%）。

---

## 4. 语义演化（跨轮组件功能追踪）

### 着陆/接触信号（success landing bonus / landing success）
| iter | 组件名 | ep_sum_mean | active_rate | 阈值 (dist/speed/tilt) | 触发方式 |
|------|--------|-------------|-------------|---------------------|---------|
| 2 | success_landing_bonus | 5.000 | 0.1% | 0.1/0.2/0.1 | any_contact 持续判断 |
| 3 | success_landing_bonus | 0.000 | 0.0% | 0.1/0.2/0.1 | any_contact 持续判断 |
| 4 | success_landing_bonus | 30.000 | 0.4% | 0.2/0.5/0.2 | any_contact 持续判断 |
| 5 | success_landing_bonus | 1195.000 | 1.6% | 0.4/1.0/0.5 | any_contact 持续判断 |
| 6 | success_landing_bonus | 0.000 | 0.0% | 0.3/0.3/0.3 | any_contact 转变事件 |
| 7 | success_landing_bonus | 5.000 | 0.1% | 0.4/1.0/0.5 | any_contact 转变事件 |
| 9 | success_landing_bonus | 5330.000 | 7.4% | 0.4/1.0/0.5 | both_contact 转变事件 |
| 10 | success_landing_bonus | 6100.000 | 7.3% | 0.4/1.0/0.5 | both_contact 转变事件 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆信号 | v2_success_landing_bonus → v3_success_landing_bonus → v4_success_landing_bonus → v5_success_landing_bonus → v6_success_landing_bonus → v7_success_landing_bonus → v9_success_landing_bonus → v10_success_landing_bonus | -120 → -203 → -114 → -47 → -111 → -99 → 145 → 131 | 持续性→转变事件：1次切换（v6）；阈值收紧↔放宽：4次切换（v4放宽→v6收紧→v7放宽→v9不变）；接触条件：any→both：1次切换（v9） |

---

### 接近/进度信号（progress/progress_shaping）
| iter | 组件名 | ep_sum_mean | active_rate | 权重 | 数学形式 |
|------|--------|-------------|-------------|------|---------|
| 1 | progress_shaping | 1.582 | 100.0% | 5.0 | 势能差：5×((-next_dist)-(-dist)) |
| 2 | progress_shaping | 5.583 | 100.0% | 5.0 | 同上 |
| 3 | progress_shaping | 1.488 | 100.0% | 5.0 | 同上 |
| 4 | progress_shaping | 5.597 | 100.0% | 5.0 | 同上 |
| 5 | progress_shaping | 5.105 | 99.9% | 5.0 | 同上 |
| 6 | progress_shaping | 5.603 | 100.0% | 5.0 | 同上 |
| 7 | progress_shaping | 5.553 | 100.0% | 5.0 | 同上 |
| 8 | progress | 2.667 | 100.0% | 2.0 | 距离减少量：dist-next_dist |
| 9 | progress | 2.638 | 99.2% | 2.0 | 同上 |
| 10 | progress | 24.704 | 100.0% | 20.0 | 同上 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近信号 | v1_progress_shaping → v2_progress_shaping → ... → v7_progress_shaping → v8_progress → v9_progress → v10_progress | -156 → -120 → -203 → -114 → -47 → -111 → -99 → 26 → 145 → 131 | 势能差→距离减少量：1次切换（v8）；权重调整：5.0→2.0→20.0，2次切换 |

---

### 着陆速度约束
| iter | 组件名 | ep_sum_mean | active_rate | 形式 |
|------|--------|-------------|-------------|------|
| 1 | landing_velocity_penalty | -0.067 | 5.3% | 门控 hinge（y<0.5） |
| 2—10 | — | — | — | 移除，未恢复 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆速度约束 | v1_landing_velocity_penalty → (删除) | -156 | 仅存1轮后删除 |

---

### 稳定性/约束信号（angle_stab/angle_penalty + vel_damping）
| iter | 组件名 | ep_sum_mean | active_rate | 权重 | 形式 |
|------|--------|-------------|-------------|------|------|
| 1 | angle_penalty | -1.141 | 100.0% | -0.5 | body_angle² |
| 2—7 | angle_penalty | -0.087~-6.828 | 100.0% | -0.5 | body_angle² |
| 8—10 | angle_stab | -0.263~-0.748 | 100.0% | -0.05 | angle²+angvel²（next_obs） |
| 8—10 | vel_damping | -1.529~-2.022 | 99.2%—99.9% | -0.05 | vx²+vy²（next_obs） |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 稳定性 | v1_angle_penalty → v8_angle_stab+vel_damping | -156 → 26 | obs角度→next_obs全状态：1次切换（v8）；单组件→双组件：1次切换（v8） |

---

### 效率/惩罚信号（fuel_penalty/action_cost）
| iter | 组件名 | ep_sum_mean | active_rate | 权重 |
|------|--------|-------------|-------------|------|
| 1—7 | action_cost | -0.067~-13.469 | 4.9%—91.0% | -0.02 |
| 8—10 | fuel_penalty | -1.232~-7.934 | 59.4%—95.4% | -0.01 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 燃料效率 | v1_action_cost → v8_fuel_penalty | -156 → 26 | 权重-0.02→-0.01：1次切换 |

---

### 坠毁惩罚
| iter | 组件名 | ep_sum_mean | active_rate |
|------|--------|-------------|-------------|
| 3 | crash_penalty | 0.000 | 0.0% |
| 4—10 | — | — | 删除 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 失败惩罚 | v3_crash_penalty → (删除) | -203 | 仅存1轮，死组件即删除 |

---

**方向切换汇总**：
- 着陆信号：持续↔转变 1次，阈值收紧↔放宽 4次，接触条件 1次
- 接近信号：势能差→距离减少量 1次，权重 2次
- 稳定性：obs→next_obs 1次，单组件→双组件 1次
- 燃料效率：权重调整 1次

≥2次切换的功能：着陆信号（6次方向切换，其中阈值收紧↔放宽 4次），接近信号（3次）。

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