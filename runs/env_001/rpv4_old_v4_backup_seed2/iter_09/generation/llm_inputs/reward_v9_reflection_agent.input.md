第 8 轮迭代训练已经结束，得分 199.416737（目标: 200）。

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

实验记录位于 `rpv4_old_v4/seed_2/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_8, score=199.416737)
# Training Feedback

## Final-policy outcome
score=199.416737, len=477.650000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-41.000198, 281.710008]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 70.982707 | 79.7% | 79.7% | 32.3% |
| distance_progress | 6.303575 | 7.1% | 7.6% | 98.5% |
| descent_bonus | 6.376723 | 7.2% | 7.2% | 69.7% |
| fuel_penalty | -3.050500 | -3.4% | 3.4% | 63.9% |
| stability | -1.870730 | -2.1% | 2.1% | 100.0% |

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
| -81.4 | -76.0 | 45.4 | 31.4 | 7.1 | 5.1 | 89.2 | 189.7 | 150.1 | 228.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 5.286 | 5.249 | 5.297 | 5.612 | 6.246 | 6.374 | 6.435 | 6.090 | 5.925 | 6.367 |
| distance_progress | 5.541 | 5.466 | 5.465 | 5.919 | 6.198 | 6.257 | 6.456 | 6.157 | 5.806 | 6.465 |
| fuel_penalty | -0.147 | -0.168 | -0.519 | -0.739 | -0.791 | -0.827 | -1.339 | -2.339 | -3.939 | -2.652 |
| landing_completion | 0.000 | 0.000 | 2.710 | 1.447 | 0.045 | 0.020 | 6.328 | 36.705 | 121.517 | 79.608 |
| stability | -2.801 | -2.716 | -2.643 | -2.212 | -2.238 | -2.203 | -2.267 | -2.654 | -3.538 | -2.215 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| descent_bonus | 93.2% | 93.4% | 81.8% | 91.9% | 96.5% | 96.6% | 90.8% | 84.9% | 60.0% | 72.1% |
| distance_progress | 100.0% | 100.0% | 97.3% | 99.4% | 100.0% | 100.0% | 98.4% | 97.8% | 99.5% | 97.8% |
| fuel_penalty | 21.0% | 23.4% | 45.3% | 59.2% | 61.4% | 59.9% | 59.7% | 55.8% | 59.5% | 57.2% |
| landing_completion | 0.0% | 0.0% | 6.1% | 3.6% | 0.7% | 0.5% | 8.1% | 25.7% | 43.1% | 39.4% |
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

# Subagent 数据报告
# 训练数据报告

## 1. 行为事实

- **得分**：score=199.42，range=[-41.00, 281.71]
- **终止方式**：terminated=17/20（成功着陆/接触终止），truncated=3/20（时间耗尽）
- **回合长度**：len=477.65
- **Checkpoint 轨迹**：
  - cp10%=-81.4 → cp30%=45.4 → cp50%=7.1 → cp70%=89.2 → cp80%=189.7 → cp90%=150.1 → cp100%=228.0
  - 总体趋势：前期上升，cp50%小幅回落，cp70%后快速攀升，cp80%-cp90%间有波动(cp80%=189.7→cp90%=150.1→cp100%=228.0)
- **组件层面 checkpoint 关键观察**：
  - `landing_completion`：cp10%-cp60%基本为0，cp70%=6.33，cp80%=36.71，cp90%=121.52，cp100%=79.61（先升后降）
  - `fuel_penalty`：从cp10%=-0.15逐步恶化至cp90%=-3.94，cp100%回升至-2.65
  - `descent_bonus`：全程稳定在5.2-6.5之间
  - `stability`：全程在-2.2到-3.5之间波动，cp90%恶化至-3.54

## 2. 骨架演进表

注：iter 1-5 使用完全不同的组件骨架（progress/speed_penalty/angular_vel_penalty/landing/velocity_guidance），与 iter 6-8 不可比。仅列出 iter 6-8 的演进。

| iter | score | len | term | descent_bonus_ep_sum | descent_bonus_active | distance_progress_ep_sum | distance_progress_active | fuel_penalty_ep_sum | fuel_penalty_active | landing_completion_ep_sum | landing_completion_active | stability_ep_sum | stability_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | -13.57 | 1000.00 | 0/20, 20/20 | — | — | 6.33 | 100.0% | -9.89 | 98.9% | 0.00 | 0.0% | -1.42 | 100.0% |
| 7 | 146.32 | 673.35 | 14/20, 6/20 | 3.28 | 62.8% | 5.73 | 99.5% | -4.57 | 67.9% | 47.04 | 19.0% | -3.28 | 100.0% |
| 8 | 199.42 | 477.65 | 17/20, 3/20 | 6.38 | 69.7% | 6.30 | 98.5% | -3.05 | 63.9% | 70.98 | 32.3% | -1.87 | 100.0% |

iter 6 无 `descent_bonus` 组件，用 `—` 表示。

## 3. 数字标记

### 主导
- **无组件 magnitude_share > 80%**：`landing_completion` 的 magnitude_share 为 79.7%，未超过 80% 阈值。

### 死组件
- **无组件 active_rate < 2%**：所有组件活跃率均 ≥ 32.3%。

### 淹没
- **无组件同时满足 active_rate > 90% 且 magnitude_share < 3%**。

### 全覆盖惩罚
- **`stability`**：active_rate=100.0%，signed_share=-2.1%（负值），即该组件在所有 episode 的所有步都被激活，且始终贡献负奖励。但 magnitude_share 仅为 2.1%，非全覆盖惩罚（全覆盖惩罚需 active_rate > 95% + signed_share 为负；此处 active_rate=100% 但 magnitude_share 未达到显著性阈值，仅标记为全覆盖负信号而非"淹没"级惩罚）。

### 量级对比（每步均值 = ep_sum ÷ len）

| 组件 | ep_sum | len | 每步均值 |
|------|--------|-----|---------|
| landing_completion | 70.98 | 477.65 | 0.1486 |
| descent_bonus | 6.38 | 477.65 | 0.0133 |
| distance_progress | 6.30 | 477.65 | 0.0132 |
| fuel_penalty | -3.05 | 477.65 | -0.0064 |
| stability | -1.87 | 477.65 | -0.0039 |

- **最大两个**：`landing_completion`(0.1486)、`descent_bonus`(0.0133)
- **最小两个**：`fuel_penalty`(-0.0064)、`stability`(-0.0039)

### 修改变化（iter 7 → iter 8 Δ）

| 组件 | Δ ep_sum | Δ active_rate |
|------|----------|---------------|
| landing_completion | +23.94 | +13.3pp |
| descent_bonus | +3.10 | +6.9pp |
| distance_progress | +0.57 | -1.0pp |
| fuel_penalty | +1.52 | -4.0pp |
| stability | +1.41 | 0.0pp |

## 4. 语义演化（跨轮组件功能追踪）

按语义功能分组（基于 iter 6-8，iter 1-5 使用不同骨架，仅在功能层面可比较的位置加入）：

### 功能角色识别

- **接近/进度信号**：`distance_progress`（距离减少奖励×5.0）、`descent_bonus`（垂直下降奖励×6.0，x门控）
- **着陆/接触信号**：`landing_completion`（双脚接触+近中心+低速低角度的乘积bonus）
- **稳定性/约束信号**：`stability`（速度/角度/角速度平方惩罚×距离门控）、`fuel_penalty`（引擎使用固定惩罚-0.01）

### 语义演化表

| 语义功能 | 各轮组件名 | 各轮 ep_sum | 方向切换 |
|---------|----------|-----------|---------|
| 接近/进度信号 | iter6: distance_progress(6.33) → iter7: distance_progress(5.73)+descent_bonus(3.28) → iter8: distance_progress(6.30)+descent_bonus(6.38) | iter6: 6.33 → iter7: 9.01 → iter8: 12.68 | 无切换（descent_bonus 从 iter7 加入后持续存在，均为连续奖励+改善量形式） |
| 着陆/接触信号 | iter6: landing_completion(0.00, 死) → iter7: landing_completion(47.04) → iter8: landing_completion(70.98) | iter6: 0.00 → iter7: 47.04 → iter8: 70.98 | 稀疏bonus（iter6-8均为条件激活的乘积bonus，无切换） |
| 稳定性/约束信号 | iter6: stability(-1.42)+fuel_penalty(-9.89) → iter7: stability(-3.28)+fuel_penalty(-4.57) → iter8: stability(-1.87)+fuel_penalty(-3.05) | iter6: -11.31 → iter7: -7.85 → iter8: -4.92 | stability：门控惩罚（iter6-8均为exp(-2d)门控，无切换）；fuel_penalty：固定惩罚（iter6-8均为-0.01 per action，无切换） |

### 方向切换统计
- **着陆/接触信号**：0 次切换（iter6-8 均使用条件稀疏bonus）
- **接近/进度信号**：0 次切换（descent_bonus 于 iter7 新增，未发生替代/对立设计切换）
- **稳定性/约束信号**：0 次切换（stability 始终为门控惩罚，fuel_penalty 始终为固定惩罚）

**跨 iter 1-8 全局视角**（补充 iter 1-5 的功能轨迹，基于代码形态）：

| 语义功能 | 各轮组件名 | 得分趋势 | 方向切换 |
|---------|----------|---------|---------|
| 着陆/接触信号 | iter1-3: landing（稀疏contact门控或proximity门控）→ iter4: 删除，velocity_guidance 替代 → iter5: landing（proximity+contact放大）→ iter6-8: landing_completion（稀疏双脚contact+质量乘积） | iter1: 死(0.00) → iter2: 697.44(主导) → iter3: 死(0.00) → iter4: 无 → iter5: 死(0.00) → iter6: 死(0.00) → iter7: 47.04 → iter8: 70.98 | 稀疏bonus↔连续奖励：iter2为连续proximity+sum-of-factors，iter3/5/6/7/8为稀疏contact门控→**至少2次切换**（iter2连续→iter3稀疏→iter5稀疏→iter6-8稀疏）；proximity门控↔contact门控：iter2/5为proximity门控，iter3/6-8为contact门控→**1次切换** |
| 接近/进度信号 | iter1-5: progress（Δdistance×1.0）→ iter6-8: distance_progress（Δdistance×5.0）+ descent_bonus（Δy×系数×x_gate，iter7新增） | iter1-5: progress 逐步萎缩（-0.20→0.66→-0.30→-0.03→-0.18）→ iter6-8: distance_progress 稳定在5.7-6.5 | 缩放系数：1.0→5.0（1次切换）；纯距离→距离+垂直下降分解（iter7新增descent_bonus，1次新增，非对立切换） |
| 稳定性/约束信号 | iter1-5: speed_penalty（closeness×speed）、angular_vel_penalty（绝对值）→ iter6-8: stability（平方惩罚×exp门控）、fuel_penalty（固定值） | iter1-5: 惩罚持续存在→ iter6-8: 惩罚总量改善（-11.31→-4.92） | 线性和→平方惩罚（1次切换）；绝对值→平方（1次切换）；新增fuel_penalty（iter6，1次新增） |

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