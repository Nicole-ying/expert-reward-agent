第 7 轮迭代训练已经结束，得分 227.643558（目标: 200）。

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
- **接近目标时优先微调**：如果当前得分或 best 得分已经接近目标（差距 < 10%），**强烈推荐 L1 微调**（只调系数）。历史尝试中反复出现同一模式：分数已接近目标，却选择 L2 改组件形态，结果越改越差，连续多轮倒退。架构已有基本有效性时，大规模改动风险远大于收益。

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v4/seed_3/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_7, score=227.643558)
# Training Feedback

## Final-policy outcome
score=227.643558, len=374.600000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-11.906847, 281.988416]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 36.599581 | 55.8% | 55.8% | 20.3% |
| descent_incentive | 11.552093 | 17.6% | 17.6% | 87.9% |
| goal_proximity | 7.917188 | 12.1% | 12.5% | 96.4% |
| velocity_penalty | -6.845214 | -10.4% | 10.4% | 97.0% |
| orientation_penalty | -2.401008 | -3.7% | 3.7% | 100.0% |

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
| -32.2 | -150.0 | -4.4 | -2.7 | -9.8 | 246.1 | 157.2 | 223.1 | 248.3 | 236.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 61.025 | 412.079 | 64.461 | 51.543 | 52.380 |
| descent_incentive | 3.736 | 0.154 | 7.226 | 12.638 | 5.816 | 12.711 | 13.873 | 12.321 | 12.771 | 11.499 |
| goal_proximity | 7.432 | -0.781 | 7.855 | 7.945 | 7.894 | 7.950 | 7.939 | 7.938 | 7.948 | 7.923 |
| orientation_penalty | -5.410 | -4.971 | -3.877 | -4.144 | -3.630 | -2.529 | -2.940 | -2.522 | -2.088 | -2.614 |
| velocity_penalty | -9.295 | -2.438 | -3.948 | -4.226 | -3.657 | -6.069 | -5.620 | -6.567 | -6.694 | -7.059 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 9.7% | 33.3% | 20.3% | 9.4% | 27.0% |
| descent_incentive | 64.7% | 38.8% | 86.4% | 87.6% | 93.6% | 90.5% | 71.4% | 81.9% | 90.1% | 85.7% |
| goal_proximity | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 96.3% | 98.9% | 97.5% | 96.0% | 96.7% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 96.8% | 99.0% | 97.8% | 96.6% | 97.1% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7 reward: L2 — potential-based descent_incentive replaces linear-gate formulation.

    Problem (v6): descent_incentive uses `near_ground * descending` where both terms
    are linear. As y→0, the possible descent per step shrinks (physics), causing the
    gradient to weaken at the most critical moment — the final few cm before touchdown.
    The agent hovers near the pad but cannot bridge the last gap to landing.

    Change:
      - descent_incentive: replaced linear `(1-y/1.5) * (y_cur-y_nxt)` with
        potential-based shaping. Potential φ(y) = 1/(y+0.1). Reward = φ(y_nxt) - φ(y_cur).
        This creates a gradient that naturally INTENSIFIES as y→0, compensating for
        the shrinking descent step size.
      - contact_bonus: weight reduced 4.0→2.0 (supporting change). When contact_bonus
        fires at cp80% (1169 ep_sum), it creates a destabilizing reward spike that
        causes catastrophic forgetting of landing behavior. Lower weight reduces the
        spike while keeping it as a meaningful confirmation signal.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (L2 — potential-based shaping) ---
    # Potential φ(y) = 1/(y+eps): increases as y decreases, with derivative
    # dφ/dy = -1/(y+eps)^2 that grows strongly near y=0. This creates a
    # natural funnel: the same descent step yields MORE reward when closer
    # to the ground, exactly where the linear formulation used to weaken.
    #
    # Gated by near_center so agent only gets descent credit when aligned
    # with the landing pad.
    w_desc = 1.5
    eps = 0.1
    potential_cur = 1.0 / (y_cur + eps)
    potential_nxt = 1.0 / (y_nxt + eps)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descent_incentive = w_desc * near_center * max(0.0, potential_nxt - potential_cur)

    # --- 5. contact_bonus (sparse landing confirmation, reduced weight) ---
    # Weight reduced 4.0→2.0 to prevent the reward spike (observed up to 1169
    # ep_sum at cp80%) from destabilizing training and causing catastrophic
    # forgetting of landing behavior.
    w_contact = 2.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# Subagent 数据报告
# 训练数据报告

## 1. 行为事实

- **得分**：score=227.64，range=[-11.91, 281.99]
- **终止方式**：terminated=19/20（95% 成功着陆），truncated=1/20（5% 时间上限）
- **回合长度**：len=374.60（成功率提升，回合提前终止）
- **Checkpoint 官方得分轨迹**：
  cp10%: -32.2 → cp20%: -150.0 → cp30%: -4.4 → cp40%: -2.7 → cp50%: -9.8 → cp60%: 246.1 → cp70%: 157.2 → cp80%: 223.1 → cp90%: 248.3 → cp100%: 236.3
  - 训练在 cp60% 处分数骤升（从 -9.8 到 246.1）
  - cp70% 有所回落（157.2），随后恢复并稳定在 220-250 区间
  - 最终评估得分 227.64，与 cp100% 的 236.3 接近

---

## 2. 骨架演进表

| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | descent_incentive_ep_sum | descent_incentive_active | goal_proximity_ep_sum | goal_proximity_active | orientation_penalty_ep_sum | orientation_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -32.35 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | — | — | 3.758 | 100.0% | -3.959 | 100.0% | -0.486 | 100.0% |
| 2 | 107.13 | 775.60 | 11T/9Tr | 154.465 | 2.8% | — | — | 3.963 | 99.1% | -3.090 | 100.0% | -5.313 | 99.2% |
| 3 | -26.17 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | — | — | 7.685 | 100.0% | -4.038 | 100.0% | -4.758 | 100.0% |
| 4 | 183.52 | 710.25 | 19T/1Tr | 114.083 | 4.9% | 0.779 | 72.4% | 7.951 | 98.1% | -3.386 | 100.0% | -5.736 | 98.4% |
| 5 | -15.01 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | 0.624 | 66.1% | 7.881 | 100.0% | -3.615 | 100.0% | -6.826 | 100.0% |
| 6 | -15.73 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | 3.366 | 73.8% | 7.861 | 100.0% | -3.917 | 100.0% | -5.725 | 100.0% |
| 7 | 227.64 | 374.60 | 19T/1Tr | 36.600 | 20.3% | 11.552 | 87.9% | 7.917 | 96.4% | -2.401 | 100.0% | -6.845 | 97.0% |

---

## 3. 数字标记

### 主导
- **contact_bonus** magnitude_share=55.8%，超过 80% 阈值？**否**（55.8% < 80%），无组件构成主导

### 死组件
- 无组件 active_rate < 2%

### 淹没
- **orientation_penalty**：active_rate=100.0%，magnitude_share=3.7%（< 3%？否，3.7% ≥ 3%，不满足淹没条件，但极低）
- **无其他组件**同时满足 active_rate > 90% 且 magnitude_share < 3%

### 全覆盖惩罚
- **orientation_penalty**：active_rate=100.0%（> 95%），signed_share=-3.7%（为负）→ **全覆盖惩罚**：该组件在几乎每一步都被激活且始终产生负奖励

### 量级对比（每步均值 = ep_sum ÷ len）

| 组件 | ep_sum_mean | len | 每步均值 |
|---|---|---|---|
| contact_bonus | 36.600 | 374.60 | **0.0977** ← 最大正向 |
| descent_incentive | 11.552 | 374.60 | 0.0308 |
| goal_proximity | 7.917 | 374.60 | 0.0211 |
| orientation_penalty | -2.401 | 374.60 | **-0.00641** ← 最小（绝对值） |
| velocity_penalty | -6.845 | 374.60 | -0.0183 |

- 每步均值最大两个正向组件：**contact_bonus (0.0977)**、**descent_incentive (0.0308)**
- 每步均值最小两个（绝对值最小）：**orientation_penalty (-0.00641)**、**velocity_penalty (-0.0183)**

### 修改变化（iter 7 vs iter 6，Δ = iter7 − iter6）

注意：iter 6 的 len 为 1000，iter 7 的 len 为 374.60，ep_sum 变化受回合长度缩短影响。

| 组件 | Δ ep_sum | Δ active_rate |
|---|---|---|
| contact_bonus | +36.600 | +20.3%（从 0% → 20.3%） |
| descent_incentive | +8.186 | +14.1%（从 73.8% → 87.9%） |
| goal_proximity | +0.056 | -3.6%（从 100% → 96.4%） |
| orientation_penalty | +1.516（绝对值减小） | 0%（维持 100%） |
| velocity_penalty | -1.120（绝对值增大） | -3.0%（从 100% → 97.0%） |

---

## 4. 语义演化（跨轮组件功能追踪）

### 语义功能分组

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| **着陆/接触信号** | iter1: contact_bonus(0.0,死) → iter2: contact_bonus(154.5,主导) → iter3: contact_bonus(0.0,死) → iter4: contact_bonus(114.1,主导) → iter5: contact_bonus(0.0,死) → iter6: contact_bonus(0.0,死) → iter7: contact_bonus(36.6,主导) | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 → 227.6 | 稀疏bonus↔死组件：**6次切换**（iter1死→iter2活→iter3死→iter4活→iter5死→iter6死→iter7活） |
| **接近/进度信号** | iter1-3: goal_proximity → iter4-7: goal_proximity + descent_incentive | 得分见上 | 门控线性→潜力式：**1次切换**（iter6 desc为线性门控，iter7 desc改为潜力式 shaping） |
| **稳定性/约束信号** | iter1-7: velocity_penalty + orientation_penalty | 得分见上 | 门控形式：iter1 Manhattan距离门控 → iter2-7 指数高度门控，**1次切换** |

### 方向切换详情

- **着陆/接触信号**：在"稀疏bonus"和"死组件"之间交替了 6 次。该功能在成功轮（iter2、iter4、iter7）中表现为主导奖励，在失败轮（iter1、iter3、iter5、iter6）中完全死亡。无"连续奖励"替代方案——该功能始终使用稀疏条件性 bonus 数学形态。

- **接近/进度信号**：iter4-6 使用线性门控 descent_incentive（`near_ground * descending`），iter7 切换为潜力式 shaping（`φ(y_nxt)-φ(y_cur)`）。在 iter1-3 中无 descent_incentive 组件，仅由 goal_proximity 承担接近引导。**切换次数：1 次**（线性→潜力式，发生在 iter6→iter7）。

- **稳定性/约束信号**：velocity_penalty 从 iter1 的 Manhattan 距离门控（`1/(1+dist)`）切换为 iter2 的指数高度门控（`exp(-y/h)`），此后维持不变。orientation_penalty 自 iter1 起未改变数学形态。**切换次数：1 次**（迭代速度门控形式，发生在 iter1→iter2）。

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推测为 float32  
- obs[0]: x_position，水平坐标（相对于目标平台的 x 偏移），reward_usable: true  
- obs[1]: y_position，垂直坐标（相对于平台高度，即飞行器底部到平台表面的距离），reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，身体倾斜角度（弧度 or 归一化角度），reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  

备注：所有维度均为连续数字，接触标志虽为 0/1 但以浮点形式给出。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine（不点火，靠惯性/重力运动）  
- action 1: left_orientation_engine（点火左向姿态调节引擎，产生向左旋转力矩）  
- action 2: main_engine（点火主引擎，产生向上的推力，通常用于减速/悬停）  
- action 3: right_orientation_engine（点火右向姿态调节引擎，产生向右旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能代表身体已静止，结合 y_position≈0 且左右支撑接触为 1 时可视为成功着陆。但 termination 本身是模糊的，因为 crash 后也可能导致静止。  
- failure-like termination: crash_or_body_contact 中“crash”部分，horizontal_position_outside_viewport（水平超出视口）  
- ambiguous termination: body_not_awake_or_settled 的具体触发条件未知，可能是身体进入休眠/静止状态；crash_or_body_contact 中的 body_contact 可能包含正常着陆接触，需依赖位置判断。  
- truncation: 未提供  

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）  
- explicit_failure_flag_available: false（同样无 failure 字段）  
- allowed_info_fields: 无（info 返回空字典 {}）  
- forbidden_or_uncertain_info_fields: 所有 internal 字段均不可用，如 termination_reason 等

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action 值（0~3），可用于燃料惩罚或推力鼓励
- other: 通过 next_obs 计算上述量的变化率（如速度变化、位置变化）