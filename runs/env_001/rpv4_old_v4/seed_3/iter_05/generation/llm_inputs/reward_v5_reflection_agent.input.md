第 4 轮迭代训练已经结束，得分 183.523455（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(4)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

# 上一轮训练反馈 (iter_4, score=183.523455)
# Training Feedback

## Final-policy outcome
score=183.523455, len=710.250000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[90.971361, 220.088340]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 114.082823 | 86.3% | 86.3% | 4.9% |
| goal_proximity | 7.951091 | 6.0% | 6.2% | 98.1% |
| velocity_penalty | -5.735901 | -4.3% | 4.3% | 98.4% |
| orientation_penalty | -3.386074 | -2.6% | 2.6% | 100.0% |
| descent_incentive | 0.779257 | 0.6% | 0.6% | 72.4% |

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
| -102.4 | -140.7 | -37.0 | -36.3 | -9.8 | -13.7 | -24.7 | -18.2 | 2.3 | 187.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 11.081 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 54.656 | 104.372 |
| descent_incentive | 0.137 | 0.010 | 0.220 | 0.255 | 0.576 | 0.652 | 0.413 | 0.489 | 0.810 | 0.762 |
| goal_proximity | 2.483 | -0.186 | 7.403 | 7.397 | 7.889 | 7.876 | 7.687 | 7.800 | 7.949 | 7.945 |
| orientation_penalty | -2.390 | -3.046 | -3.807 | -3.773 | -3.649 | -3.721 | -3.643 | -3.937 | -3.939 | -3.574 |
| velocity_penalty | -5.005 | -3.236 | -3.101 | -4.284 | -4.877 | -5.520 | -3.955 | -4.226 | -6.323 | -5.586 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.8% | 5.6% |
| descent_incentive | 32.6% | 13.7% | 62.5% | 61.5% | 64.9% | 63.4% | 65.6% | 67.3% | 66.3% | 74.3% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 97.9% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 98.2% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4 reward: added descent_incentive — continuous gradient for final descent phase

    Key change from v3: new `descent_incentive` component rewards downward progress
    when the agent is near the landing zone (y < 0.8, |x| < 0.5). This bridges the
    gap between "hovering near target" (where goal_proximity gradient vanishes at
    low y due to squared-distance form) and "committing to touchdown."
    
    Contact_bonus is kept at w=4.0 as a terminal landing confirmation, now reachable
    thanks to the continuous approach gradient provided by descent_incentive.
    
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
    # Encourage moving toward (0,0). Larger weight on vertical axis to avoid hard landings.
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (NEW — continuous descent gradient near landing zone) ---
    # Reward downward progress when near the landing pad.
    # Gates ensure this only fires when the agent is in position to land:
    #   near_ground: linearly decays from 1 at y=0 to 0 at y>=0.8
    #   near_center: linearly decays from 1 at x=0 to 0 at |x|>=0.5
    #   descending:  positive only when moving downward (y_cur > y_nxt)
    # This provides a continuous gradient that peaks during the final descent,
    # naturally tapering to zero once landed (y≈0, no further descent possible).
    w_desc = 2.0
    near_ground = max(0.0, 1.0 - y_nxt / 0.8)       # active below y=0.8
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descending = max(0.0, y_cur - y_nxt)             # positive when going down
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event bonus) ---
    # Reward when both legs touch and position is near (0,0).
    # Kept at moderate weight — descent_incentive now provides the approach gradient.
    w_contact = 4.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
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
# 训练数据报告 — iter_04 (v4)

## 1. 行为事实

- **得分**：score=183.52，range=[90.97, 220.09]
- **终止方式**：terminated=19/20（成功着陆），truncated=1/20（时间上限）
- **回合长度**：len=710.25
- **Checkpoint 轨迹**：官方得分从 cp1=-102.4 → cp3=-37.0 → cp9=2.3 → cp10=187.5（持续上升，cp9→cp10 跳跃 +185.2）
- **奖励错误**：0

## 2. 骨架演进表

| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | descent_incentive_ep_sum | descent_incentive_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -32.35 | 1000.00 | 0/0/20 | 0.000 | 0.0% | 3.758 | 100.0% | -0.486 | 100.0% | -3.959 | 100.0% | — | — |
| 2 | 107.13 | 775.60 | 11/9/0 | 154.465 | 2.8% | 3.963 | 99.1% | -5.313 | 99.2% | -3.090 | 100.0% | — | — |
| 3 | -26.17 | 1000.00 | 0/20/0 | 0.000 | 0.0% | 7.685 | 100.0% | -4.758 | 100.0% | -4.038 | 100.0% | — | — |
| 4 | 183.52 | 710.25 | 19/0/1 | 114.083 | 4.9% | 7.951 | 98.1% | -5.736 | 98.4% | -3.386 | 100.0% | 0.779 | 72.4% |

## 3. 数字标记

### 主导
- **contact_bonus**：magnitude_share=86.3%（>80%），主导总奖励量级

### 死组件
- 无（所有组件 active_rate ≥ 4.9%）

### 淹没
- 无（无组件同时满足 active_rate>90% 且 magnitude_share<3%）

### 全覆盖惩罚
- **orientation_penalty**：active_rate=100%（>95%）且 signed_share 为负（-2.6%）

### 量级对比（每步均值 = ep_sum ÷ len，从 iter_04 计算）
- **最大两个**：contact_bonus=0.1606，goal_proximity=0.0112
- **最小两个**：descent_incentive=0.0011，orientation_penalty=-0.0048

### 修改变化（iter_03 → iter_04 的 Δ）
| component | Δ ep_sum | Δ active_rate |
|---|---|---|
| contact_bonus | 0.000 → 114.083 (+114.083) | 0.0% → 4.9% (+4.9pp) |
| goal_proximity | 7.685 → 7.951 (+0.266) | 100.0% → 98.1% (-1.9pp) |
| velocity_penalty | -4.758 → -5.736 (-0.978) | 100.0% → 98.4% (-1.6pp) |
| orientation_penalty | -4.038 → -3.386 (+0.652) | 100.0% → 100.0% (0pp) |
| descent_incentive | 新增 | 新增 |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| **着陆/接触信号** | v1_contact_bonus(死) → v2_contact_bonus → v3_contact_bonus(死) → v4_contact_bonus | -32.35 → 107.13 → -26.17 → 183.52 | 稀疏bonus：v1失效→v2活跃→v3失效→v4活跃，**2次切换** |
| **接近/进度信号** | v1_goal_proximity → v2_goal_proximity → v3_goal_proximity → v4_goal_proximity + v4_descent_incentive | 同上 | 改善量(Δ)：v4新增descent_incentive为连续状态值形式，**1次新增** |
| **稳定性/约束信号-速度** | v1_velocity_penalty → v2_velocity_penalty → v3_velocity_penalty → v4_velocity_penalty | 同上 | 门控形式：v1曼哈顿距离门控→v2指数高度门控（沿用至v4），**1次切换** |
| **稳定性/约束信号-姿态** | v1_orientation_penalty → v2_orientation_penalty → v3_orientation_penalty → v4_orientation_penalty | 同上 | 无切换（四轮形式一致：|angle| + angvel²） |

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