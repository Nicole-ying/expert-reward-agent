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
第 2 轮迭代训练已经结束，得分 107.129410（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(2)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

# 上一轮训练反馈 (iter_2, score=107.129410)
# Training Feedback

## Final-policy outcome
score=107.129410, len=775.600000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-33.715767, 264.822182]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 154.464574 | 92.5% | 92.5% | 2.8% |
| velocity_penalty | -5.313230 | -3.2% | 3.2% | 99.2% |
| goal_proximity | 3.962801 | 2.4% | 2.5% | 99.1% |
| orientation_penalty | -3.090273 | -1.9% | 1.9% | 100.0% |

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
| -168.7 | -181.1 | -88.9 | -0.4 | -16.3 | 103.9 | -31.8 | -12.4 | -28.9 | 135.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.000 | 0.000 | 0.000 | 1.834 | 0.000 | 156.061 | 0.000 | 0.000 | 0.000 | 183.313 |
| goal_proximity | 2.359 | 0.723 | 2.486 | 3.947 | 3.900 | 3.948 | 3.811 | 3.936 | 3.839 | 3.968 |
| orientation_penalty | -2.558 | -2.601 | -3.682 | -3.696 | -4.393 | -3.744 | -4.092 | -3.925 | -3.655 | -3.152 |
| velocity_penalty | -6.511 | -3.912 | -2.429 | -3.520 | -4.931 | -4.406 | -4.869 | -3.928 | -4.024 | -5.862 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.0% | 0.0% | 0.0% | 0.3% | 0.0% | 4.8% | 0.0% | 0.0% | 0.0% | 3.8% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 98.9% | 100.0% | 100.0% | 100.0% | 98.7% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.1% | 100.0% | 100.0% | 100.0% | 98.8% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward: goal proximity + height-gated velocity damping + orientation penalty + safe contact bonus

    Key change from v1: velocity penalty now uses exponential height-based gate
    instead of Manhattan distance. Gate is ~1.0 near ground (y≈0) and decays
    rapidly with height, so the agent only needs to slow down when landing.
    Weight increased 10x (0.05 → 0.5) to match the magnitude of other components.
    
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

    # --- 1. goal_proximity (improvement_delta) ---
    # encourage moving toward (0,0) by reducing squared distance
    # larger weight on vertical axis to avoid hard landings
    w_x, w_y = 1.0, 2.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated, L2 change) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    # This lets the agent fly fast at high altitude but forces deceleration for landing.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus (joint condition proxy) ---
    # reward when both legs touch and position is near (0,0)
    w_contact = 10.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```

# Subagent 数据报告
# 训练数据报告（记者模式）

## 1. 行为事实
- **最终策略得分**：score = 107.13，range = [-33.72, 264.82]
- **终止方式**：terminated（成功着陆）11/20，truncated（时间上限）9/20
- **回合长度**：len = 775.60
- **Checkpoint 得分轨迹**（官方环境得分，cp1 = 10% 步数，cp10 = 100% 步数）：

  | cp1 | cp2 | cp3 | cp4 | cp5 | cp6 | cp7 | cp8 | cp9 | cp10 |
  |------|------|------|------|------|------|------|------|------|------|
  | -168.7 | -181.1 | -88.9 | -0.4 | -16.3 | 103.9 | -31.8 | -12.4 | -28.9 | 135.0 |
  
  轨迹整体上升，但 cp6 出现峰值后回落，再回升至最终 135.0。

## 2. 骨架演进表
| iter | score | len | term (term/trunc) | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | orientation_penalty_ep_sum | orientation_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|------|-------|-----|-------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------------|----------------------------|-------------------------|-------------------------|
| 1 | -32.35 | 1000 | 0/20, 20/20 | 0.00 | 0.0% | 3.76 | 100.0% | -3.96 | 100.0% | -0.49 | 100.0% |
| 2 | 107.13 | 775.6 | 11/20, 9/20 | 154.46 | 2.8% | 3.96 | 99.1% | -3.09 | 100.0% | -5.31 | 99.2% |

## 3. 数字标记
- **主导**：contact_bonus 的 magnitude_share = 92.5%（>80%），主导最终回报。
- **死组件**：无（所有组件 active_rate ≥ 2.8%）。
- **淹没**：
  - goal_proximity：active_rate = 99.1%，magnitude_share = 2.5%（<3%）。
  - orientation_penalty：active_rate = 100.0%，magnitude_share = 1.9%（<3%）。
- **全覆盖惩罚**（active_rate > 95% 且 signed_share 为负）：
  - orientation_penalty：active 100%，signed_share = -1.9%。
  - velocity_penalty：active 99.2%，signed_share = -3.2%。
- **量级对比**（每步均值绝对值，ep_sum ÷ len）：
  - 最大两个：contact_bonus（0.199）、velocity_penalty（0.00685）。
  - 最小两个：goal_proximity（0.0051）、orientation_penalty（0.00398）。
- **修改变化（iter 1 → iter 2）**：
  - contact_bonus：ep_sum 0.00 → 154.46（+154.46），active_rate 0.0% → 2.8%（+2.8pp）。
  - goal_proximity：ep_sum 3.76 → 3.96（+0.20），active_rate 100.0% → 99.1%（-0.9pp）。
  - orientation_penalty：ep_sum -3.96 → -3.09（+0.87），active_rate 不变 100.0%。
  - velocity_penalty：ep_sum -0.49 → -5.31（-4.82），active_rate 100.0% → 99.2%（-0.8pp）。

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分（ep_sum_mean） | 方向切换 |
|----------|-----------|------------------------|----------|
| 接近/进度信号 | goal_proximity（iter1 → iter2） | 3.76 → 3.96 | 无（始终保持delta改善量） |
| 着陆/接触信号 | contact_bonus（iter1 → iter2） | 0.00 → 154.46 | 无（始终为稀疏bonus） |
| 稳定性-姿态 | orientation_penalty（iter1 → iter2） | -3.96 → -3.09 | 无（始终为连续惩罚） |
| 稳定性-速度 | velocity_penalty（iter1 → iter2） | -0.49 → -5.31 | 门控方式：距离反比门控 → 指数高度门控（1次切换） |

（注：velocity_penalty 的权重同时从 0.05 增加到 0.5，属于同一修改内的参数变化，不计为额外切换。）

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
```
