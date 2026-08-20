# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：训练反馈出来后，自主调查、诊断问题、选择一个最小修改、输出新代码。

# 你可以自主查阅任何历史记录

以下工具随时可用，**应该在诊断过程中主动调用，不要等系统给你组装好**：

- `read_memory()` — 先读这个！一眼看完所有迭代：每轮的 skeleton、score、best、len、key_signal、action。快速掌握全貌。
- `get_component_history(component_name)` — 查看某个组件在所有迭代中的 ep_sum_mean 和 active_rate。如果值翻了 5 倍以上，它会警告你**只做 L1 系调**。
- `read_training_feedback(iteration)` — 读某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 读你某轮自己的反思记录（证据、诊断、干预选择）。动手改组件之前必须查：这个方向之前试过吗？结果如何？
- `read_reward_code(iteration)` — 读某轮的完整奖励函数代码。对比两个迭代的代码差异时使用。

# 工作流程

1. **读当前反馈** — 理解 score、len、termination、每个组件的 active_rate 和 per-step 量级。
2. **自主调查** — 根据你的判断，调用工具查历史。至少：
   - 查当前每个组件的 `get_component_history`，看有没有刚激活或刚崩塌的。
   - 如果某个组件值有异常跳变，调用 `read_past_reflection` 看当时做了什么。
3. **诊断** — 解释 agent 学到了什么行为，哪个组件最值得干预。
4. **选干预层级**：
   - **L1**：只调系数/权重，不改数学表达式。
   - **L2**：改一个组件的数学形态（稀疏→连续、状态值→改善量、乘积→加性、transition→sustained 等）。
   - **L3**：换主信号框架。触发条件：多轮迭代从未拿到可接受分数，且历史最佳迭代的骨架也无改进空间。
5. **动手前检查**：`get_component_history` 是否警告你"刚激活"？如果是 → 只做 L1。自传或 `read_past_reflection` 是否显示同样修改已失败过？如果是 → 换个方向。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。
- 除 L3 外，每轮只改一个组件。

# 输出

1. `evidence`：支持判断的关键数据和历史记录
2. `behavior_diagnosis`：策略当前的失败行为
3. `signal_completeness`：必要职责是否完备
4. `selected_level`：L1/L2/L3 及触发条件
5. `selected_intervention`：目标组件及具体修改内容
6. `falsifiable_hypothesis`：为什么该修改应改善策略
7. `expected_next_round`：下一轮哪些指标应如何变化
8. `main_risk`：最可能引入的新问题

然后输出完整 Python 代码。
```

## User Prompt

```markdown
第 10 轮迭代训练已结束，得分 145.168065（目标: 200）。

用 `read_memory()` 快速确认全局状态，用 `get_component_history` 逐组件检查是否有死组件（active<2%）或全覆盖惩罚（active>95% 且负贡献）。
找到问题组件后，用 `read_past_reflection` 确认你之前是否已经改过它、结果如何。

实验记录目录：
```
rpv4_old_v3/seed_2/
  iter_01/ ... iter_09/     ← 历史迭代
    generation/response_records/agent_reflection.md   ← 每轮反思决策
    generation/behavior_analyst_output_vN.md          ← Subagent 数据报告
    training/eval_result.json                          ← 训练评估结果
  memory/reward_memory.md                       ← 历史记忆表
  best/best_reward.py                            ← 历史最佳奖励函数
```

# 本轮训练反馈 (score=145.168065)
# Training Feedback

## Final-policy outcome
score=145.168065, len=718.800000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[33.531595, 288.928579]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 5330.000000 | 99.8% | 99.8% | 7.4% |
| fuel_penalty | -4.267500 | -0.1% | 0.1% | 59.4% |
| progress | 2.637606 | 0.0% | 0.1% | 99.2% |
| vel_damping | -2.021547 | -0.0% | 0.0% | 99.2% |
| angle_stab | -0.748251 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
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

# Subagent 数据报告
# 训练数据报告（iter=09）

## 1. 行为事实

- **得分**：score=145.168, range=[33.532, 288.929]
- **终止方式**：terminated=9/20（腿部触地终止），truncated=11/20（时间上限）
- **回合长度**：len=718.800
- **Checkpoint 轨迹**：官方得分 cp4=0.00 → cp8=0.00（持平）

## 2. 骨架演进表

| iter | score | len | term | action_cost/fuel_penalty_ep_sum | active | angle_penalty/angle_stab_ep_sum | active | landing_vel_penalty_ep_sum | active | progress_shaping/progress_ep_sum | active | crash_penalty_ep_sum | active | success_landing_bonus_ep_sum | active | vel_damping_ep_sum | active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -155.98 | 154.40 | 20/0 | -1.804 | 58.4% | -1.141 | 100.0% | -0.067 | 5.3% | 1.582 | 100.0% | — | — | — | — | — | — |
| 2 | -119.51 | 68.30 | 20/0 | -0.071 | 5.2% | -0.205 | 100.0% | — | — | 5.583 | 100.0% | — | — | 5.000 | 0.1% | — | — |
| 3 | -203.45 | 154.15 | 20/0 | -1.472 | 47.7% | -2.460 | 100.0% | — | — | 1.488 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% | — | — |
| 4 | -114.38 | 68.45 | 20/0 | -0.097 | 7.1% | -0.087 | 100.0% | — | — | 5.597 | 100.0% | — | — | 30.000 | 0.4% | — | — |
| 5 | -46.51 | 740.15 | 7/13 | -13.469 | 91.0% | -6.828 | 100.0% | — | — | 5.105 | 99.9% | — | — | 1195.000 | 1.6% | — | — |
| 6 | -111.18 | 68.45 | 20/0 | -0.067 | 4.9% | -0.020 | 100.0% | — | — | 5.603 | 100.0% | — | — | 0.000 | 0.0% | — | — |
| 7 | -98.93 | 69.60 | 20/0 | -0.178 | 12.8% | -0.221 | 100.0% | — | — | 5.553 | 100.0% | — | — | 5.000 | 0.1% | — | — |
| 8 | 26.45 | 206.50 | 20/0 | -1.232 | 59.6% | -0.263 | 100.0% | — | — | 2.667 | 100.0% | — | — | — | — | -1.577 | 99.9% |
| 9 | 145.17 | 718.80 | 9/11 | -4.268 | 59.4% | -0.748 | 100.0% | — | — | 2.638 | 99.2% | — | — | 5330.000 | 7.4% | -2.022 | 99.2% |

注：iter=8-9 使用新命名体系（fuel_penalty 对应旧 action_cost，angle_stab 对应旧 angle_penalty，progress 对应旧 progress_shaping，新增 vel_damping）。

## 3. 数字标记

- **主导**：success_landing_bonus magnitude_share=99.8% > 80%
- **死组件**：无（所有组件 active_rate ≥ 7.4%）
- **淹没**：无（无组件同时满足 active_rate > 90% 且 magnitude_share < 3%）
- **全覆盖惩罚**：无（无组件同时满足 active_rate > 95% 且 signed_share < 0）
- **量级对比**（每步均值 = ep_sum ÷ len，len=718.80）：
  - 最大：success_landing_bonus (7.415)，progress (0.00367)
  - 最小：fuel_penalty (-0.00594)，angle_stab (-0.00104)
- **修改变化**（iter=08 → iter=09）：
  | component | Δ ep_sum | Δ active_rate | Δ 每步均值 |
  |---|---|---|---|
  | progress | -0.029 | -0.8pp | -0.00128 |
  | vel_damping | -0.445 | -0.7pp | -0.00973 |
  | angle_stab | -0.485 | 0.0pp | -0.00149 |
  | fuel_penalty | -3.036 | -0.2pp | -0.00597 |
  | success_landing_bonus | +5330.0（新增） | +7.4pp | +7.415（新增） |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| 接近/进度信号 | v1_progress_shaping → v2-7_progress_shaping → v8-9_progress | -156→-119→-203→-114→-47→-111→-99→26→145 | 无切换（始终连续改善量，始终全高度，始终势能shaping） |
| 着陆/接触信号 | v1_landing_velocity_penalty(死) → v2_success_landing_bonus(死) → v3_success_landing_bonus(死) → v4_success_landing_bonus(死) → v5_success_landing_bonus(死) → v6_success_landing_bonus(死) → v7_success_landing_bonus(死) → v9_success_landing_bonus | -156→-119→-203→-114→-47→-111→-99→145 | 连续奖励→稀疏bonus：1次（v1→v2）；稀疏bonus↔连续奖励：0次；any_contact↔both_legs：1次（v7→v9）；状态值↔事件触发：1次（v5→v6）；阈值波动≥3次（0.1→0.2→0.4→0.3→0.4→0.4） |
| 稳定性/约束-姿态 | v1_angle_penalty → v2-7_angle_penalty → v8-9_angle_stab | -156→-119→-203→-114→-47→-111→-99→26→145 | 仅角度→角度+角速度：1次（v7→v8） |
| 稳定性/约束-速度 | v8-9_vel_damping | 26→145 | 无历史（v8新增） |
| 效率/约束-能耗 | v1-7_action_cost → v8-9_fuel_penalty | -156→-119→-203→-114→-47→-111→-99→26→145 | 权重不变（均为0.01-0.02量级） |
| 失败惩罚 | v3_crash_penalty(死) | -203 | 仅在v3存在，立即被移除，无切换 |

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
```
