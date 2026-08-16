# ⚠️ 上一版代码验证失败
错误信息：Reward v1 failed validation: 出现禁止模式: import ; 修复要求：删除所有import和numpy调用；平方根使用**0.5，有界函数使用max/min或不需要库的代数表达式 (record: runs/env_001/rpv4_old_v4_notools/seed_3/iter_08/generation/validations/reward_v1.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励设计：基于 delta 距离的进展奖励 + 软着陆连续奖励 + 姿态速度约束 + 燃油惩罚
    """
    # 权重系数
    w_approach  = 1.0
    w_soft_land = 1.0
    w_stability = 0.05
    w_energy    = 0.05
    w_contact   = 0.1   # 双腿接触对齐小鼓励

    # 辅助常量
    speed_gate_alpha = 2.0
    land_coeff_v     = 2.0
    land_coeff_d     = 2.0

    # --- 当前和下一步状态提取 ---
    # 相对目标平台中心的位置
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 速度
    vx, vy = next_obs[2], next_obs[3]

    # 姿态
    angle      = next_obs[4]
    ang_vel    = next_obs[5]

    # 接触
    left_contact  = next_obs[6]
    right_contact = next_obs[7]

    # 动作：主引擎点燃是 2
    main_engine_used = 1.0 if action == 2 else 0.0

    # --- 计算欧氏距离 ---
    current_dist = (x**2 + y**2)**0.5
    next_dist    = (nx**2 + ny**2)**0.5

    # --- 1. 进展奖励（delta 距离 + 速度门控）---
    delta_dist = current_dist - next_dist   # 正值表示靠近目标
    abs_speed  = abs(vx) + abs(vy)
    speed_gate = 2.718281828 ** (-speed_gate_alpha * abs_speed)   # 用 e 的幂次，避免 import numpy
    approach_delta_gated = w_approach * delta_dist * speed_gate

    # --- 2. 软着陆连续奖励 ---
    # 双腿接触 + 低速 + 靠近中心 → 高奖励
    both_legs = left_contact * right_contact     # 0 或 1
    speed_factor = 2.718281828 ** (-land_coeff_v * abs_speed)
    dist_factor  = 2.718281828 ** (-land_coeff_d * next_dist)
    soft_landing_bonus = w_soft_land * both_legs * speed_factor * dist_factor

    # --- 3. 姿态/角速度稳定性惩罚 ---
    stability_penalty = -w_stability * (abs(angle) + abs(ang_vel))

    # --- 4. 主引擎燃油消耗惩罚 ---
    energy_penalty = -w_energy * main_engine_used

    # --- 5. 双腿接触对齐的小鼓励 ---
    contact_alignment = w_contact * both_legs   # 只有同时接触时才会得到小正向奖励

    # 总奖励
    total_reward = (
        approach_delta_gated
        + soft_landing_bonus
        + stability_penalty
        + energy_penalty
        + contact_alignment
    )

    components = {
        "approach_delta_gated": approach_delta_gated,
        "soft_landing_bonus":   soft_landing_bonus,
        "stability_penalty":    stability_penalty,
        "energy_penalty":       energy_penalty,
        "contact_alignment":    contact_alignment,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **navigation_goal_reaching**，子类型 **goal_approach_and_soft_contact**  
  任务核心：可控飞行器到达中央目标平台，并在双腿接触条件下以低速稳定停靠。

## selected reward roles
根据 `environment_card.md` 的 `reward_role_decomposition`，本次设计覆盖以下职责：

- **主学习信号：goal_proximity**  
  使用 delta 距离 + 速度门控的 `approach_delta_gated` 替代传统绝对值距离奖励。  
  变化：每一步奖励 **距离的减小量**（`current_dist - next_dist`），同时用速度门控压制高速靠近的收益，迫使 agent 在缓慢接近中获得更高奖励。  
  该设计完全依赖环境卡片中明确可用的 `obs[0:2]` 和 `next_obs[2:3]`。

- **软着陆成功：soft_contact_success**  
  采用 **乘积型软条件**（三个因子：双腿接触、低速、高接近度）构建连续奖励。  
  当 agent 双腿同时触地、速度极低且靠近平台中心时，获得接近 1.0 的稠密正向信号，引导最终稳定停靠。  
  该组件仅在条件满足时活跃，与主进展信号互补。

- **健康/稳定约束：stability_penalty**  
  惩罚身体角度（`next_obs[4]`）和角速度（`next_obs[5]`）的绝对值，防止姿态失控。

- **效率约束：energy_penalty**  
  每次使用主引擎（action==2）施加微小惩罚，抑制不必要的燃料消耗。

- **接触对齐辅助：contact_alignment**  
  小幅度奖励双腿同时接触（乘积），进一步鼓励对齐着陆，避免单腿弹跳。

## role_to_signal_mapping and formula operators
| Role | Signals used | Formula operator(s) |
|------|-------------|---------------------|
| goal_proximity | `obs[0:2]`, `next_obs[0:2]`, `next_obs[2:4]` | `improvement_delta` + `preview_conditioned_reward` 风格的 gate：`delta_dist * exp(-α·abs_speed)` |
| soft_contact_success | `next_obs[6]`, `next_obs[7]`, `next_obs[2:4]`, `next_obs[0:2]` | `joint_condition_proxy` (乘积形式) + `bounded_signal` (指数衰减) |
| stability_penalty | `next_obs[4]`, `next_obs[5]` | `absolute_value` 作为线性惩罚 |
| energy_efficiency | `action == 2` | `binary_penalty` |
| contact alignment | `next_obs[6]`, `next_obs[7]` | `product_of_contacts` 小正奖励 |

## excluded roles and reasons
- **terminal_success_reward / terminal_failure_penalty**：环境卡片显式声明 `explicit_success_flag_available: false` 且 `info` 为空，因此无法使用任何基于终止状态的硬奖励/惩罚。
- **fast_approach**：风险过高，可能诱导高速坠毁或失控，与 soft_contact 目标冲突，留到后续迭代。
- **action_smoothing**：离散动作空间，无平滑切换需求，且可能阻碍快速响应。
- **survival_balance_coherence**：本任务无平衡存活需求，主体不会因倾斜直接失败（除非撞击）。

## design hypothesis – why this structure is materially different
- 之前尝试的结构多依赖 **静态距离奖励**（如 `goal_proximity`）与分散的接触奖励。  
  本设计将距离信号改为 **delta进展 + 速度门控**，迫使 agent 在低速条件下逐步靠近目标，从根本上缓解了“高速冲撞依然得到高距离奖励”的风险。
- 软着陆奖励改用 **三个连续因子的乘积**（接触 × 低速 × 接近度），形成在理想着陆条件下唯一的、平滑递增的引导山丘，而不是二元事件奖励。
- 速度门控起到类似 `soft_health_gate` 的作用，在每步动态调节进展收益，鼓励减速靠近，这是之前设计未曾使用的机制。

## future iterations
- 可引入 **动态课程门控**，在训练后期增大 `soft_landing_bonus` 权重。
- 若 agent 出现单腿弹跳问题，可加入 **持续接触时长** 的奖励。
- 待训练稳定后，可降低 `stability_penalty` 权重，仅保留必要的姿态提醒。

## expected failure modes to monitor after training
- **低速徘徊不降落**：agent 可能在远离平台的地方低速震荡，需要检查 `approach_delta_gated` 和 `soft_landing_bonus` 的权重平衡。
- **双腿接触后提前奖励饱和导致不减速**：若 `soft_landing_bonus` 过早给予高值，agent 可能满足于有接触但速度较大的状态，需要调整系数 `land_coeff_v`。
- **过分保守导致难以抵达**：速度门控过严可能抑制必要机动，观察平均接近速度，必要时降低 `speed_gate_alpha`。
- **能量惩罚过大影响探索**：若 `main_engine_used` 比例过低，适当减小 `w_energy`。

下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 第一原则：黑盒实验，不是代码重构

你无法预知新奖励函数会引导出什么行为，只能训练完后看结果。
**看到好趋势 → 先理解为什么好，再精准微调。**
大规模重写等于丢弃已有实验证据，重新掷骰子。

**硬约束：如果本轮得分比上一轮改善超过 30 分，你必须先回答：**
上一轮的哪个修改促成了这个改善？证据是什么？
**如果你回答不了，本轮只做 L1，不改数学形态。**

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

### 本轮数据

# 上一轮训练反馈 (iter_7, score=-307.974882)
# Training Feedback

## Final-policy outcome
score=-307.974882, len=956.800000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-348.075695, -134.394434]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 5647.930624 | 99.9% | 99.9% | 100.0% |
| progress_reward | -3.342327 | -0.1% | 0.1% | 100.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


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
| -753.2 | -1520.9 | -1270.1 | -2162.8 | -643.7 | -924.6 | -199.1 | -412.0 | -568.0 | -298.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| crash_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| landing_reward | 453.561 | 803.509 | 1319.238 | 4740.880 | 1973.111 | 5610.138 | 5996.872 | 5857.065 | 5482.960 | 5906.878 |
| progress_reward | -6.403 | -19.050 | -17.959 | -35.727 | -8.791 | -15.049 | -1.149 | -5.323 | -8.195 | -3.153 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| crash_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| landing_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- observations ----------
    x_pos, y_pos = obs[0], obs[1]
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_left_contact, n_right_contact = next_obs[6], next_obs[7]

    # ---------- distances ----------
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    n_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # ---------- 1. progress reward (approach incentive) ----------
    progress_reward = 5.0 * (dist - n_dist)  # positive when moving toward target

    # ---------- 2. touchdown bonus (task completion) ----------
    # Large one-time reward when both legs touch the platform.
    # Assuming the episode terminates immediately after this condition,
    # so no repeated accumulation of this bonus.
    both_contacts = (n_left_contact > 0.5) and (n_right_contact > 0.5)
    touchdown_bonus = 200.0 if both_contacts else 0.0

    # ---------- 3. crash / boundary constraints ----------
    x_violation = max(0.0, abs(nx_pos) - 1.5)
    v_violation = max(0.0, -ny_vel - 3.0)   # excessive downward speed
    crash_penalty = -1.0 * x_violation - 1.0 * v_violation

    total_reward = progress_reward + touchdown_bonus + crash_penalty

    components = {
        "progress_reward": progress_reward,
        "touchdown_bonus": touchdown_bonus,
        "crash_penalty": crash_penalty,
    }
    return float(total_reward), components
```

---

以上是本轮数据。以下信息可通过工具自主查阅：
- `read_memory()` — 全局记忆表，简单记录每一轮迭代iteration的socre，len长度，奖励函数组件组成
- `read_training_feedback(N)` — 任意迭代的详细训练反馈，了解以前的iteration有过什么好的或者坏的行为，可以避免重蹈覆辙
- `read_reward_code(N)` — 任意迭代的奖励函数代码，方便你深入查看比较好的奖励函数的具体设计
- `read_past_reflection(N)` — 任意迭代的诊断和干预记录，可以查看由好到坏或者由坏到好的记录，吸收学习经验
- `get_component_history(name)` — 组件在所有迭代中的演化，能看清楚某一组件的好的或者坏的演化过程
- `read_environment_card()` — 环境的观察空间、动作空间、终止条件，用来不确定的时候理解每个输入的观测和动作空间的维度信息
- `read_checkpoint_trend(N)` — 训练过程中各组件的趋势，可以看到从训练过程中看出组件是否发挥作用，这样就不仅只看最终反馈，更全面

现在开始调查，做出一个语义方向上的修改。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。
- `signal_completeness`：必要职责是否完备？缺什么信号？
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修预计改善策略？
- `expected_next_round`：下一轮策略学习到的行为模式预计是什么？

**2. 完整 Python 代码**（```python ... ``` 代码块）。