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

# 上一轮训练反馈 (iter_8, score=-110.443065)
# Training Feedback

## Final-policy outcome
score=-110.443065, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-126.734188, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 0.277499 | 40.2% | 40.2% | 1.7% |
| stability_penalty | -0.213771 | -31.0% | 31.0% | 100.0% |
| contact_alignment | 0.115000 | 16.7% | 16.7% | 1.7% |
| approach_delta_gated | 0.070212 | 10.2% | 12.2% | 100.0% |
| energy_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| -110.9 | -111.0 | -109.5 | -108.0 | -107.9 | -109.0 | -108.3 | -108.5 | -108.9 | -110.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_delta_gated | 0.069 | 0.070 | 0.071 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 | 0.070 |
| contact_alignment | 0.115 | 0.115 | 0.135 | 0.125 | 0.125 | 0.125 | 0.135 | 0.130 | 0.125 | 0.115 |
| energy_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| soft_landing_bonus | 0.284 | 0.351 | 0.263 | 0.294 | 0.299 | 0.314 | 0.340 | 0.318 | 0.319 | 0.295 |
| stability_penalty | -0.361 | -0.285 | -0.244 | -0.229 | -0.186 | -0.188 | -0.185 | -0.173 | -0.211 | -0.211 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_delta_gated | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_alignment | 1.7% | 1.7% | 2.0% | 1.8% | 1.8% | 1.8% | 2.0% | 1.9% | 1.8% | 1.7% |
| energy_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| soft_landing_bonus | 1.7% | 1.7% | 2.0% | 1.8% | 1.8% | 1.8% | 2.0% | 1.9% | 1.8% | 1.7% |
| stability_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
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
    speed_gate = 2.718281828 ** (-speed_gate_alpha * abs_speed)   # 用 e 的幂次，无需额外库
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

---

以上是本轮数据。现在开始调查，做出一个语义方向上的修改。

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