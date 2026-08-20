第 2 轮迭代训练已经结束，得分 63.720709（目标: 200）。

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

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v4/seed_1/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_2, score=63.720709)
# Training Feedback

## Final-policy outcome
score=63.720709, len=668.050000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-208.175480, 265.833071]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1075.630017 | 98.2% | 98.2% | 5.2% |
| velocity_penalty | -9.516978 | -0.9% | 0.9% | 99.0% |
| upright_penalty | -7.300626 | -0.7% | 0.7% | 100.0% |
| goal_proximity | 1.835102 | 0.2% | 0.3% | 98.8% |

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
| -73.8 | -168.1 | -119.8 | -97.7 | -92.9 | -2.9 | 70.8 | -44.1 | -17.7 | 48.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 46.256 | 0.000 | 2.878 | 16.155 | 67.625 | 1049.699 | 4038.918 | 1514.911 | 675.163 | 746.690 |
| goal_proximity | 2.241 | 0.328 | 1.020 | 1.168 | 1.391 | 1.615 | 1.992 | 1.421 | 1.478 | 1.856 |
| upright_penalty | -10.327 | -2.963 | -3.341 | -3.688 | -7.615 | -8.324 | -8.611 | -7.933 | -8.543 | -7.125 |
| velocity_penalty | -32.668 | -9.988 | -10.782 | -9.769 | -9.524 | -9.912 | -8.834 | -10.570 | -11.150 | -8.938 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 2.0% | 0.0% | 0.3% | 0.2% | 1.3% | 3.4% | 16.0% | 5.3% | 4.0% | 3.7% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.6% | 99.5% | 99.4% | 99.1% |
| upright_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% | 99.6% | 99.6% | 99.4% | 99.1% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 2): velocity penalty is now height-gated via
    exp(-height / 2.0). At high altitude, velocity penalty is relaxed,
    allowing the agent to move freely toward the target. Near the ground,
    the penalty ramps up to full strength for soft landing.
    Goal proximity weight increased to 2.0 to strengthen approach incentive.
    """
    # --- goal proximity: encourage reducing Euclidean distance to pad ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 2.0 * (prev_dist - next_dist)   # delta reward (weight raised from 1.0)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 near ground (full penalty), decays to ~0 at high altitude
    height = max(next_obs[1], 0.0)  # clamp to avoid amplification below pad
    gate = 2.718281828 ** (-height / 2.0)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- contact bonus: reward simultaneous leg contact, modulated by horizontal centering ---
    contact_bonus = 0.0
    if next_obs[6] == 1.0 and next_obs[7] == 1.0:
        # Gaussian-like decay with horizontal offset
        centering_factor = 2.718281828 ** (-5.0 * (next_obs[0]**2))
        contact_bonus = 50.0 * centering_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + contact_bonus

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "contact_bonus": contact_bonus
    }
    return total_reward, components
```

# Subagent 数据报告
# 训练数据报告

## 1. 行为事实
- 得分：score=63.720709，range=[-208.175480, 265.833071]
- 终止方式：terminated 17/20，truncated 3/20
- 回合长度：len=668.050000
- Checkpoint 轨迹：官方得分从 cp10%=-73.8 → cp20%=-168.1 → cp30%=-119.8 → cp40%=-97.7 → cp50%=-92.9 → cp60%=-2.9 → cp70%=70.8 → cp80%=-44.1 → cp90%=-17.7 → cp100%=48.7（波动，先降后升再到正分）

## 2. 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | upright_penalty_ep_sum | upright_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -76.561074 | 975.150000 | terminated=1/20, truncated=19/20 | 0.000000 | 0.0% | 0.635053 | 100.0% | -3.714416 | 100.0% | -12.561269 | 100.0% |
| 2 | 63.720709 | 668.050000 | terminated=17/20, truncated=3/20 | 1075.630017 | 5.2% | 1.835102 | 98.8% | -7.300626 | 100.0% | -9.516978 | 99.0% |

## 3. 数字标记

### 主导
- **contact_bonus**：magnitude_share=98.2%（>80%），主导总奖励量级

### 死组件
- 无组件 active_rate < 2%（contact_bonus 为 5.2%，未触发阈值）

### 淹没
- 无组件同时满足 active_rate > 90% 且 magnitude_share < 3%（goal_proximity 为 98.8%/0.3%，upright_penalty 为 100%/0.7%，velocity_penalty 为 99.0%/0.9%，均未同时触发双条件）

### 全覆盖惩罚
- 无组件同时满足 active_rate > 95% 且 signed_share 为负

### 量级对比
- 每步均值（ep_sum ÷ len）：
  - contact_bonus：1075.630017 ÷ 668.05 = **1.610**（最大）
  - goal_proximity：1.835102 ÷ 668.05 = **0.00275**（第二大，绝对值）
  - velocity_penalty：-9.516978 ÷ 668.05 = **-0.01425**（第三大，绝对值）
  - upright_penalty：-7.300626 ÷ 668.05 = **-0.01093**（最小，绝对值）
- 最大的两个：contact_bonus（1.610），velocity_penalty（0.01425，绝对值）
- 最小的两个：goal_proximity（0.00275，绝对值），upright_penalty（0.01093，绝对值）

### 修改变化（iter 2 vs iter 1）
- contact_bonus：ep_sum Δ=+1075.630017，active_rate Δ=+5.2%
- goal_proximity：ep_sum Δ=+1.200049，active_rate Δ=-1.2%
- upright_penalty：ep_sum Δ=-3.586210，active_rate Δ=0.0%
- velocity_penalty：ep_sum Δ=+3.044291（惩罚减轻），active_rate Δ=-1.0%

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_goal_proximity → v2_goal_proximity | -76.56 → 63.72 | 无切换（均为delta距离改善，权重从1.0→2.0） |
| 着陆/接触信号 | v1_contact_bonus(死) → v2_contact_bonus | -76.56(0.00) → 63.72(1075.63) | 稀疏bonus形态未变：0次切换 |
| 稳定性/约束信号 | v1_velocity_penalty → v2_velocity_penalty | -76.56(-12.56) → 63.72(-9.52) | 全高度→高度门控：1次切换 |
| 稳定性/约束信号 | v1_upright_penalty → v2_upright_penalty | -76.56(-3.71) → 63.72(-7.30) | 无切换（均为连续二次型惩罚） |

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
该任务要求控制一个具有两条支撑腿的 2D 刚体飞行器，从视口顶部中央附近开始（带有随机初始力），**快速、低燃料地到达并稳定停靠在中央目标垫上**。核心是到达指定目标位置并实现安全软着陆，附属目标是最小化发动机使用次数（燃料）与总时间。代理人必须学会平滑接近目标、减速、保持竖直姿态并产生双腿的稳定接触。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64
- obs[0]: `x_position` — 飞行器相对于目标垫中心的水平坐标。可用于奖励对靶心的逼近。reward_usable: true
- obs[1]: `y_position` — 飞行器相对于目标垫表面的垂直坐标。可用于奖励下降与接近。reward_usable: true
- obs[2]: `x_velocity` — 水平线速度。可用于惩罚大水平漂移。reward_usable: true
- obs[3]: `y_velocity` — 垂直线速度。控制软着陆冲击，极其重要。reward_usable: true
- obs[4]: `body_angle` — 刚体倾角。竖直向上时接近 0。奖励维持竖直。reward_usable: true
- obs[5]: `angular_velocity` — 角速度。鼓励姿态稳定。reward_usable: true
- obs[6]: `left_support_contact` — 左腿接触标志（0/1）。表示左腿已与地面接触。reward_usable: true
- obs[7]: `right_support_contact` — 右腿接触标志（0/1）。表示右腿已与地面接触。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不开启任何引擎，仅受重力/惯性影响。
- action 1: `left_orientation_engine` — 启动左侧姿态引擎，主要产生逆时针力矩（可能附带微小侧向力），用于校正右倾。
- action 2: `main_engine` — 启动主引擎，产生向上的推力，是主要的减速和高度控制手段。
- action 3: `right_orientation_engine` — 启动右侧姿态引擎，主要产生顺时针力矩，校正左倾。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 当 agent 以低速度、接近竖直的姿态通过双腿接触目标垫，且不再运动（`body_not_awake_or_settled`）时触发。这通常意味着安全着陆成功。
- **failure-like termination**:
  - `crash_or_body_contact`：飞行器任意部位（特别是非腿部）与地面或其它物体发生硬碰撞，导致坠毁。
  - `horizontal_position_outside_viewport`：飞行器水平漂出窗口，视为失控。
- **ambiguous termination**: 若仅因 `body_not_awake_or_settled` 终止，但未满足成功条件（例如悬浮在空中静止，或偏离目标垫静止），则不属于成功；但这类情形在合理动力学下罕见。
- **truncation**: 任务描述中未出现时间截断条件，但实际训练中可能存在未声明的最大步数截断；当前 step 源无 `truncated` 信息，奖励函数中不应依赖该假设。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
  step 返回的 `info` 固定为 `{}`，没有任何成功/失败标志或终止原因字段。奖励函数必须完全从 `next_obs` 的物理状态（位置、速度、接触）中推断是否成功着陆或发生硬碰撞。
- **allowed_info_fields**: 无（info 为空）。
- **forbidden_or_uncertain_info_fields**: `info["success"]`, `info["failure"]`, `info["termination_reason"]` 等未声明字段均禁止使用。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]`（水平差距）, `next_obs[1]`（垂直高度）。两者可直接用于构建距离/接近奖励。
- **velocity**: `next_obs[2]`, `next_obs[3]`。尤其垂向速度对软着陆至关重要。
- **orientation**: `next_obs[4]`（倾角）, `next_obs[5]`（角速度）。可用于姿态稳定性奖励。
- **contact**: `next_obs[6]`, `next_obs[7]`。表示双腿是否触地，是判断着陆成功的关键二值特征。
- **action/engine**: 可利用 `action` 实现燃料惩罚（例如对 1,2,3 动作添加负奖励）。
- **other**: 无其他活跃信号。