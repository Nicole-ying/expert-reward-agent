# ⚠️ 上一版代码验证失败
错误信息：Reward v6 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs/env_001/rpv4_old_v4_notools/seed_3/iter_06/generation/validations/reward_v6.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
我先查阅历史记录和训练反馈，以理解之前的修改路径，避免重复错误。Action: read_memory
Action Input: {}

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

# 上一轮训练反馈 (iter_5, score=-307.974882)
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
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left_contact, n_right_contact = next_obs[6], next_obs[7]

    # Distance to target platform (center at 0,0)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    n_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # ---------- 1. Main progress signal: improvement delta ----------
    progress_reward = 2.0 * (dist - n_dist)  # positive when approaching

    # ---------- 2. Soft landing proxy: joint condition average ----------
    # Proximity factor
    prox = 1.0 / (1.0 + 5.0 * n_dist)
    # Contact factor
    contact = (n_left_contact + n_right_contact) / 2.0
    # Angle factor
    angle_factor = max(0.0, 1.0 - abs(n_body_angle) / 0.8)
    # Angular velocity factor
    angvel_factor = max(0.0, 1.0 - abs(n_ang_vel) / 2.0)
    # Speed factor
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    speed_factor = max(0.0, 1.0 - speed / 2.0)

    landing_proxy = (prox + contact + angle_factor + angvel_factor + speed_factor) / 5.0
    landing_reward = 10.0 * landing_proxy

    # ---------- 3. Crash / boundary constraint: hinge penalties ----------
    # Horizontal drift penalty
    x_violation = max(0.0, abs(nx_pos) - 1.5)
    # Excessive downward velocity penalty
    v_violation = max(0.0, -ny_vel - 3.0)
    crash_penalty = -1.0 * x_violation - 1.0 * v_violation

    total_reward = progress_reward + landing_reward + crash_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "crash_penalty": crash_penalty,
    }
    return total_reward, components
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