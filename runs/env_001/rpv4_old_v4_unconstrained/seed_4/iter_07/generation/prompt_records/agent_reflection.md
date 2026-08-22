# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件。
- `read_checkpoint_trend(iteration)` — 查看某轮训练中各组件随训练的演化轨迹。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。


```

## User Prompt

```markdown
下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_6, score=-5.348697)
# Training Feedback

## Final-policy outcome
score=-5.348697, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-29.437328, 29.879625]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_warmth | 45.131376 | 45.7% | 45.7% | 100.0% |
| fuel_penalty | -27.123000 | -27.5% | 27.5% | 90.4% |
| approach_reward | 19.772798 | 20.0% | 23.2% | 100.0% |
| velocity_penalty | -1.960114 | -2.0% | 2.0% | 100.0% |
| angle_penalty | -1.670077 | -1.7% | 1.7% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
| -39.2 | -370.7 | -230.1 | -236.8 | -191.1 | -82.9 | -22.8 | -39.0 | -15.2 | -5.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | -1.179 | -0.961 | -0.923 | -0.932 | -0.908 | -1.000 | -1.261 | -1.443 | -1.319 | -1.741 |
| approach_reward | 8.866 | -32.140 | -11.429 | -8.161 | -2.078 | 6.896 | 16.776 | 14.232 | 18.116 | 19.869 |
| boundary_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fuel_penalty | -13.305 | -18.394 | -17.917 | -15.817 | -17.586 | -19.207 | -23.487 | -24.171 | -25.409 | -27.104 |
| landing_bonus | 282.902 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| proximity_warmth | 22.081 | 12.799 | 22.775 | 23.109 | 24.909 | 33.301 | 40.848 | 39.244 | 42.925 | 45.090 |
| velocity_penalty | -3.036 | -1.041 | -0.856 | -1.172 | -1.483 | -0.921 | -1.169 | -1.402 | -1.375 | -1.993 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| boundary_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fuel_penalty | 80.7% | 61.4% | 61.3% | 62.0% | 72.8% | 64.0% | 78.3% | 80.6% | 84.7% | 90.3% |
| landing_bonus | 11.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| proximity_warmth | 100.0% | 78.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化 v6 — 融合 iter_4 有效信号 + iter_5 轻量惩罚。

    核心设计：
    - approach_reward：强 delta 驱动（15.0 * dist_improvement），不可 exploit
    - proximity_warmth：微小状态奖励，仅解决冷启动，远小于着陆收益
    - velocity/angle 惩罚：门控 + 轻量，远距离几乎为零
    - landing_bonus：渐进乘积式，从靠近到完美着陆的连续梯度
    - boundary_penalty：预防飞出视口
    - fuel_penalty：轻量引擎使用惩罚
    """

    # --- 当前状态 ---
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    # --- 下一状态 ---
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # --- 派生量 ---
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ================================================================
    # 1. approach_reward — 强 delta 驱动（核心学习信号）
    #
    #    纯势能差：reward = 15 * (current_dist - next_dist)
    #    - 靠近目标 → 正奖励（每 0.1 单位 ≈ +1.5）
    #    - 远离目标 → 对称惩罚
    #    - 悬停/振荡 → 净收益 = 0（不可 exploit）
    #    总 episode 贡献 = 15 * (initial_dist - final_dist)，路径无关
    # ================================================================
    dist_improvement = current_dist - next_dist
    approach_reward = 15.0 * dist_improvement

    # ================================================================
    # 2. proximity_warmth — 微小状态奖励，仅防冷启动
    #
    #    线性衰减：dist=0 → 0.05, dist=1 → 0.033, dist=3 → 0
    #    即使 agent 在 dist=1 悬停 1000 步也只获得 ~33 点——远小于
    #    着陆后的 landing_bonus（可达 5+/步），不可 exploit。
    #    唯一作用：给早期随机探索的 agent 一个活下来的微弱理由。
    # ================================================================
    proximity_warmth = 0.05 * max(0.0, 1.0 - next_dist / 3.0)

    # ================================================================
    # 3. velocity_penalty — 近距门控速度管理
    #
    #    gate = 1/(1+5*dist)：dist=0→1.0, dist=1→0.17, dist=5→0.038
    #    远处几乎不干预，近处温和引导减速。
    #    系数 0.3 相比 iter_4 的 -1.0*next_speed*prox_gate 大幅降低，
    #    防止 iter_4 中 settling_stability 以 59% magnitude 碾压一切的问题。
    # ================================================================
    vel_gate = 1.0 / (1.0 + 5.0 * next_dist)
    velocity_penalty = -0.3 * vel_gate * (nvx**2 + nvy**2)

    # ================================================================
    # 4. angle_penalty — 近距门控姿态约束
    #
    #    比 iter_4/iter_5 都更温和，避免 attitude 惩罚阻碍中距离机动。
    #    gate 在远处趋近 0，近处才生效。
    # ================================================================
    angle_gate = 1.0 / (1.0 + 3.0 * next_dist)
    angle_penalty = -angle_gate * (1.0 * n_angle**2 + 0.5 * n_ang_vel**2)

    # ================================================================
    # 5. boundary_penalty — 水平越界预警
    #
    #    阈值 1.2（比 iter_5 的 1.0 稍宽）用二次惩罚在越界时急剧增大。
    # ================================================================
    safe_x_bound = 1.2
    if abs(nx) > safe_x_bound:
        boundary_penalty = -2.0 * (abs(nx) - safe_x_bound) ** 2
    else:
        boundary_penalty = 0.0

    # ================================================================
    # 6. landing_bonus — 渐进着陆质量奖励
    #
    #    四个连续因子（0~1）的乘积 × 12.0：
    #    - prox_factor:   dist=0→1.0,  dist≥0.4→0（靠近目标垫板）
    #    - speed_factor:  speed=0→1.0, speed≥0.4→0（低速着陆）
    #    - contact_factor: 0(无接触), 0.5(单腿), 1.0(双腿)
    #    - angle_factor:  |angle|=0→1.0, |angle|≥0.25→0（姿态平稳）
    #
    #    与 iter_4 的关键区别：全部使用连续因子而非二元门控，
    #    即使只满足部分条件也有部分奖励，创造平滑学习梯度。
    #    iter_4 证明 landing 信号可触达（active 0.8%, ep_sum +10.5）；
    #    连续化后 accessibility 应显著提升。
    # ================================================================
    prox_factor = max(0.0, 1.0 - next_dist / 0.4)
    speed_factor = max(0.0, 1.0 - next_speed / 0.4)
    contact_factor = 0.5 * (n_left + n_right)  # 0.0 / 0.5 / 1.0
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.25)

    landing_bonus = 12.0 * prox_factor * speed_factor * contact_factor * angle_factor

    # ================================================================
    # 7. fuel_penalty — 轻量引擎使用惩罚
    # ================================================================
    fuel_penalty = -0.03 * float(action != 0)

    # ================================================================
    # 汇总
    # ================================================================
    total_reward = (
        approach_reward +
        proximity_warmth +
        velocity_penalty +
        angle_penalty +
        boundary_penalty +
        landing_bonus +
        fuel_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "proximity_warmth": proximity_warmth,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "boundary_penalty": boundary_penalty,
        "landing_bonus": landing_bonus,
        "fuel_penalty": fuel_penalty,
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

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

现在开始调查，分析后输出诊断和代码。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。

**2. 完整 Python 代码**（```python ... ``` 代码块）。
```
