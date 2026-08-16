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
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=-117.237675)
# Training Feedback

## Final-policy outcome
score=-117.237675, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.279795, -98.837163]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_damping | -21.082070 | -44.1% | 44.1% | 100.0% |
| proximity_reward | -19.972254 | -41.8% | 41.8% | 100.0% |
| landing_bonus | 2.500000 | 5.2% | 5.2% | 0.4% |
| action_reward | 1.744500 | 3.7% | 4.3% | 100.0% |
| contact_reward | 1.625000 | 3.4% | 3.4% | 3.1% |
| orientation_penalty | -0.540341 | -1.1% | 1.1% | 100.0% |

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
| -117.7 | -119.8 | -124.0 | -126.1 | -119.1 | -118.1 | -119.6 | -121.8 | -120.7 | -119.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 1.944 | 1.688 | 1.728 | 1.772 | 1.764 | 1.686 | 1.716 | 1.796 | 1.782 | 1.742 |
| contact_reward | 2.075 | 1.750 | 1.525 | 1.550 | 1.625 | 1.800 | 1.800 | 1.625 | 1.525 | 1.575 |
| landing_bonus | 1.500 | 1.500 | 1.500 | 1.500 | 1.500 | 2.500 | 2.500 | 2.500 | 2.500 | 2.500 |
| orientation_penalty | -1.070 | -0.900 | -0.861 | -0.899 | -0.751 | -0.652 | -0.635 | -0.773 | -0.772 | -0.752 |
| proximity_reward | -20.191 | -20.021 | -20.003 | -19.995 | -19.984 | -20.000 | -19.996 | -19.983 | -19.960 | -19.961 |
| vel_damping | -21.301 | -21.180 | -21.099 | -21.124 | -21.097 | -21.139 | -21.153 | -21.077 | -21.067 | -21.076 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_reward | 4.9% | 4.0% | 3.5% | 3.5% | 3.4% | 3.5% | 3.6% | 3.5% | 3.0% | 3.0% |
| landing_bonus | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.4% | 0.4% | 0.4% | 0.4% | 0.4% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_damping | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Complete rewrite.
    - Negative distance to pad (smooth, always active goal proximity)
    - Orientation penalty (quadratic on angle + angvel)
    - Velocity damping via exponential proximity gate (always active)
    - Progressive contact reward (any foot → bonus, both → more)
    - Landing success bonus (large per-step reward when settled)
    - Action efficiency (reward idle, penalize engine use)
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ──── 1. Goal proximity: negative distance to origin ────
    dist = (x ** 2 + y ** 2) ** 0.5
    w_proximity = 0.3
    proximity_reward = -w_proximity * dist

    # ──── 2. Orientation penalty ────
    w_angle = 0.08
    w_angvel = 0.03
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ──── 3. Velocity damping (exponential gate, always active) ────
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # gate ≈ 1 when dist ≈ 0, gate → 0 when dist ≫ σ
    sigma_dist = 1.5
    proximity_gate = 2.718281828 ** (- (dist ** 2) / (sigma_dist ** 2))
    w_damp = 0.4
    vel_damping = -w_damp * proximity_gate * speed

    # ──── 4. Progressive contact reward ────
    # 0, 1, or 2 feet on ground — any contact is progress
    contact_sum = left_contact + right_contact
    w_contact = 0.5
    contact_reward = w_contact * contact_sum

    # ──── 5. Landing success bonus ────
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist < 0.3)
    upright = (abs(angle) < 0.15)
    slow = (speed < 0.3)
    is_good_landing = both_on_ground and near_pad and upright and slow
    w_landing = 10.0
    landing_bonus = w_landing * (1.0 if is_good_landing else 0.0)

    # ──── 6. Action efficiency ────
    if action == 0:          # no engine — reward gliding
        action_reward = 0.03
    elif action == 2:        # main engine — highest cost
        action_reward = -0.08
    else:                    # orientation engines (1 or 3) — low cost
        action_reward = -0.03

    # ──── Aggregate ────
    total_reward = (
        proximity_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
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
