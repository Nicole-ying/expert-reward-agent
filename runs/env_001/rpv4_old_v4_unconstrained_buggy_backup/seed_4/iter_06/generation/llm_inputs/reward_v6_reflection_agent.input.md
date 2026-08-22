下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_5, score=82.657106)
# Training Feedback

## Final-policy outcome
score=82.657106, len=961.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[38.817958, 254.357664]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 478.885087 | 91.4% | 91.4% | 58.4% |
| fuel_efficiency | -16.294500 | -3.1% | 3.1% | 56.5% |
| attitude_constraint | -12.265481 | -2.3% | 2.3% | 100.0% |
| goal_proximity | -10.262364 | -2.0% | 2.0% | 100.0% |
| settling_stability | -6.315288 | -1.2% | 1.2% | 99.9% |

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
| -36.9 | -91.5 | -56.9 | 97.0 | 233.1 | 226.4 | 112.7 | 89.9 | 274.1 | 86.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -2.277 | -9.922 | -10.240 | -15.475 | -11.200 | -9.578 | -10.779 | -9.825 | -6.050 | -11.454 |
| fuel_efficiency | -1.320 | -4.033 | -7.011 | -10.543 | -4.909 | -4.332 | -15.318 | -14.212 | -5.911 | -15.457 |
| goal_proximity | -8.318 | -13.026 | -17.169 | -15.374 | -9.932 | -9.475 | -10.085 | -10.224 | -10.077 | -10.087 |
| landing_contact | 0.454 | 5.927 | 4.112 | 8.160 | 28.171 | 14.662 | 411.186 | 544.711 | 23.168 | 509.282 |
| settling_stability | -11.932 | -14.954 | -16.059 | -9.269 | -7.313 | -9.080 | -6.233 | -5.978 | -5.698 | -6.347 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 48.9% | 59.9% | 63.7% | 60.6% | 47.2% | 61.4% | 57.9% | 48.4% | 71.6% | 53.4% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 3.4% | 20.2% | 5.7% | 9.1% | 44.8% | 23.5% | 57.5% | 62.8% | 27.3% | 61.0% |
| settling_stability | 100.0% | 99.6% | 98.9% | 98.6% | 97.4% | 95.6% | 99.6% | 99.9% | 95.4% | 99.9% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v5。
    
    核心改动：landing_contact 从持续状态奖励 → 状态转移奖励。
    - sustain: 小额维持奖励（0.3x），防止信号完全消失但不再鼓励徘徊
    - transition: 大额转移奖励（4.0x），仅在接触程度增加时发放
    - 使用 obs 和 next_obs 的接触状态差来检测转移事件
    
    设计原则：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 转移型接触奖励 → 奖励"取得接触"而非"保持接触"
    - attitude_constraint: 姿态代价 → 保持平稳
    - fuel_efficiency: 燃料代价 → 节省推力
    
    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """
    
    # ========== 信号提取 ==========
    # 上一状态（用于检测转移）
    prev_left_contact = obs[6]
    prev_right_contact = obs[7]
    
    # 当前状态（转移后）
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== 辅助量 ==========
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    
    # proximity_gate: 1.0 在目标处，~0.1 在 distance≈3
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)
    
    # ========== Component A: goal_proximity — 二次距离代价 ==========
    # 梯度 = -0.2 * distance，远处梯度强，近处梯度弱（天然 curriculum）
    goal_proximity = -0.1 * distance ** 2
    
    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度随 speed 和 proximity 增强，接近目标时减速约束更强
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)
    
    # ========== Component C: attitude_constraint — 姿态代价 ==========
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    # gate 范围：远距离 0.3，近距离 1.0
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate
    
    # ========== Component D: landing_contact — 转移型着陆奖励 ==========
    # v5 核心改动：从持续状态奖励改为状态转移奖励。
    # - sustain: 小额维持信号（系数 0.3），不再鼓励在接触区徘徊
    # - transition: 大额转移信号（系数 4.0），强烈奖励"取得/增加接触"
    # 这防止了 agent 在 near_pad+slow_enough 区域内无限徘徊领取 contact 奖励。
    prev_contact_frac = (prev_left_contact + prev_right_contact) / 2.0
    curr_contact_frac = (left_contact + right_contact) / 2.0
    contact_gain = max(0.0, curr_contact_frac - prev_contact_frac)  # 仅奖励接触增加
    
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    
    if near_pad and slow_enough:
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        
        # 维持奖励：微弱正向信号，保持接触状态的价值感知
        sustain = 0.3 * prox_factor * speed_factor * curr_contact_frac
        # 转移奖励：接触程度增加时的大额奖励，驱动"完成着陆"这个动作
        transition = 4.0 * prox_factor * speed_factor * contact_gain
        
        landing_contact = sustain + transition
    else:
        landing_contact = 0.0
    
    # ========== Component E: fuel_efficiency — 燃料代价 ==========
    fuel_efficiency = -0.03 * float(action != 0)
    
    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )
    
    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
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