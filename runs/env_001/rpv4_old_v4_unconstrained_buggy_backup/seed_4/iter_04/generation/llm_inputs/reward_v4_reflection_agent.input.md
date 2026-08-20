下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_3, score=-23.641136)
# Training Feedback

## Final-policy outcome
score=-23.641136, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-51.845382, 21.329442]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 647.239433 | 91.2% | 91.2% | 88.0% |
| fuel_efficiency | -28.876500 | -4.1% | 4.1% | 96.3% |
| goal_proximity | -16.396123 | -2.3% | 2.3% | 100.0% |
| attitude_constraint | -12.157995 | -1.7% | 1.7% | 100.0% |
| settling_stability | -4.985995 | -0.7% | 0.7% | 100.0% |

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
| 12.0 | -97.1 | -60.4 | -2.6 | 104.7 | 8.0 | -37.0 | 3.3 | 3.7 | -27.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -4.525 | -5.814 | -9.520 | -10.553 | -10.033 | -10.144 | -9.056 | -11.430 | -12.893 | -11.556 |
| fuel_efficiency | -4.781 | -9.031 | -17.270 | -24.893 | -12.985 | -19.476 | -21.246 | -26.015 | -25.214 | -28.524 |
| goal_proximity | -13.930 | -23.008 | -25.183 | -17.817 | -19.144 | -21.642 | -21.873 | -14.437 | -14.056 | -17.702 |
| landing_contact | 48.426 | 43.281 | 292.270 | 471.840 | 222.964 | 301.671 | 381.694 | 724.510 | 723.965 | 588.197 |
| settling_stability | -8.662 | -7.166 | -6.436 | -5.728 | -6.035 | -5.009 | -4.680 | -4.777 | -5.046 | -4.832 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 67.5% | 71.2% | 69.5% | 89.9% | 72.7% | 86.9% | 79.8% | 93.2% | 90.9% | 95.1% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 33.2% | 29.0% | 61.5% | 87.5% | 60.9% | 67.0% | 72.3% | 86.0% | 88.1% | 86.9% |
| settling_stability | 98.9% | 99.8% | 100.0% | 99.8% | 99.2% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。
    
    核心改动：从线性 state-cost → 二次 state-cost（梯度随状态变化）
    - goal_proximity: 二次距离惩罚，梯度 ∝ distance（远处强、近处弱）
    - settling_stability: 二次速度惩罚，梯度 ∝ speed（高速强、低速弱）
    - landing_contact: 渐进式奖励，软化门控提高 active_rate
    - attitude_constraint: 保持结构，提高远距离基线权重
    
    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """
    
    # ========== 信号提取（使用 next_obs 评估转移后状态） ==========
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
    # 梯度 = -0.2 * distance，远处梯度强（d=3→-0.6），近处梯度弱（d=0.3→-0.06）
    # 天然 curriculum：远处强力驱动靠近，近处精细调控
    goal_proximity = -0.1 * distance ** 2
    
    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度 = -0.6 * speed * gate，高速时梯度强，且随接近目标增强
    # gate 范围：远距离约 0.28，近距离约 1.0，提供 ~3.6x 动态范围
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    # 角速度惩罚（新增独立项）
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)
    
    # ========== Component C: attitude_constraint — 姿态代价 ==========
    # 保持线性但将远距离基线从 0.15 提高到 0.3，更早开始姿态学习
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate
    
    # ========== Component D: landing_contact — 渐进式着陆奖励 ==========
    # 软化门控：distance<0.5（原 0.3），speed<1.0（原 0.6）
    # 使用连续缩放因子替代二元门控，提供稠密梯度
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    
    if near_pad and slow_enough:
        # 连续因子：越近越好、越慢越好
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        contact_frac = (left_contact + right_contact) / 2.0  # 0.0~1.0
        
        # base_bonus: 靠近+低速即有奖励，引导 agent 先学会接近
        base_bonus = 1.5 * prox_factor * speed_factor
        # contact_bonus: 接触额外奖励，推动完成最后一步
        contact_bonus = 2.0 * prox_factor * speed_factor * contact_frac
        
        landing_contact = base_bonus + contact_bonus
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