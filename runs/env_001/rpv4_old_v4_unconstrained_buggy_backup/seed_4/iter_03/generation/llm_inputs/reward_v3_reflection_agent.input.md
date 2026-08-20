下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=-113.598118)
# Training Feedback

## Final-policy outcome
score=-113.598118, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.587355, -96.181860]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -33.249548 | -61.5% | 61.5% | 100.0% |
| settling_stability | -18.853121 | -34.9% | 34.9% | 100.0% |
| landing_contact | 1.125000 | 2.1% | 2.1% | 0.7% |
| attitude_constraint | -0.752751 | -1.4% | 1.4% | 100.0% |
| fuel_efficiency | -0.091500 | -0.2% | 0.2% | 4.5% |

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
| -120.8 | -117.8 | -123.4 | -122.2 | -118.4 | -118.2 | -115.8 | -116.0 | -115.5 | -113.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -2.023 | -1.559 | -1.616 | -1.370 | -1.042 | -0.998 | -0.878 | -0.882 | -0.833 | -0.702 |
| fuel_efficiency | -0.183 | -0.195 | -0.084 | -0.078 | -0.098 | -0.075 | -0.081 | -0.075 | -0.080 | -0.088 |
| goal_proximity | -33.294 | -33.267 | -33.274 | -33.260 | -33.248 | -33.259 | -33.252 | -33.253 | -33.251 | -33.252 |
| landing_contact | 0.790 | 0.790 | 0.875 | 0.790 | 0.875 | 0.915 | 0.830 | 0.830 | 0.915 | 1.165 |
| settling_stability | -18.932 | -18.889 | -18.918 | -18.913 | -18.893 | -18.950 | -18.946 | -18.946 | -18.957 | -18.883 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 8.9% | 9.5% | 4.1% | 3.8% | 4.8% | 3.7% | 3.9% | 3.7% | 3.9% | 4.3% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 0.5% | 0.5% | 0.5% | 0.5% | 0.5% | 0.6% | 0.6% | 0.6% | 0.6% | 0.7% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    
    核心改动：从 improvement_delta（改进量）框架切换到 state-cost（状态代价）框架。
    - 不再奖励"距离减小"（重力会免费提供），改为惩罚"距离大"
    - 不再奖励"速度降低"，改为惩罚"速度高"，且随接近目标增强
    - 着陆奖励分层，软化门控提高 active_rate
    
    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """
    
    # ========== 信号提取（主要使用 next_obs 评估转移后的状态） ==========
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
    
    # proximity_gate: 1.0 在目标处，~0.1 在远处 (distance≈3)
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)
    
    # ========== Component A: goal_proximity — 距离代价 ==========
    # 距离越远代价越大，驱使 agent 靠近目标垫板
    # 线性代价，在 distance≈3 时约 -1.5，在 distance≈0 时约 0
    goal_proximity = -0.5 * distance
    
    # ========== Component B: settling_stability — 速度代价 ==========
    # 速度越高代价越大，且随接近目标急剧增强
    # 远距离时弱（允许快速接近），近距离时强（强制减速）
    settling_stability = -0.6 * speed * (0.15 + 0.85 * proximity_gate)
    
    # ========== Component C: attitude_constraint — 姿态代价 ==========
    # 角度和角速度偏离的代价，同样随 proximity 增强
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    attitude_constraint = -0.4 * angle_error * (0.15 + 0.85 * proximity_gate)
    attitude_constraint += -0.15 * ang_vel_mag * (0.15 + 0.85 * proximity_gate)
    
    # ========== Component D: landing_contact — 着陆奖励 ==========
    # 分层奖励：双接触+直立 > 单接触 > 靠近但无接触
    near_pad = distance < 0.3
    slow_enough = speed < 0.6
    upright = angle_error < 0.3
    dual_contact = (left_contact > 0.5 and right_contact > 0.5)
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    
    if near_pad and slow_enough:
        if dual_contact and upright:
            landing_contact = 2.5  # 完美软着陆
        elif dual_contact:
            landing_contact = 1.5  # 双接触但姿态不佳
        elif any_contact:
            landing_contact = 0.8  # 单侧接触
        else:
            landing_contact = 0.2  # 靠近+低速但未接触（鼓励继续下降至接触）
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