下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

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

# 上一轮训练反馈 (iter_1, score=-121.506912)
# Training Feedback

## Final-policy outcome
score=-121.506912, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-173.501602, 11.386677]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | 0.443023 | 5.5% | 36.0% | 100.0% |
| goal_proximity | 2.148804 | 26.5% | 29.6% | 100.0% |
| attitude_constraint | -1.807692 | -22.3% | 25.4% | 100.0% |
| landing_contact | 0.500000 | 6.2% | 6.2% | 0.2% |
| fuel_efficiency | -0.227500 | -2.8% | 2.8% | 6.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
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
| -113.9 | -112.6 | -113.6 | -114.9 | -121.1 | -113.9 | -120.0 | -121.5 | -113.8 | -121.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -1.443 | -1.409 | -1.418 | -1.496 | -1.421 | -1.486 | -1.906 | -1.833 | -1.456 | -1.764 |
| fuel_efficiency | -0.090 | -0.110 | -0.120 | -0.090 | -0.238 | -0.135 | -0.198 | -0.185 | -0.140 | -0.217 |
| goal_proximity | 2.149 | 2.149 | 2.148 | 2.149 | 2.150 | 2.149 | 2.158 | 2.152 | 2.149 | 2.149 |
| landing_contact | 0.500 | 0.300 | 0.300 | 0.300 | 0.500 | 0.300 | 0.300 | 0.300 | 0.300 | 0.500 |
| settling_stability | 0.563 | 0.567 | 0.564 | 0.523 | 0.525 | 0.517 | 0.421 | 0.435 | 0.522 | 0.445 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 2.6% | 3.1% | 3.4% | 2.6% | 6.8% | 3.9% | 5.8% | 5.4% | 4.0% | 6.4% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 0.2% | 0.1% | 0.1% | 0.1% | 0.2% | 0.1% | 0.1% | 0.1% | 0.1% | 0.2% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v1。
    核心目标：尽快到达中心目标垫板并稳定停靠，同时鼓励双支撑腿安全着陆。
    
    Args:
        obs: 当前观测 (8,) [x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact]
        action: 当前动作 (0-3: no_engine, left_orient, main_engine, right_orient)
        next_obs: 下一观测 (8,)
        original_reward: 原始奖励（不使用）
        info: 信息字典（为空，不使用）
        training_progress: 训练进度（v1不使用）
    
    Returns:
        float: 总奖励
        dict: 各奖励组件
    """
    
    # ========== 信号提取 ==========
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    # ========== Component A: 主学习信号 - 目标接近与稳定 ==========
    # role: goal_proximity + settling_stability
    # 策略：使用 improvement_delta 鼓励朝目标方向移动并减速
    # 水平方向：减少 |x_pos|，理想为 0（对准垫板中心）
    # 垂直方向：减少 |y_pos|，理想为 0（贴齐垫板表面）
    # 同时鼓励速度归零（settling），权重随接近目标而升高
    
    # 当前位置距离（L1范数，简单有效）
    current_distance = abs(x_pos) + abs(y_pos)
    next_distance = abs(next_x_pos) + abs(next_y_pos)
    
    # 位置改进奖励：距离减少得正奖励（improvement_delta）
    distance_improvement = current_distance - next_distance
    
    # 速度惩罚：鼓励减速，尤其在接近目标时
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    
    # 速度改进：速度降低得奖励（越接近目标，权重越高）
    # 使用 gating: 距离越近，速度惩罚越重要
    proximity_gate = 1.0 / (1.0 + 5.0 * current_distance)  # 距离远时≈0，距离0时≈1
    speed_improvement = current_speed - next_speed
    
    # 组合：位置改进为主驱动，速度改进为辅约束
    goal_proximity_reward = 2.0 * distance_improvement  # 主导航信号
    settling_reward = 0.5 * speed_improvement * (1.0 + proximity_gate * 3.0)  # 接近时加强速度控制
    
    # ========== Component B: 姿态稳定约束 ==========
    # role: upright_orientation (条件激活) + 部分 settling_stability
    # 仅当接近目标时激活角度惩罚，避免干扰大角度机动
    
    # 当前和下一姿态误差
    current_angle_error = abs(body_angle)
    next_angle_error = abs(next_body_angle)
    current_ang_vel_mag = abs(ang_vel)
    next_ang_vel_mag = abs(next_ang_vel)
    
    # 姿态改进：角度减小、角速度减小得奖励
    angle_improvement = current_angle_error - next_angle_error
    ang_vel_improvement = current_ang_vel_mag - next_ang_vel_mag
    
    # 仅当接近目标时施加姿态约束（y_pos < 0.5 且 distance < 1.0）
    attitude_gate = 1.0 if (abs(y_pos) < 0.5 and current_distance < 1.0) else 0.1
    attitude_reward = attitude_gate * (1.0 * angle_improvement + 0.5 * ang_vel_improvement)
    
    # ========== Component C: 安全着陆接触奖励 ==========
    # role: safe_landing_contact
    # 当接近目标、速度低、且双支撑腿接触时给予正向奖励
    # 使用 joint_condition_proxy 构造软接触成功信号
    
    # 接近目标条件
    near_target = abs(next_x_pos) < 0.2 and abs(next_y_pos) < 0.2
    
    # 低速条件（L1速度小于阈值）
    low_speed = (abs(next_x_vel) + abs(next_y_vel)) < 0.5
    
    # 双接触条件
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)
    
    # 接触改进：从单边或无边到双接触
    contact_improvement = 0.0
    if both_contacts and not (left_contact > 0.5 and right_contact > 0.5):
        contact_improvement = 0.5  # 新获得双接触
    
    # 组合门控：仅当接近且低速时，接触才有意义
    if near_target and low_speed:
        landing_reward = 3.0 * float(both_contacts)  # 鼓励维持双接触
        landing_reward += 2.0 * contact_improvement  # 额外奖励首次双接触
    else:
        landing_reward = 0.0
    
    # ========== Component D: 燃料效率惩罚（轻量） ==========
    # role: fuel_efficiency
    # 对使用引擎施加轻微惩罚，避免无意义动作
    # action != 0 表示使用了引擎（离散动作空间）
    fuel_penalty = -0.05 * float(action != 0)
    
    # ========== 汇总奖励 ==========
    total_reward = (
        goal_proximity_reward +
        settling_reward +
        attitude_reward +
        landing_reward +
        fuel_penalty
    )
    
    components = {
        'goal_proximity': goal_proximity_reward,
        'settling_stability': settling_reward,
        'attitude_constraint': attitude_reward,
        'landing_contact': landing_reward,
        'fuel_efficiency': fuel_penalty
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