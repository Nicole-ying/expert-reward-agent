# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的数据和原则，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action
- `read_training_feedback(N)` — 某轮训练的详细反馈：分数、每个组件的 ep_sum/active_rate/magnitude_share
- `read_reward_code(N)` — 某轮的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某组件在所有迭代中的 ep_sum_mean 和 active_rate 演化
- `read_environment_card()` — 环境信息：任务目标、观察空间（8维）、动作空间（4动作）、终止条件
- `read_checkpoint_trend(N)` — 某轮训练中各组件在不同训练阶段的趋势

# 修改层级

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态。常见模式参考：

| 证据模式 | 参考变换 | 验证标准 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | 稀疏→连续过程证据 | active_rate 改善，不产生 proxy 徘徊 |
| 极端值支配奖励 | 无界→归一化有界 | 极端轨迹支配下降，方差下降 |
| 占据好状态即可持续获奖 | 状态值→改善量/势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | 全局→阶段门控 | 早期探索与局部约束同时改善 |
| 多个小因子相乘导致塌缩 | 乘积→加性/几何平均 | 非零反馈增多，联合约束保留 |
| 持续事件被重复领取 | 持续→状态转移事件 | 重复积累下降，外部完成保持 |
| proxy 提高但外部任务不升 | 代理→任务完成对齐 | proxy 与外部分数重新同向 |
| 稠密 proxy 形成中分平台 | 全程 proxy→局部/转移任务信号 | 刷新 best，完成相关行为增加 |

- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。

# 关键规则

- **避免打转**：如果 `read_past_reflection` 显示之前已经改过同一语义且结果不好，换方向。
- **接近目标时优先 L1**：如果当前得分或 best 得分接近目标（差距 < 10%），优先 L1 微调。因为之前的经验发现已多次出现分数接近目标时做 L2，结果越改越差。

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

# 上一轮训练反馈 (score=-121.506912)
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

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。

### 输出要求

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.

```
