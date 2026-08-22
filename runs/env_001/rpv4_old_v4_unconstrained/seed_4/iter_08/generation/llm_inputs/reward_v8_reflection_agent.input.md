下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_7, score=-115.610232)
# Training Feedback

## Final-policy outcome
score=-115.610232, len=948.450000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-227.490867, 13.995974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 12.052421 | 16.0% | 41.1% | 100.0% |
| fuel_penalty | -28.453500 | -37.7% | 37.7% | 100.0% |
| velocity_penalty | -9.035890 | -12.0% | 12.0% | 100.0% |
| close_proximity | 4.308586 | 5.7% | 5.7% | 29.9% |
| angle_penalty | -2.619085 | -3.5% | 3.5% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -24.7 | -119.2 | -150.3 | -118.6 | -92.2 | -116.5 | -5.2 | -123.9 | -143.8 | -98.2 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | -6.850 | -14.346 | -2.987 | -2.340 | -2.787 | -2.704 | -4.981 | -3.659 | -2.734 | -2.720 |
| approach_reward | 15.498 | 10.701 | 9.172 | 10.938 | 13.455 | 9.801 | 19.797 | 11.995 | 9.662 | 13.172 |
| boundary_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| close_proximity | 3.350 | 0.461 | 0.058 | 3.856 | 4.821 | 3.266 | 18.487 | 9.989 | 5.132 | 4.817 |
| contact_reward | 543.700 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 15.700 | 35.650 | 0.000 | 0.000 |
| fuel_penalty | -5.847 | -15.315 | -18.675 | -16.225 | -19.076 | -21.197 | -25.278 | -23.784 | -23.018 | -29.054 |
| velocity_penalty | -28.179 | -21.954 | -10.755 | -12.629 | -10.117 | -11.238 | -9.007 | -11.902 | -11.891 | -8.637 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| approach_reward | 99.7% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| boundary_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| close_proximity | 34.0% | 9.7% | 8.9% | 34.7% | 43.0% | 26.1% | 64.6% | 39.6% | 31.5% | 32.7% |
| contact_reward | 6.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.3% | 0.0% | 0.0% |
| fuel_penalty | 76.0% | 63.4% | 80.5% | 87.6% | 83.2% | 99.6% | 99.9% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化 v7 — 根治租金 exploit，重构着陆信号。

    核心设计哲学：
    - 移除所有可在中途"收租"的状态奖励（proximity_warmth 是 iter_6 的罪魁祸首）
    - approach_reward（路径无关势能差）是唯一 backbone 信号，不可 exploit
    - 着陆奖励改为加法独立组件：每条腿独立奖励 + 双腿额外奖励 + 完美着陆大奖
      降低发现门槛，确保 agent 在探索中能偶然触及
    - 极近距离的微弱梯度（<0.4）仅用于在目标附近提供额外方向指引
    - velocity/angle 惩罚在远处几乎为零，近处温和引导减速和姿态
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
    # 1. approach_reward — 路径无关势能差（backbone，不可 exploit）
    #
    #    reward = 20.0 * (current_dist - next_dist)
    #    靠近目标 → 正，远离 → 负，悬停/振荡 → 净≈0
    #    总 episode 贡献 = 20 * (initial_dist - final_dist)，完全路径无关
    #    比 iter_6 的 15 提升到 20，使接近目标的收益更显著
    # ================================================================
    dist_improvement = current_dist - next_dist
    approach_reward = 20.0 * dist_improvement

    # ================================================================
    # 2. close_proximity — 极近距离微弱梯度（仅 < 0.4 单位）
    #
    #    二次衰减：dist=0 → 0.15, dist=0.2 → 0.0375, dist=0.4 → 0
    #    即使 agent 在 dist=0.2 悬停 1000 步也只有 ~37 点——远小于
    #    着陆后 contact_reward 可达的 16-46 每步。
    #    唯一作用：在目标垫板附近提供额外的细粒度梯度。
    #    关键区别 vs iter_6 proximity_warmth：阈值为 0.4 而非 3.0，不会创建租金区。
    # ================================================================
    if next_dist < 0.4:
        close_proximity = 0.15 * (1.0 - next_dist / 0.4) ** 2
    else:
        close_proximity = 0.0

    # ================================================================
    # 3. velocity_penalty — 距离门控速度抑制
    #
    #    gate = 1/(1+2*dist)：dist=0→1.0, dist=1→0.33, dist=3→0.14
    #    远处几乎不干预，近处温和引导减速。
    #    系数 0.8 比 iter_6 的 0.3 略强，但仍在合理范围。
    # ================================================================
    vel_gate = 1.0 / (1.0 + 2.0 * next_dist)
    velocity_penalty = -0.8 * vel_gate * (nvx**2 + nvy**2)

    # ================================================================
    # 4. angle_penalty — 距离门控姿态约束
    #
    #    与 velocity_penalty 使用相同的门控逻辑。
    # ================================================================
    angle_gate = 1.0 / (1.0 + 2.0 * next_dist)
    angle_penalty = -angle_gate * (1.5 * n_angle**2 + 0.8 * n_ang_vel**2)

    # ================================================================
    # 5. contact_reward — 加法独立着陆奖励（核心创新）
    #
    #    关键设计决策：
    #    - 不再是 iter_6 的乘法形式（四个因子乘积，任一为零则全为零）
    #    - 每条支撑腿独立给予奖励，降低发现门槛
    #    - 条件适度宽松（dist<0.3, speed<0.6），确保 agent 在接近目标时
    #      有较高概率偶然获得首次接触奖励
    #    - 双腿同时接触 + 完美着陆有额外叠加奖励
    #
    #    奖励结构：
    #    - 左腿接触: +8
    #    - 右腿接触: +8
    #    - 双腿同时: 额外 +15  (总计 8+8+15=31)
    #    - 完美着陆: 额外 +30  (双腿+极低速+极正姿态, 总计 8+8+15+30=61)
    #
    #    iter_4 证明二元门控着陆信号可达（active 0.8%, ep_sum=10.5），
    #    本设计的加法独立性应进一步提升可发现性。
    # ================================================================
    contact_reward = 0.0
    near_pad = next_dist < 0.3
    slow_approach = next_speed < 0.6

    if near_pad and slow_approach:
        # 每条腿独立奖励
        if n_left > 0.5:
            contact_reward += 8.0
        if n_right > 0.5:
            contact_reward += 8.0
        # 双腿稳定接触额外奖励
        if n_left > 0.5 and n_right > 0.5:
            contact_reward += 15.0
        # 完美着陆大奖：双腿 + 极低速 + 姿态端正
        if (n_left > 0.5 and n_right > 0.5 and
                next_speed < 0.15 and abs(n_angle) < 0.15):
            contact_reward += 30.0

    # ================================================================
    # 6. fuel_penalty — 轻量引擎使用惩罚
    # ================================================================
    fuel_penalty = -0.03 * float(action != 0)

    # ================================================================
    # 7. boundary_penalty — 水平越界安全网
    #
    #    阈值 1.3，二次惩罚在越界时急剧增大。
    #    iter_6 中 boundary_penalty 全程为 0（agent 学会了不越界），
    #    保留作为安全网但不调整。
    # ================================================================
    safe_x_bound = 1.3
    if abs(nx) > safe_x_bound:
        boundary_penalty = -3.0 * (abs(nx) - safe_x_bound) ** 2
    else:
        boundary_penalty = 0.0

    # ================================================================
    # 汇总
    # ================================================================
    total_reward = (
        approach_reward +
        close_proximity +
        velocity_penalty +
        angle_penalty +
        contact_reward +
        fuel_penalty +
        boundary_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "close_proximity": close_proximity,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "boundary_penalty": boundary_penalty,
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