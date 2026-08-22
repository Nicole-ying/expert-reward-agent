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

# 上一轮训练反馈 (iter_8, score=-45.740324)
# Training Feedback

## Final-policy outcome
score=-45.740324, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.464295, -4.664443]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 16.733679 | 28.0% | 42.0% | 100.0% |
| fuel_penalty | -19.617000 | -32.8% | 32.8% | 65.4% |
| proximity_warmth | 10.573632 | 17.7% | 17.7% | 100.0% |
| velocity_penalty | -3.238598 | -5.4% | 5.4% | 100.0% |
| angle_penalty | -1.277877 | -2.1% | 2.1% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_readiness | 0.000000 | 0.0% | 0.0% | 0.0% |
| perfect_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -96.0 | -105.7 | -46.2 | -9.4 | -39.3 | -41.3 | -28.7 | -5.8 | -24.4 | -36.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | -3.602 | -0.946 | -0.969 | -1.410 | -1.598 | -1.623 | -1.756 | -1.447 | -2.447 | -1.363 |
| approach_reward | 21.658 | 11.235 | 16.801 | 23.455 | 18.313 | 17.344 | 23.483 | 24.702 | 21.057 | 18.384 |
| boundary_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| contact_reward | 4.007 | 0.000 | 0.000 | 14.012 | 2.019 | 2.143 | 3.630 | 0.000 | 12.405 | 0.000 |
| fuel_penalty | -0.417 | -16.138 | -18.765 | -19.833 | -19.721 | -19.407 | -20.061 | -19.243 | -18.307 | -19.992 |
| landing_readiness | 0.004 | 0.000 | 0.000 | 3.516 | 0.358 | 0.162 | 2.166 | 4.587 | 1.222 | 0.000 |
| perfect_landing | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| proximity_warmth | 0.558 | 8.804 | 10.712 | 11.718 | 10.679 | 9.987 | 11.744 | 12.522 | 11.023 | 10.994 |
| velocity_penalty | -16.219 | -4.032 | -2.816 | -3.555 | -3.589 | -3.863 | -3.478 | -3.559 | -3.991 | -3.248 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angle_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| approach_reward | 100.0% | 100.0% | 100.0% | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |
| boundary_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| contact_reward | 3.1% | 0.0% | 0.0% | 0.6% | 0.2% | 0.3% | 0.3% | 0.0% | 0.8% | 0.0% |
| fuel_penalty | 19.5% | 63.6% | 62.5% | 69.4% | 68.5% | 71.4% | 69.4% | 64.1% | 66.3% | 66.6% |
| landing_readiness | 0.2% | 0.0% | 0.0% | 26.9% | 7.7% | 5.2% | 24.4% | 37.5% | 14.9% | 0.0% |
| perfect_landing | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| proximity_warmth | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.8% | 100.0% | 100.0% | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化 v8 — 漏斗式着陆梯度。

    核心创新：消除「租金陷阱 + 着陆信号不可达」的矛盾。
    
    设计原则：
    - 不存在可被 exploit 的中等距离租金源（iter_3/iter_6 的教训）
    - 着陆奖励从粗到精连续可微分（iter_6/iter_7 的教训：门控太窄）
    - 微小 proximity 仅维持靠近目标的动力，但量级远不足以支撑悬停策略
    - 「着陆就绪」预奖励桥接 proximity 和 contact 之间的梯度断层
    
    梯度漏斗结构：
      远距 → approach_reward (delta, 路径无关, 不可 exploit)
      中距 → 极微 proximity_warmth (仅方向指引, ~0.01/step)
      近距 → landing_readiness (dist<0.2, slow, upright → ~0.05/step)
      接触 → contact_reward (additive, 宽松初始条件 → ~3-8/step)
      完美 → perfect_landing (双腿+极低速+正姿 → +15/step 叠加)
      
    关键比率：contact_reward / proximity_warmth ≈ 400-1200x
    """

    # ================================================================
    # 信号提取
    # ================================================================
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # 派生量
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ================================================================
    # 1. approach_reward — 路径无关 delta 主干（不可 exploit）
    #
    #    总 episode 贡献 = 20 * (initial_dist - final_dist)，完全路径无关。
    #    靠近目标 → 正，远离 → 负，悬停振荡 → 净≈0。
    #    从 iter_6 的 15 提升到 20，使靠近目标的收益更显著。
    # ================================================================
    dist_improvement = current_dist - next_dist
    approach_reward = 20.0 * dist_improvement

    # ================================================================
    # 2. proximity_warmth — 极微方向指引（不可 exploit）
    #
    #    从 iter_6 的 0.05*(1-dist/3) 大幅削减：
    #    - 系数：0.05 → 0.015（削减 70%）
    #    - 范围：3.0 → 2.0（收缩三分之一）
    #    在 dist=0.5 处：0.015*0.75=0.011/step → 1000 步 ≈ 11 点
    #    vs iter_6 在 dist=0.5 处：0.05*0.833=0.042/step → 1000 步 ≈ 42 点
    #    仅够提供"朝目标走"的微弱动机，不可能成为租金来源。
    # ================================================================
    proximity_warmth = 0.015 * max(0.0, 1.0 - next_dist / 2.0)

    # ================================================================
    # 3. landing_readiness — 「着陆就绪」预奖励
    #
    #    解决 iter_6/iter_7 中 "proximity→contact 梯度断层" 问题。
    #    当 agent 进入极近距（dist<0.2）、低速（speed<0.35）、姿态不极端时，
    #    给予微小正向信号，引导 agent 进入"预备着陆"姿态。
    #    这个奖励非常小（max ~0.06/step），但创建了从"靠近"到"接触"
    #    之间的连续梯度，使 agent 不会在 proximity 区域停滞。
    # ================================================================
    landing_readiness = 0.0
    if next_dist < 0.2 and next_speed < 0.35:
        d_factor = 1.0 - next_dist / 0.2       # 1 at dist=0
        s_factor = 1.0 - next_speed / 0.35     # 1 at speed=0
        a_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)  # 1 at angle=0
        landing_readiness = 0.06 * d_factor * s_factor * a_factor
        # 典型值 dist=0.1, speed=0.15, angle=0.1:
        #   0.06 * 0.5 * 0.571 * 0.667 = 0.011/step — 极微，纯梯度信号

    # ================================================================
    # 4. contact_reward — 加法独立着陆奖励（核心创新）
    #
    #    对比 iter_7 contact_reward 的关键改进：
    #    (a) 放宽距离门控：0.5 vs 0.3
    #    (b) 移除速度硬门控：不再要求 speed<0.6，改为质量乘数
    #    (c) 质量因子作为乘数而非门控：保留 30% 基础值，高速粗接触也有信号
    #    (d) 每条腿独立奖励 + 叠加
    #
    #    设计保证：即使在 dist=0.4, speed=0.8（高速俯冲）, angle=0.2 时，
    #    单腿接触仍能获得：1*6.0*0.2*(0.3+0.7*0*0.5) = 0.36/step
    #    虽小但非零——agent 能感知到"接触是好事"。
    #    随着条件改善，奖励平滑增长到 ~8-15/step。
    # ================================================================
    contact_reward = 0.0
    if next_dist < 0.5:
        d_quality = 1.0 - next_dist / 0.5            # 1 at dist=0
        s_quality = max(0.0, 1.0 - next_speed / 0.6) # 1 at speed=0
        a_quality = max(0.0, 1.0 - abs(n_angle) / 0.4)  # 1 at angle=0
        
        contact_count = n_left + n_right  # 0.0, 1.0, or 2.0
        
        if contact_count > 0:
            # 每腿基础奖励 × 质量乘数（最低保底 30%）
            quality_mult = 0.3 + 0.7 * s_quality * a_quality
            contact_reward = contact_count * 6.0 * d_quality * quality_mult
            
            # 双腿稳定接触叠加奖励
            if contact_count >= 2.0:
                contact_reward += 8.0 * d_quality * s_quality * a_quality

    # ================================================================
    # 5. perfect_landing — 完美着陆超级奖励
    #
    #    当所有条件完美满足时叠加：双腿接触 + 极低速 + 极正姿态 + 极近。
    #    这是一个"锦上添花"的信号，帮助 agent 从"粗糙着陆"
    #    精炼到"完美软着陆"。
    # ================================================================
    perfect_landing = 0.0
    if (n_left > 0.5 and n_right > 0.5 
            and next_speed < 0.08 
            and abs(n_angle) < 0.08 
            and next_dist < 0.1):
        perfect_landing = 15.0

    # ================================================================
    # 6. velocity_penalty — 距离门控速度抑制
    #
    #    gate = 1/(1+3*dist)：dist=0→1.0, dist=0.5→0.4, dist=2→0.14
    #    远处几乎不干预，近处温和引导减速。
    #    系数 0.5 适度，不会像 iter_7 的 0.8 那样强势。
    # ================================================================
    vel_gate = 1.0 / (1.0 + 3.0 * next_dist)
    velocity_penalty = -0.5 * vel_gate * (nvx**2 + nvy**2)

    # ================================================================
    # 7. angle_penalty — 距离门控姿态约束
    #
    #    与 velocity_penalty 使用相同的门控逻辑。
    #    系数温和，避免阻碍中距离机动。
    # ================================================================
    angle_gate = 1.0 / (1.0 + 3.0 * next_dist)
    angle_penalty = -angle_gate * (1.0 * n_angle**2 + 0.5 * n_ang_vel**2)

    # ================================================================
    # 8. fuel_penalty — 引擎使用惩罚
    #
    #    保持 0.03 不变，与 iter_6/iter_7 一致。
    #    足够轻量以不阻碍必要探索。
    # ================================================================
    fuel_penalty = -0.03 * float(action != 0)

    # ================================================================
    # 9. boundary_penalty — 水平越界安全网
    #
    #    保持 iter_7 的参数不变（阈值 1.3，二次惩罚）。
    #    历史数据显示 boundary_penalty 始终为 0，agent 已经学会不越界。
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
        proximity_warmth +
        landing_readiness +
        contact_reward +
        perfect_landing +
        velocity_penalty +
        angle_penalty +
        fuel_penalty +
        boundary_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "proximity_warmth": proximity_warmth,
        "landing_readiness": landing_readiness,
        "contact_reward": contact_reward,
        "perfect_landing": perfect_landing,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
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
```
