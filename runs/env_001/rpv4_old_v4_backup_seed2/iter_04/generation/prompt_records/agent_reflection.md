# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的调查指引，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。快速掌握全貌。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。动手前查询以避免重复失败的修改。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。对比代码差异时使用。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。

# 输出

输出两个部分：

**1. 诊断报告**（在代码之前）：

- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的失败行为
- `signal_completeness`：必要职责是否完备，缺什么信号
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修改应改善策略
- `expected_next_round`：下一轮哪些指标应如何变化
- `main_risk`：最可能引入的新问题

**2. 完整 Python 代码**（```python ... ``` 代码块）。

```

## User Prompt

```markdown
第 3 轮迭代训练已经结束，得分 -222.850054（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(3)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

3. **查全局历史避免打转**——用 `read_memory()` 看所有迭代。**关键判断：你是不是在同一个语义功能上反复修改但无突破？**（注意：组件名字可能变了，但语义功能相同——比如 success_landing_bonus 和 success_bonus 都是"着陆成功信号"。）如果某个语义已经改了 2-3 轮都不好，不要再改它了——换一个语义方向。

4. **参考 best code 但要验证**——用 `read_reward_code(best_iter)` 看历史最佳奖励函数的代码。它的哪些组件可能促成了当时的最高分？但不要盲目照搬——结合 best 那轮的 feedback（`read_training_feedback(best_iter)`）看：它的得分分布、终止率、组件 activation 是否健康。best 只是记录最高分，不一定代表正确的引导方式。

5. **Checkpoint 是训练过程的证据**——如果某个组件在 checkpoint 40%→80% 期间 ep_sum 持续为 0 或不变，说明它没有提供有效梯度，agent 在训练中忽略了它。

### 修改原则

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态（稀疏→连续、状态值→改善量、均匀→高度感知等）。适用于组件的引导逻辑有问题（如均匀惩罚导致无近地减速梯度）。
- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。
- **⚠️ 组件被 exploit 时必须干预**：如果一个组件 magnitude_share > 90%，即使它刚被激活，也必须修改它——加触发条件限制、加每 episode 次数上限、或收紧阈值。"刚激活所以只做 L1"不适用于 exploit 场景。
- **避免打转**：如果 `read_past_reflection` 显示你之前已经改过同一语义且结果不好，换方向。

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v4/seed_2/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_3, score=-222.850054)
# Training Feedback

## Final-policy outcome
score=-222.850054, len=209.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-336.317345, -168.881892]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -0.735491 | -46.9% | 46.9% | 14.8% |
| progress | -0.298984 | -19.1% | 29.6% | 100.0% |
| angular_vel_penalty | -0.367962 | -23.5% | 23.5% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
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
| -758.2 | -498.0 | -368.5 | -318.5 | -272.5 | -239.4 | -218.5 | -228.0 | -226.4 | -228.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angular_vel_penalty | -1.200 | -1.088 | -0.732 | -0.577 | -0.431 | -0.387 | -0.400 | -0.413 | -0.394 | -0.394 |
| landing | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| progress | -3.205 | -1.672 | -1.096 | -0.849 | -0.611 | -0.513 | -0.373 | -0.284 | -0.300 | -0.267 |
| speed_penalty | -0.577 | -0.584 | -0.635 | -1.348 | -0.671 | -0.697 | -0.745 | -0.771 | -0.753 | -0.762 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angular_vel_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 21.2% | 17.5% | 19.4% | 24.1% | 18.5% | 14.9% | 15.6% | 14.9% | 15.9% | 14.6% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：contact-gated + quality-scaled
    #    仅在脚接触目标垫时激活，杜绝悬停 exploit
    contact = (next_left_contact + next_right_contact) / 2.0  # 0 / 0.5 / 1.0

    if contact > 0.0:
        # 各项着陆质量因子（每个在 [0,1]）
        pos_quality = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 0.5)
        vel_quality = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 1.0)
        angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)

        # contact 作为放大器：单脚=0.5x，双脚=1.0x
        # 任何质量因子的改善都有连续梯度
        landing_score = contact * (pos_quality + vel_quality + angle_quality) / 3.0
    else:
        landing_score = 0.0

    w_land = 5.0  # 提升权重补偿稀疏性
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```

# Subagent 数据报告
# 训练数据报告（iter=3）

## 1. 行为事实

- **得分**：score=-222.85（本轮），range=[-336.32, -168.88]
- **终止方式**：terminated 20/20（全部因碰撞/失败终止），truncated 0/20
- **回合长度**：len=209.60
- **提前终止**：early_terminal 7/20（<150步 且 score<-50）
- **跨轮比较**：iter_01 score=-203.53, len=185.8 → iter_02 score=-37.03, len=962.55 → iter_03 score=-222.85, len=209.60

### Checkpoint 官方得分轨迹（iter_03）

| cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---|---|---|---|---|---|---|---|---|
| -758.2 | -498.0 | -368.5 | -318.5 | -272.5 | -239.4 | -218.5 | -228.0 | -226.4 | -228.6 |

趋势：cp10%→cp70% 持续上升（-758.2→-218.5，+539.7），cp70%→cp100% 在 -218~-229 之间持平/轻微回落。

---

## 2. 骨架演进表

| iter | score | len | term | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | progress_ep_sum | progress_active | landing_ep_sum | landing_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -203.53 | 185.80 | 20/0 | -0.669 | 15.5% | -0.434 | 100.0% | -0.200 | 100.0% | 0.000 | 0.0% |
| 2 | -37.03 | 962.55 | 4/16 | -42.130 | 100.0% | -2.457 | 100.0% | 0.665 | 100.0% | 697.436 | 99.6% |
| 3 | -222.85 | 209.60 | 20/0 | -0.735 | 14.8% | -0.368 | 100.0% | -0.299 | 100.0% | 0.000 | 0.0% |

---

## 3. 数字标记

### 死组件
- **landing**：active_rate=0.0%（全线为零，cp1→cp100 均为 0.0%）

### 量级对比（每步均值 = ep_sum_mean ÷ len）

| 组件 | 每步均值 | 量级排序 |
|---|---|---|
| speed_penalty | -0.00351 | 最大 |
| angular_vel_penalty | -0.00176 | 第二 |
| progress | -0.00143 | 第三 |
| landing | 0.00000 | 最小（零） |

### 修改变化（iter_02 → iter_03 Δ）

| 组件 | Δ ep_sum | Δ active_rate |
|---|---|---|
| landing | 697.436 → 0.000 (−697.44) | 99.6% → 0.0% (−99.6pp) |
| speed_penalty | −42.130 → −0.735 (+41.39) | 100.0% → 14.8% (−85.2pp) |
| angular_vel_penalty | −2.457 → −0.368 (+2.09) | 100.0% → 100.0% (0pp) |
| progress | 0.665 → −0.299 (−0.96) | 100.0% → 100.0% (0pp) |

---

## 4. 语义演化（跨轮组件功能追踪）

### 着陆/接触信号

| iter | 组件名 | 门控方式 | 因子组合 | 得分 | 活性 | 备注 |
|---|---:|---|---|---|---|---|
| 1 | landing | 双脚contact乘积(0/1) | 乘积式 pos_x×pos_y×vel×angle | -203.53 | 0.0% 死组件 | 稀疏bonus，无梯度 |
| 2 | landing | proximity门控(距离<1.5) | sum-of-factors均值 | -37.03 | 99.6% | 连续奖励，接触作为soft bonus |
| 3 | landing | contact门控(>0) | sum-of-factors均值，contact作放大器 | -222.85 | 0.0% 死组件 | 重回接触门控，活性归零 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆信号 | v1_landing → v2_landing → v3_landing | -203.53 → -37.03 → -222.85 | 乘积→sum-of-factors：1次，proximity门控↔contact门控：1次（v2→v3），合计2次切换 |

### 接近/进度信号

| iter | 组件名 | 数学形式 | 得分 | 活性 |
|---|---:|---|---|---:|
| 1 | progress | Δ距离(曼哈顿) | -203.53 | 100.0% |
| 2 | progress | Δ距离(曼哈顿) | -37.03 | 100.0% |
| 3 | progress | Δ距离(曼哈顿) | -222.85 | 100.0% |

三轮均为同一数学形式(delta距离)，无设计切换。

### 稳定性/约束信号

| 子功能 | 各轮组件名 | 各轮得分 | 设计演变 |
|---|---:|---|---|
| 角速度惩罚 | v1/v2/v3 angular_vel_penalty | -203.53/-37.03/-222.85 | w=0.1常开，全高度作用，三轮未变 |
| 速度惩罚 | v1/v2/v3 speed_penalty | -203.53/-37.03/-222.85 | w=0.5，closeness门控(<1.5)，三轮未变 |

稳定性信号三轮设计一致，无方向切换。

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：proximity门控 + sum-of-factors，提供连续梯度
    # 激活条件：进入目标附近（曼哈顿距离 < 1.5）
    proximity = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 1.5)

    if proximity > 0.0:
        # 各项着陆质量因子（放松阈值，每个在 [0,1]）
        pos_x_factor = max(0.0, 1.0 - abs(next_x) / 0.5)
        pos_y_factor = max(0.0, 1.0 - abs(next_y) / 0.5)
        vel_factor = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 1.0)
        angle_factor = max(0.0, 1.0 - abs(next_angle) / 0.5)
        # 接触：0/0.5/1.0，作为 soft bonus
        contact_score = (next_left_contact + next_right_contact) / 2.0

        # sum-of-factors 取平均：任何方面的改善都有梯度
        landing_score = proximity * (
            pos_x_factor + pos_y_factor + vel_factor + angle_factor + contact_score
        ) / 5.0
    else:
        landing_score = 0.0

    w_land = 3.0
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
主体是一个2D类车辆刚体，从视口上方中央附近出发，带有随机初始作用力。任务是**尽快到达并稳定停靠在中央的目标垫上，同时尽可能减少发动机推力消耗**。理想行为是：平滑接近目标、降低线速度、保持垂直姿态（零角度）、以低速和正确朝向使左右两个支撑点同时接触目标垫并固化。次要目标包含节省燃料和/或快速完成，但不应混淆为主目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (默认) 或 float32
- obs[0]: x_position — 相对目标垫的水平坐标 (reward_usable: true)
- obs[1]: y_position — 相对目标垫垫面高度的垂直坐标 (reward_usable: true)
- obs[2]: x_velocity — 水平线速度 (reward_usable: true)
- obs[3]: y_velocity — 垂直线速度 (reward_usable: true)
- obs[4]: body_angle — 车身朝向角 (reward_usable: true)
- obs[5]: angular_velocity — 角速度 (reward_usable: true)
- obs[6]: left_support_contact — 左支撑脚接触标志，接触为1.0 (reward_usable: true)
- obs[7]: right_support_contact — 右支撑脚接触标志，接触为1.0 (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，靠惯性滑行
- action 1: left_orientation_engine — 点燃左姿态发动机（提供转向力矩）
- action 2: main_engine — 点燃主发动机（提供向上/向前的推力）
- action 3: right_orientation_engine — 点燃右姿态发动机（提供相反力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact: 主体发生碰撞或非预期的身体接触（很可能为失败，但不排除目标垫接触也属于body_contact，需结合后续稳定信号区分）
- horizontal_position_outside_viewport: 水平位置超出视口边界（明确失败）
- body_not_awake_or_settled: 主体不再活动或已稳定（speed/angular接近零且可能接触垫）；此条件可能是成功（settled on pad）或失败（settled elsewhere or falling asleep）
- 没有显式截断（truncation）条件提及，但实际环境可能有最大步数限制，目前未在信息中给出。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info为空字典)
- explicit_failure_flag_available: false
- allowed_info_fields: （当前step返回的info = {}，无可用字段）
- forbidden_or_uncertain_info_fields: 无任何显式success/failure/termination_reason字段；不可使用info进行reward或终止判定

## 7. 可用于奖励函数的信号
- position: x_position, y_position (相对目标垫)  
- velocity: x_velocity, y_velocity  
- orientation: body_angle  
- angular_velocity: angular_velocity  
- contact: left_support_contact, right_support_contact (二值浮点，1.0表示接触)  
- action/engine: action index (可用于动作惩罚，或结合下一步状态推断推力效果)
- other: 无额外传感器
```
