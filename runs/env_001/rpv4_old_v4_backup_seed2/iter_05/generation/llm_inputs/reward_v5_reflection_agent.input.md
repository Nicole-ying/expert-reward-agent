第 4 轮迭代训练已经结束，得分 -295.326995（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(4)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

# 上一轮训练反馈 (iter_4, score=-295.326995)
# Training Feedback

## Final-policy outcome
score=-295.326995, len=138.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-343.965185, -239.829467]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_guidance | -1.595910 | -18.9% | 53.9% | 100.0% |
| speed_penalty | -2.096809 | -24.9% | 24.9% | 63.2% |
| angular_vel_penalty | -1.265816 | -15.0% | 15.0% | 100.0% |
| progress | -0.026234 | -0.3% | 6.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
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
| -574.8 | -653.7 | -605.5 | -571.7 | -514.4 | -441.3 | -395.5 | -334.2 | -325.3 | -289.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angular_vel_penalty | -1.206 | -3.432 | -3.279 | -2.857 | -2.209 | -2.040 | -1.811 | -1.488 | -1.621 | -1.258 |
| progress | -2.055 | -0.647 | -0.438 | -0.383 | -0.417 | -0.275 | -0.210 | -0.130 | -0.064 | 0.008 |
| speed_penalty | -7.605 | -1.030 | -1.203 | -1.362 | -1.377 | -1.607 | -1.483 | -1.392 | -1.554 | -2.627 |
| velocity_guidance | -15.862 | -4.525 | -3.196 | -3.183 | -3.446 | -2.829 | -2.823 | -2.704 | -2.419 | -1.395 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| angular_vel_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 33.2% | 49.1% | 55.9% | 57.7% | 61.2% | 63.6% | 58.3% | 58.3% | 58.6% | 68.2% |
| velocity_guidance | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

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

    # 4. 速度方向引导（新组件，替代 landing）
    #    奖励「速度方向指向目标」，提供即时正向导航梯度
    #    距离衰减：远处需要导航引导，近处让减速信号接管
    if old_dist > 0.05:
        # 单位方向向量：从当前位置指向目标 (0,0)
        dir_x = -x / old_dist
        dir_y = -y / old_dist

        # 速度方向与目标方向的余弦相似度
        spd = (next_vx ** 2 + next_vy ** 2) ** 0.5 + 0.01
        alignment = (next_vx * dir_x + next_vy * dir_y) / spd  # [-1, 1]

        # 距离衰减权重：远处=1.0，距离0→0（线性衰减，5单位外满权重）
        distance_weight = min(1.0, old_dist / 5.0)

        w_guidance = 0.5
        guidance_reward = w_guidance * distance_weight * alignment
    else:
        guidance_reward = 0.0

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + guidance_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "velocity_guidance": guidance_reward
    }
    return float(total_reward), components
```

# Subagent 数据报告
# 训练数据记者报告 — iter_04

## 1. 行为事实
- **得分**：score = -295.33，range = [-343.97, -239.83]
- **终止方式**：terminated = 20/20（全部成功/失败终止），truncated = 0/20（无超时截断）
- **回合长度**：len = 138.9 步
- **Checkpoint 轨迹**（官方得分）：
  cp10% = -574.8 → cp20% = -653.7 → cp30% = -605.5 → cp40% = -571.7 → cp50% = -514.4 → cp60% = -441.3 → cp70% = -395.5 → cp80% = -334.2 → cp90% = -325.3 → cp100% = -289.6
  整体趋势：得分从 -574.8 连续提升至 -289.6（负值减小）。

## 2. 骨架演进表

| iter | score | len | term | progress_ep_sum | progress_active | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | landing_ep_sum | landing_active | velocity_guidance_ep_sum | velocity_guidance_active |
|------|-------|------|------|------------------|------------------|-----------------------|-----------------------|----------------------------|----------------------------|----------------|----------------|---------------------------|---------------------------|
| 1 | -203.53 | 185.80 | 20/20,0 | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | 0.000 | 0.0% | – | – |
| 2 | -37.03 | 962.55 | 4/20,16/20 | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | 697.436 | 99.6% | – | – |
| 3 | -222.85 | 209.60 | 20/20,0 | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | 0.000 | 0.0% | – | – |
| 4 | -295.33 | 138.90 | 20/20,0 | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | – | – | -1.596 | 100.0% |

## 3. 数字标记

- **主导**：无组件 magnitude_share > 80%。
- **死组件**：无组件 active_rate < 2%。
- **淹没**：无组件同时满足 active_rate > 90% 且 magnitude_share < 3%。
- **全覆盖惩罚**（active_rate > 95% 且 signed_share 为负）：
  - `velocity_guidance`：active=100.0%，signed_share=-18.9%
  - `angular_vel_penalty`：active=100.0%，signed_share=-15.0%
  - `progress`：active=100.0%，signed_share=-0.3%
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - 最大两个（负值中数值最大，即最不负面）：
    - `progress` ≈ -0.00019
    - `angular_vel_penalty` ≈ -0.00911
  - 最小两个（负值中数值最小，即最负面）：
    - `speed_penalty` ≈ -0.01509
    - `velocity_guidance` ≈ -0.01149
- **修改变化**（相比上一轮 iter_03）：
  - `progress`：Δ ep_sum = +0.273（从 -0.299 → -0.026），Δ active = +0 pp（100% → 100%）
  - `speed_penalty`：Δ ep_sum = -1.361（从 -0.735 → -2.097），Δ active = +48.4 pp（14.8% → 63.2%）
  - `angular_vel_penalty`：Δ ep_sum = -0.898（从 -0.368 → -1.266），Δ active = +0 pp（100% → 100%）
  - `landing`：移除（原 ep_sum=0.000，active=0.0%）
  - `velocity_guidance`：新增（ep_sum=-1.596，active=100.0%）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分（ep_sum） | 方向切换 |
|---------|----------|-----------------|---------|
| 接近/进度信号 | v1_progress, v2_progress, v3_progress, v4_progress + velocity_guidance | -0.200, 0.665, -0.299, progress=-0.026 / velocity_guidance=-1.596 | 仅增加方向引导组件，未在功能设计上切换 |
| 着陆/接触信号 | v1_landing, v2_landing, v3_landing, v4_(无) | 0.000, 697.436, 0.000, – | 稀疏bonus↔连续奖励：2次切换（v1乘积门控→v2 sum-of-factors连续；v2连续→v3 contact-gated稀疏） |
| 稳定性约束（速度惩罚） | v1_speed_penalty, v2_speed_penalty, v3_speed_penalty, v4_speed_penalty | -0.669, -42.130, -0.735, -2.097 | 无切换（始终为靠近减速惩罚） |
| 稳定性约束（角速度惩罚） | v1_angular_vel_penalty, v2_angular_vel_penalty, v3_angular_vel_penalty, v4_angular_vel_penalty | -0.434, -2.457, -0.368, -1.266 | 无切换（始终为恒定角速度惩罚） |

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