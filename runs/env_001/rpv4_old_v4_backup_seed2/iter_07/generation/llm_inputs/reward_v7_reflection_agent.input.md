第 6 轮迭代训练已经结束，得分 -13.573963（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(6)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

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

# 上一轮训练反馈 (iter_6, score=-13.573963)
# Training Feedback

## Final-policy outcome
score=-13.573963, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-45.504561, 26.677289]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_penalty | -9.888500 | -53.2% | 53.2% | 98.9% |
| distance_progress | 6.331350 | 34.1% | 39.1% | 100.0% |
| stability | -1.416178 | -7.6% | 7.6% | 100.0% |
| landing_completion | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
| -5.3 | 25.9 | 10.9 | -54.7 | -15.2 | -4.2 | 38.6 | 164.0 | -5.1 | 106.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| distance_progress | 3.498 | 4.736 | 3.873 | 3.944 | 5.660 | 6.192 | 6.509 | 6.921 | 6.423 | 6.506 |
| fuel_penalty | -4.920 | -4.725 | -6.202 | -8.069 | -6.194 | -6.324 | -7.482 | -4.815 | -7.812 | -7.504 |
| landing_completion | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 91.914 | 273.103 | 0.000 | 43.417 |
| stability | -2.068 | -1.708 | -1.429 | -0.514 | -1.107 | -1.388 | -1.751 | -1.252 | -1.185 | -1.470 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| distance_progress | 100.0% | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 98.7% | 100.0% | 98.9% |
| fuel_penalty | 94.9% | 70.7% | 76.7% | 80.7% | 61.9% | 63.2% | 78.1% | 62.0% | 78.1% | 91.7% |
| landing_completion | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.6% | 18.1% | 0.0% | 3.1% |
| stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar lander-like navigation task.
    Components:
      A. distance_progress   – reduction in distance to landing pad
      B. stability           – penalise high velocity and large attitude when close to pad
      C. landing_completion  – soft success bonus for two-leg contact with low speed
      D. fuel_penalty        – tiny penalty for using any engine
    """
    # --- unpack observations ---
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos = next_obs[0]
    ny_pos = next_obs[1]

    # --- distances ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    velocity_norm = (x_vel**2 + y_vel**2) ** 0.5

    # =====================================================================
    # A. distance_progress: reward reduction in Euclidean distance to target
    # =====================================================================
    progress = (current_distance - next_distance) * 5.0   # scaling factor

    # =====================================================================
    # B. stability: smooth penalty for speed and attitude when near the pad
    #    gate: exp(-2 * distance)  →  close to 1 when distance small
    # =====================================================================
    gate = 2.718281828 ** (-2.0 * current_distance)     # approximate exp(-2d)
    w_vel = 0.1
    w_angle = 0.5
    w_angv = 0.5
    stability_penalty = - (w_vel  * velocity_norm**2 +
                           w_angle * angle**2 +
                           w_angv  * ang_vel**2) * gate

    # =====================================================================
    # C. landing_completion: soft success proxy – both legs on ground,
    #    near centre, with low velocity and stable attitude
    # =====================================================================
    both_feet = left_contact * right_contact   # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 2.0 * speed_quality * angle_quality * angvel_quality
    else:
        landing_reward = 0.0

    # =====================================================================
    # D. fuel_penalty: minuscule cost for any engine use
    # =====================================================================
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + stability_penalty + landing_reward + fuel_penalty

    components = {
        "distance_progress": progress,
        "stability": stability_penalty,
        "landing_completion": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# Subagent 数据报告
# 训练数据报告

## 1. 行为事实
- 本轮得分：score=-13.574，range=[-45.505, 26.677]
- 终止方式：terminated 0/20（无成功着陆），truncated 20/20（全部达到时间上限）
- 回合长度：len=1000.000（全部集数跑满 1000 步）
- 官方得分Checkpoint轨迹（每10%步数）：-5.3 → 25.9 → 10.9 → -54.7 → -15.2 → -4.2 → 38.6 → 164.0 → -5.1 → 106.7
  - 总体趋势：波动剧烈，cp80%达到最高点164.0，之后回落
- 组件活跃率：distance_progress 100.0%，fuel_penalty 98.9%，stability 100.0%，landing_completion 0.0%

## 2. 骨架演进表
| iter | score | len | term | progress/distance_progress ep_sum | progress/distance_progress active | speed_penalty/landing ep_sum | speed_penalty/landing active | angular_vel_penalty/stability ep_sum | angular_vel_penalty/stability active | landing/landing_completion/fuel_penalty ep_sum | landing/landing_completion/fuel_penalty active | velocity_guidance ep_sum | velocity_guidance active |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | -203.531 | 185.800 | 20/0 | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | 0.000 | 0.0% | — | — |
| 2 | -37.031 | 962.550 | 4/16 | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | 697.436 | 99.6% | — | — |
| 3 | -222.850 | 209.600 | 20/0 | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | 0.000 | 0.0% | — | — |
| 4 | -295.327 | 138.900 | 20/0 | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | — | — | -1.596 | 100.0% |
| 5 | -197.455 | 185.500 | 20/0 | -0.183 | 100.0% | -0.717 | 15.8% | -0.343 | 100.0% | 0.000 | 0.0% | — | — |
| 6 | -13.574 | 1000.000 | 0/20 | 6.331 | 100.0% | — | — | -1.416 | 100.0% | -9.889(fuel) / 0.000(landing) | 98.9%(fuel) / 0.0%(landing) | — | — |

注：iter 1-5 组件名为 progress, speed_penalty, angular_vel_penalty, landing, velocity_guidance；iter 6 重新命名为 distance_progress, stability, landing_completion, fuel_penalty。表中按语义列对齐，同名显示为合并列。

## 3. 数字标记

### 死组件
- **landing_completion**：active_rate=0.0%（<2%），ep_sum_mean=0.000，在所有20集评估中从未激活

### 全覆盖惩罚
- 无。fuel_penalty active_rate=98.9%（>95%）但 signed_share 为负（-53.2%），检查：active_rate 98.9% > 95%，signed_share -53.2% < 0 → **fuel_penalty 全覆盖惩罚**（每步几乎都触发，且全部为负贡献）

### 主导
- **fuel_penalty**：magnitude_share=53.2%（>80%？否，53.2%不满足>80%条件）
- 无组件 magnitude_share > 80%

### 量级对比（每步均值 ep_sum ÷ len）
- 最大两个：
  - distance_progress：6.331 ÷ 1000 = 0.00633
  - stability：|-1.416| ÷ 1000 = 0.00142 → 实际第二大为 fuel_penalty：|-9.889| ÷ 1000 = 0.00989
  - 修正：fuel_penalty 每步均值 0.00989，distance_progress 每步均值 0.00633，stability 每步均值 0.00142，landing_completion 每步均值 0.00000
  - 最大两个：fuel_penalty (0.00989), distance_progress (0.00633)
  - 最小两个：landing_completion (0.00000), stability (0.00142)

### 修改变化（iter 5 → iter 6，按语义对应）
本报告不做组件映射推断，仅提供原始数据对比。iter 6 重命名了全部四个组件，无法建立一一对应关系。

## 4. 语义演化（跨轮组件功能追踪）

### 接近/进度信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 进度信号 | v1_progress → v2_progress → v3_progress → v4_progress → v5_progress → v6_distance_progress | -0.200 → 0.665 → -0.299 → -0.026 → -0.183 → 6.331 | 缩放因子×5：iter 6 引入（旧版均为×1.0） |

- v1-v5：progress = old_dist - new_dist（状态值，×1.0），始终在 [-0.3, 0.7] 范围内
- v6：distance_progress = (old_dist - new_dist) × 5.0（状态值，×5.0），ep_sum 从 ~0.7 跳升至 6.331
- 方向切换：无（始终为状态值/改善量形式），但缩放因子从 1.0 → 5.0

### 着陆/接触信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆信号 | v1_landing(死) → v2_landing(主导) → v3_landing(死) → v5_landing(死) → v6_landing_completion(死) | 0.000 → 697.436 → 0.000 → 0.000 → 0.000 | contact-gated↔proximity-gated：4次切换 |
| 燃料成本 | v6_fuel_penalty | -9.889 | 新组件（iter 6 首次引入） |

着陆信号详细演变：
- v1：contact-gated + quality factors，权重2.0 → 死组件（active_rate 0.0%）
- v2：proximity-gated + sum-of-factors + contact_score，权重3.0 → 主导组件（ep_sum 697.4，magnitude_share 93.8%），agent 学会 exploit 悬停
- v3：contact-gated + quality factors（无 proximity 门控），权重5.0 → 死组件（active_rate 0.0%）
- v4：移除 landing，替换为 velocity_guidance
- v5：proximity-gated + contact_multiplier（1x/3x/5x），权重0.5 → 死组件（active_rate 0.0%）
- v6：contact-gated（both_feet > 0.5）+ 条件性乘积奖励，权重2.0 → 死组件（active_rate 0.0%）

方向切换（不同设计范式之间的切换）：
- 稀疏bonus（contact-gated）↔ 连续奖励（proximity-gated）：v2（连续proximity）→ v3（稀疏contact）→ v5（连续proximity）→ v6（稀疏contact）= **3次切换**
- v1→v2：从 contact-gated 切换到 proximity-gated（第1次）
- v2→v3：从 proximity-gated 切换到 contact-gated（第2次）
- v3→v5：从 contact-gated 切换到 proximity-gated（第3次）
- v5→v6：从 proximity-gated 切换到 contact-gated（第4次）
- 总计：4次切换

### 稳定性/约束信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 速度惩罚 | v1_speed_penalty → v2_speed_penalty → v3_speed_penalty → v4_speed_penalty → v5_speed_penalty | -0.669 → -42.130 → -0.735 → -2.097 → -0.717 | 门控位置：3次切换（closeness-on-new_dist → closeness-on-new_dist → closeness-on-new_dist → closeness-on-new_dist → closeness-on-new_dist，无切换） |
| 角速度惩罚 | v1_angular_vel_penalty → v2_angular_vel_penalty → v3_angular_vel_penalty → v4_angular_vel_penalty → v5_angular_vel_penalty | -0.434 → -2.457 → -0.368 → -1.266 → -0.343 | 始终为全高度 |w·abs(ang_vel)| |
| 综合稳定性 | v6_stability | -1.416 | 门控：exp(-2d)，形式从线性变为二次+门控 |

v1-v5：速度惩罚 + 角速度惩罚 分离为两个组件
- speed_penalty：-w_speed × closeness × speed，closeness = max(0, 1 - new_dist/1.5)
- angular_vel_penalty：-w_angvel × |next_ang_vel|，始终激活

v6：合并为 stability，形式变为 -(w_vel×velocity² + w_angle×angle² + w_angv×ang_vel²) × exp(-2d)
- 从基于 next_state 切换到基于当前 state（obs而非next_obs）
- 惩罚从线性转为二次项
- 门控从分段线性（closeness）切换到指数衰减（exp(-2d)）
- 方向切换：速度惩罚从"线性门控"切换到"二次指数门控" = 1次切换

### 完成/成功代理
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 导航引导 | v4_velocity_guidance | -1.596 | 仅存在于 iter 4 |
| 着陆完成 | v6_landing_completion | 0.000 | 条件性乘积形式，仅激活于 both_feet 接触 |

- v4 velocity_guidance：奖励速度方向与目标方向的对齐度（余弦相似度 × 距离衰减），ep_sum=-1.596（agent 未能学会利用）
- v6 landing_completion：contact-gated 完成奖金（both_feet + close_to_centre），乘积形式（speed_quality × angle_quality × angvel_quality），ep_sum=0.000（全程未触发）
- 设计从"连续导航引导"切换到"稀疏完成奖金" = 1次切换

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