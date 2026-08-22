# 1. Search objective
- target_score: 200.000000
- current_score: 142.321177
- gap_to_target: 57.678823
- target_achievement_ratio: 71.161%

# 2. 上一轮奖励函数代码（该轮得分: 142.321177）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward modulated by quality factors
    """
    
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    # === Component A: goal_approach_progress (main learning signal) ===
    # Use potential-based shaping: reward improvement in distance to target
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance  # positive when approaching
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    # === Component B: velocity_brake_near_target (safety constraint) ===
    # Quadratic velocity penalty, scaled by proximity to encourage slow approach
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)  # approaches 1 when close to target
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    # === Component C: orientation_stability (safety constraint) ===
    # Quadratic penalty on angle deviation and angular velocity
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    # === Component D: contact_landing_reward (dense contact guidance) ===
    # Continuous reward for contact, modulated by landing quality factors
    # Base: any contact (left or right)
    contact_active = max(next_left_contact, next_right_contact)
    
    # Quality modulator 1: velocity (lower is better, bounded to [0,1])
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    
    # Quality modulator 2: orientation (smaller angle is better, bounded to [0,1])
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    
    # Quality modulator 3: proximity to target (closer is better, bounded to [0,1])
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    # Combined: contact reward scaled by quality factors
    # Using product to require joint satisfaction, but normalized to avoid collapse
    quality_factor = (velocity_quality * angle_quality * proximity_quality) ** 0.333
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 984.85 | -82.86 | ✅ |
| 2 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 721.10 | 142.32 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

## 1. 当前策略行为画像

Agent 已经学会了"接近目标 → 触地 → 维持稳定接触"的完整链路：score 从 iter1 的 -82.86 跃升到 142.32（+225），len 从 984.9 步压缩到 721.1 步，terminated 从 1/20 暴增到 16/20。这说明策略不再是漂浮拖时间，而是主动去触发着陆终止。score_range 上限 256.68、下限 -49.84，说明大部分回合走向成功着陆，但仍有部分回合失败（可能是坠毁或未触地漂浮，对应 4/20 truncated）。

## 2. 组件健康度

- contact_landing_reward: 主导（90.6% share），active_rate 14.6% 符合"完成类信号只在末段生效"的天然 profile，且有 velocity/angle/proximity 三重质量门控，从 score/len/terminated 的同步改善看，方向是对齐的，不是纯粹的 reward hacking 信号。
- velocity_brake_near_target: 健康。active 99%，per-step ≈ -0.036，是稳定的安全约束，量级适中，未被完全淹没。
- goal_approach_progress: 淹没风险。active 98.8%（覆盖率达标），但 per-step ≈ 0.017，是四个组件里最小的一个——比 velocity_brake 的 per-step 惩罚还小 2 倍，比 contact_landing_reward 的 per-active-step（≈4.15）低约 240 倍。这是典型的"指南针信号还在但音量几乎为零"，目前没有暴露问题是因为 contact 信号本身质量门控里包含了 proximity_quality，间接替代了引导作用；一旦后续调参改变 contact 的门控方式，goal_approach_progress 可能无法独立支撑早期接近阶段的引导。
- orientation_stability: 偏弱但非死信号。100% active，per-step ≈ -0.0087，量级最小,基本只起微弱的姿态约束作用,目前不是瓶颈。

## 3. 修改效果（N-1 → N）

注意：本轮反馈与上一轮反馈的 score、len、terminated、reward 代码完全一致，这两份数据看起来是同一版本的重复快照，不构成一次真实的迭代对比。真正的因果对比来自"逐组件修改效果"表，对应 memory 里 iter1 → iter2：

- 删除 terminal_soft_landing_bonus（仅 28.3% active，一次性稀疏奖励）→ 新增 contact_landing_reward（持续质量门控的接触奖励）
- score: -82.86 → 142.32（+225.19）
- len: 984.9 → 721.1（-263.8，回合明显缩短，说明不再漂浮拖时间）
- terminated: 1/20 → 16/20（+15，成功触发着陆终止的比例大幅提升）

结论：这次骨架级修改（把稀疏终局奖励换成密集质量门控接触奖励）是有效的，是本次分数跃升的主要驱动。

## 4. 历史模式

两轮记录都标记"new_best"且预判列均为 ✅，还没有出现连续失败模式，未触发"连续≥3轮预判错误→Level3重建"的警戒线。当前处于上升通道，尚不构成"历史模式"（样本量太少，仅2次迭代）。

## 5. 行为结论

达标且方向正确，但存在潜在隐患：goal_approach_progress 作为主要的密集引导信号，其 per-step 量级被 contact_landing_reward 压制约240倍，目前靠 contact 组件自带的 proximity_quality 间接兜底,没有暴露出问题，但这是一个脆弱的耦合，不是真正独立的双重保障。

最需要干预的组件：goal_approach_progress。它覆盖率达标（98.8%）却量级过小，是"淹没型"信号而非"死信号"，建议下一轮评估是否需要提高其系数，以确保它在接触发生前的飞行阶段能独立起到有效引导作用，而不是完全依赖 contact_landing_reward 里嵌套的 proximity_quality 来兜底。

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward modulated by quality factors
    """
    
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    # === Component A: goal_approach_progress (main learning signal) ===
    # Use potential-based shaping: reward improvement in distance to target
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance  # positive when approaching
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    # === Component B: velocity_brake_near_target (safety constraint) ===
    # Quadratic velocity penalty, scaled by proximity to encourage slow approach
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)  # approaches 1 when close to target
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    # === Component C: orientation_stability (safety constraint) ===
    # Quadratic penalty on angle deviation and angular velocity
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    # === Component D: contact_landing_reward (dense contact guidance) ===
    # Continuous reward for contact, modulated by landing quality factors
    # Base: any contact (left or right)
    contact_active = max(next_left_contact, next_right_contact)
    
    # Quality modulator 1: velocity (lower is better, bounded to [0,1])
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    
    # Quality modulator 2: orientation (smaller angle is better, bounded to [0,1])
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    
    # Quality modulator 3: proximity to target (closer is better, bounded to [0,1])
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    # Combined: contact reward scaled by quality factors
    # Using product to require joint satisfaction, but normalized to avoid collapse
    quality_factor = (velocity_quality * angle_quality * proximity_quality) ** 0.333
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| contact_landing_reward | — (新增) | 2.104 | 新增 | — | 50.8% | — |
| goal_approach_progress | 0.031 | 0.027 | -0.004 | 100.0% | 100.0% | +0.0pp |
| orientation_stability | -0.071 | -0.062 | +0.010 | 100.0% | 100.0% | +0.0pp |
| terminal_soft_landing_bonus | 14.147 | — (新增) | 已删除 | 28.3% | — | — |
| velocity_brake_near_target | -0.200 | -0.175 | +0.025 | 100.0% | 100.0% | +0.0pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -82.86 | 142.32 | +225.19 |
| 平均回合长度 | 984.9 | 721.1 | -263.8 |
| 终止回合数 | 1/20 | 16/20 | +15 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=142.321177, len=721.100000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-49.842686, 256.675901]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 437.264109 | 90.6% | 90.6% | 14.6% |
| velocity_brake_near_target | -26.151255 | -5.4% | 5.4% | 99.0% |
| goal_approach_progress | 12.375709 | 2.6% | 2.7% | 98.8% |
| orientation_stability | -6.254211 | -1.3% | 1.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主任务是控制一个二维刚体（类飞行器/着陆器）从顶部中央区域出发，在受到随机初始扰动力的情况下，尽快移动到画面中央的目标着陆区，并以低速、稳定姿态完成安全接触（“软着陆”）。次要目标是尽量减少引擎使用次数/推力消耗，追求燃料效率。不应将“存活/不坠毁”本身当作主目标——存活只是安全着陆的必要条件，而不是与到达目标并列的独立主目标；同时也不应把“悬停不动”误当作成功，因为任务要求最终稳定接触目标垫而非无限期悬空。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 未在源码中显式声明，推测为 float32（未确认，需以实际环境返回类型为准）
- obs[0]: x_position，相对目标垫的水平坐标，reward_usable: true
- obs[1]: y_position，相对目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体姿态角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右侧接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不启动任何引擎
- action 1: left_orientation_engine，启动左侧姿态引擎（产生反向力矩/侧向修正）
- action 2: main_engine，启动主引擎（主要提供向上/减速推力）
- action 3: right_orientation_engine，启动右侧姿态引擎（与 action 1 相对的修正方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功标志，但 `body_not_awake_or_settled`（机体进入静止/休眠状态）在其发生时若伴随 obs[6]、obs[7] 均为 1 且位置接近目标（obs[0]、obs[1] 接近 0）、速度接近 0，可视为“成功着陆”的间接证据；仅当满足这些位置+速度+接触组合条件时才可推断为成功，不能仅凭该终止标志本身判定成功。
- failure-like termination: `crash_or_body_contact`（发生坠毁/异常碰撞）通常对应失败；若发生该终止但同时接触点状态异常（例如姿态角过大、速度过大）则更确定为失败。
- ambiguous termination: `horizontal_position_outside_viewport`（水平方向越界）本身既可能是失败（飞出边界），但也可能与正常物理仿真边界裁剪有关，需结合速度/位置综合判断，不能单独作为成功或失败的确定信号。
- truncation: 源码中固定返回 `False`，即当前 masked step 未展示基于时间步数的截断机制；若存在最大步数限制，需以外部 wrapper 确认，此处不能假设。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（`info={}`，源码未声明任何字段）
- forbidden_or_uncertain_info_fields: `info["success"]`、`info["failure"]`、`info["termination_reason"]` 及任何未在 step 源码中声明的字段均不可假设存在，禁止使用。

## 7. 可用于奖励函数的信号
- position: obs[0]（相对目标水平位置）、obs[1]（相对目标垫垂直位置）
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）
- orientation: obs[4]（姿态角）、obs[5]（角速度）
- contact: obs[6]（左接触）、obs[7]（右接触）
- action/engine: action 本身（0=无动作，1/3=姿态引擎，2=主引擎），可用于估算引擎使用频率/能耗代理
- other: terminated 标志（由环境返回，可用于判断阶段结束，但需结合 obs 状态推断具体原因，不能直接读取“原因文本”）

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | goal_approach_progress + orientation_stability + terminal_soft_landing_bonus + velocity_brake_near_target | -82.86 | -82.86 | 0.00 | 984.85 | goal_approach_progress=0.031 orientation_stability=-0.071 terminal_soft_landing_bonus=14.147 velocity_brake_near_target=-0.200 | new_best |
| 2 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 142.32 | 142.32 | 0.00 | 721.10 | contact_landing_reward=2.104 goal_approach_progress=0.027 orientation_stability=-0.062 velocity_brake_near_target=-0.175 | new_best |
