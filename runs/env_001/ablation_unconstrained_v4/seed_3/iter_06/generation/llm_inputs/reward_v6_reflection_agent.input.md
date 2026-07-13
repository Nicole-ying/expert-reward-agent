# Search objective
- target_score: 200.000000
- current_score: -104.187741
- gap_to_target: 304.187741
- target_achievement_ratio: -52.094%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -104.187741）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Hyperparameters
    w_progress = 1.5
    k_landing = 4.0
    w_contact = 0.3
    w_speed_landing = 0.2
    w_angle = 0.5
    w_global_speed = 0.01

    # Current and next distances to target pad (relative coordinates)
    dist_now = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # --- Component A: progress toward target ---
    progress_reward = w_progress * (dist_now - dist_next)

    # --- Component B: soft‑landing (activated by proximity) ---
    landing_factor = 1.0 / (1.0 + k_landing * dist_next)
    contact_bonus = (next_obs[6] + next_obs[7]) * w_contact
    speed_penalty = w_speed_landing * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    soft_landing = landing_factor * (contact_bonus - speed_penalty - angle_penalty)

    # --- Component C: gentle global speed suppression ---
    global_speed_penalty = -w_global_speed * (abs(next_obs[2]) + abs(next_obs[3]))

    total_reward = progress_reward + soft_landing + global_speed_penalty
    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "global_speed_penalty": global_speed_penalty
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-104.187741, len=68.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-118.922361, -84.695453]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -4.490755 | -58.8% | 65.2% | 100.0% |
| progress_reward | 1.679642 | 22.0% | 22.8% | 100.0% |
| global_speed_penalty | -0.916694 | -12.0% | 12.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个二维飞行器的轨迹优化问题。主体从视口上方中央附近以随机初始受力出发，需要在最短时间内到达并稳定在中央目标着陆垫上，同时尽量少用引擎推力。智能体必须学会向目标靠近，减小水平和垂直速度，保持稳定姿态，并实现安全接触（双支撑腿同时触垫）。次要目标是节约燃料（即减少引擎使用次数），但绝不能以牺牲成功着陆和安全为代价。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: 默认 float32（连续值，接触标志为 0.0 或 1.0）  
- 各维度含义：
  - obs[0]：`x_position`，相对目标垫的水平坐标（负/正代表左右），可用于奖励计算 `reward_usable: true`
  - obs[1]：`y_position`，相对垫高度的垂直坐标，可用于奖励计算 `reward_usable: true`
  - obs[2]：`x_velocity`，水平线速度，可用于奖励计算 `reward_usable: true`
  - obs[3]：`y_velocity`，垂直线速度，可用于奖励计算 `reward_usable: true`
  - obs[4]：`body_angle`，机体偏转角（以弧度计），可用于奖励计算 `reward_usable: true`
  - obs[5]：`angular_velocity`，角速度，可用于奖励计算 `reward_usable: true`
  - obs[6]：`left_support_contact`，左支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`
  - obs[7]：`right_support_contact`，右支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - 0：`no_engine`，不做任何推力  
  - 1：`left_orientation_engine`，点燃左姿态引擎（通常产生顺时针力矩和微小侧向力）  
  - 2：`main_engine`，点燃主引擎（通常产生向上的主要推力）  
  - 3：`right_orientation_engine`，点燃右姿态引擎（通常产生逆时针力矩和微小侧向力）  

说明：动作是离散的推进器选择，每个时间步只能选择一种操作。因此奖励中无法直接获得连续推力值，但可基于动作类型判断是否使用了引擎，从而惩罚燃料消耗。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `termination_conditions` 原文，共三种终止原因：
1. **crash_or_body_contact（碰撞或身体接触）**  
   - 很可能是与地面或障碍物发生不当碰撞，通常代表失败。
2. **horizontal_position_outside_viewport（水平位置超出视口）**  
   - 机体横向飞出边界，代表失败。
3. **body_not_awake_or_settled（机体未醒或已稳定）**  
   - 语义存疑：可能表示机体已进入休眠状态（着陆后稳定静止）或未能稳定（翻倒后停止）。由于原文将它与前两类并列且未注明成功/失败，该终止条件本身是 **模糊的**。在典型同类任务中，机体成功着陆后会进入“睡眠”（awake=False），因而该条件可能是 **成功终止**。但在正式设计 reward 时，不能直接依赖该条件为成功标志，必须结合接触、位置、速度等观测自行判断。

### 5.2 步骤返回信息
`step()` 返回 `(state, masked_reward, terminated, False, {})`，其中 `info` 为空字典。因此：
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因 info 为空，无可用信号）

## 7. 可用于奖励函数的信号
基于 `obs`、`action` 和 `next_obs`，可提取以下信号：
- **位置信号**：`obs[0]`、`obs[1]`（或 `next_obs[0]`、`next_obs[1]`），表示相对目标垫的水平和垂直偏差。
- **速度信号**：`obs[2]`、`obs[3]`，水平与垂直线速度。
- **姿态信号**：`obs[4]`（偏转角）、`obs[5]`（角速度）。
- **接触信号**：`obs[6]`、`obs[7]`，左右支撑腿是否触地。
- **动作/引擎信号**：`action` 值，用于判断是否使用了引擎（0为无推力，1、2、3均消耗燃料）。
- **附加信号**：还可以计算 `next_obs` 与目标的距离变化、速度变化等差分信号。

所有这些信号均源自观测，可直接用于塑造奖励，无需依赖 info。

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_reward + gated_goal_reward | -10.16 | -10.16 | 0.00 | 1000.00 | contact_reward=0.237 gated_goal_reward=0.489 | new_best |
| 2 | contact_bonus + engine_penalty + gated_goal + height_reward | -63.81 | -10.16 | -53.64 | 1000.00 | contact_bonus=0.903 engine_penalty=-0.036 gated_goal=0.381 height_reward=0.680 | no_meaningful_improvement |
| 3 | contact_reward + engine_penalty + goal_reward + height_reward + landing_bonus | -20.70 | -10.16 | -10.54 | 1000.00 | contact_reward=0.250 engine_penalty=-0.074 goal_reward=0.441 height_reward=0.654 landing_bonus=0.913 | no_meaningful_improvement |
| 4 | contact_reward + descent_reward + engine_penalty + landing_bonus | -123.19 | -10.16 | -113.03 | 80.10 | contact_reward=0.016 descent_reward=-0.074 engine_penalty=-0.009 landing_bonus=0.038 | unsolved_stagnation_fresh_restart |
| 5 | global_speed_penalty + progress_reward + soft_landing | -104.19 | -10.16 | -94.02 | 68.55 | global_speed_penalty=-0.013 progress_reward=0.024 soft_landing=-0.066 | no_meaningful_improvement |
