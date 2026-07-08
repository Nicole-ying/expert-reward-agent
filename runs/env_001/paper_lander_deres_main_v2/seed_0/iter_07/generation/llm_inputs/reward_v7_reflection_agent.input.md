# 上一轮奖励函数代码（该轮得分: 59.609695）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 接触事件奖励。
    
    核心变换：将 persistent quality(s) 替换为 potential_delta = Φ(next) - Φ(prev)，
    agent 只为改善状态获得正奖励，消除"停留农耕"。
    
    Φ(s) = 1 / (1 + cost)，cost 加权综合距离/速度/姿态。
    """
    # ── 当前状态 ──
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    # ── 上一步状态 ──
    px = obs[0]
    py = obs[1]
    pvx = obs[2]
    pvy = obs[3]
    p_angle = obs[4]
    p_angvel = obs[5]

    # ── 当前代价 ──
    dist = (x * x + y * y) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    cost = (
        0.5 * dist +
        0.5 * speed +
        2.0 * abs_angle +
        1.0 * abs_angvel
    )

    # ── 上一步代价 ──
    prev_dist = (px * px + py * py) ** 0.5
    prev_speed = (pvx * pvx + pvy * pvy) ** 0.5
    prev_abs_angle = abs(p_angle)
    prev_abs_angvel = abs(p_angvel)

    prev_cost = (
        0.5 * prev_dist +
        0.5 * prev_speed +
        2.0 * prev_abs_angle +
        1.0 * prev_abs_angvel
    )

    # ── 势能函数：Φ(s) = 1/(1+cost)，有界 ∈ (0, 1] ──
    phi = 1.0 / (1.0 + cost)
    prev_phi = 1.0 / (1.0 + prev_cost)

    # ── 势能差塑形：Φ(next) - Φ(prev)，乘以缩放因子 ──
    progress_scale = 10.0
    landing_progress = progress_scale * (phi - prev_phi)

    # ── 接触事件奖励：双脚着陆 × 当前状态质量 ──
    both_contact = left * right  # 1.0 仅当双脚同时接触
    contact_scale = 5.0
    contact_bonus = contact_scale * both_contact * phi

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=59.609695, len=623.400000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-196.233676, 220.965173]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 391.134345 | 90.5% | 90.5% | 16.8% |
| landing_progress | 0.800112 | 0.2% | 9.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器（或着陆器）从视口顶部中心初始区域出发，在随机初始扰动下，**尽可能快地到达并稳定降落在中央目标着陆垫上**，同时**尽量减少引擎推力的使用**。智能体需要学会接近目标、减速、保持姿态稳定，并安全地与垫子接触。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: `x_position` — 飞行器相对于目标垫的**水平坐标**（可能经过缩放/归一化）
- obs[1]: `y_position` — 飞行器相对于垫面高度的**垂直坐标**
- obs[2]: `x_velocity` — 水平线速度
- obs[3]: `y_velocity` — 垂直线速度
- obs[4]: `body_angle` — 机体朝向角（弧度）
- obs[5]: `angular_velocity` — 角速度
- obs[6]: `left_support_contact` — 左侧支撑/着陆脚接触标志（0.0 或 1.0）
- obs[7]: `right_support_contact` — 右侧支撑/着陆脚接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` — 不进行任何引擎点火，无推力。
- action 1: `left_orientation_engine` — 点火左侧姿态引擎（产生逆时针旋转力矩）。
- action 2: `main_engine` — 点火主引擎（通常提供向上的主推力，可能同时产生微小转矩或水平分量，具体取决于机体朝向）。
- action 3: `right_orientation_engine` — 点火右侧姿态引擎（产生顺时针旋转力矩）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` — 当飞行器在目标垫上稳定着陆并静止时，物理引擎将身体标记为不活跃（settled），这可能对应成功着陆。但需要结合位置和接触状态确认是否真正在目标上。
- **failure-like termination**:  
  - `crash_or_body_contact` — 飞行器发生坠毁（如高速碰撞地面、壁面）或身体某些部分与不应接触的物体接触，导致坠毁或损坏。  
  - `horizontal_position_outside_viewport` — 飞行器水平方向飞出有效范围，无法再返回。
- **ambiguous termination**:  
  `body_not_awake_or_settled` 也可能在非目标区域发生（例如坠毁后静止在视口边缘），此时视为失败。需要利用位置观测（obs[0], obs[1]）和接触标志（obs[6], obs[7]）进一步区分。
- **truncation**: 未在掩码源代码中显式声明，但环境可能内置最大步数截断。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: `{}` （空字典，无任何字段）
- **forbidden_or_uncertain_info_fields**: `info` 中不包含任何键，**禁止依赖 `info` 判断成功或失败**；`terminated` 真值未传入 `compute_reward`，**禁止使用终止标志**。

## 7. 可用于奖励函数的信号
基于 `obs` 和 `next_obs` 可以直接使用以下信号构建奖励：
- **位置**：
  - `next_obs[0]` — 水平位置误差（希望趋近 0）
  - `next_obs[1]` — 垂直位置误差（希望趋近某个理想值，通常为 0 表示刚好在垫面高度）
- **速度**：
  - `next_obs[2]`, `next_obs[3]` — 线性速度的大小，着陆时应接近于 0
- **姿态**：
  - `next_obs[4]` — 机体角度，着陆时希望接近于 0（竖直）
  - `next_obs[5]` — 角速度，着陆过程中应可控且最终为 0
- **接触**：
  - `next_obs[6]`, `next_obs[7]` — 左右支撑接触标志，着陆成功时两者通常为 1
- **动作**：
  - `action` — 可用于惩罚非零引擎使用，以促进节能
- **变化量**：
  - 可计算 `obs[:6]` 与 `next_obs[:6]` 的差分，评估朝向目标状态的改善（如距离减小、速度降低等）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_proxy + stability_penalty | -108.90 | -108.90 | 0.00 | 68.45 | progress_delta=0.016 soft_landing_proxy=0.002 stability_penalty=-0.138 | new_best |
| 2 | progress_delta + soft_landing_proxy + stability_penalty | -566.80 | -108.90 | -457.89 | 114.90 | progress_delta=-0.002 soft_landing_proxy=0.000 stability_penalty=-0.004 | no_meaningful_improvement |
| 3 | progress_delta + soft_landing_proxy + stability_penalty | -109.58 | -108.90 | -0.67 | 68.45 | progress_delta=0.016 soft_landing_proxy=0.003 stability_penalty=-0.137 | no_meaningful_improvement |
| 4 | progress_delta + soft_landing_proxy + stability_penalty | -373.71 | -108.90 | -264.81 | 150.50 | progress_delta=-0.001 soft_landing_proxy=0.000 stability_penalty=-0.005 | unsolved_stagnation_fresh_restart |
| 5 | contact_bonus + landing_quality | -11.99 | -11.99 | 0.00 | 1000.00 | contact_bonus=0.010 landing_quality=0.650 | new_best |
| 6 | contact_bonus + landing_progress | 59.61 | 59.61 | 0.00 | 623.40 | contact_bonus=0.219 landing_progress=-0.003 | new_best |
