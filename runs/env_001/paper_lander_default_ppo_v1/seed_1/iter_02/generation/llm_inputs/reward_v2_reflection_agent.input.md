# Search objective
- target_score: 200.000000
- current_score: -108.316585
- gap_to_target: 308.316585
- target_achievement_ratio: -54.158%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -108.316585）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：distance-based progress（只奖励接近，不惩罚远离） ----
    prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_distance = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_distance - next_distance
    progress_reward = max(progress, 0.0) * 2.0  # 权重 2.0，鼓励有效靠近

    # ---- 稳定/安全约束：抑制高速、大角度和高角速度 ----
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    # 轻量级线性惩罚，防止剧烈翻滚和高速撞击
    stability_penalty_value = (
        0.1 * abs(vx)
        + 0.1 * abs(vy)
        + 0.1 * abs(angle)
        + 0.05 * abs(angular_vel)
    )
    # 组件记录为负贡献
    stability_penalty = -stability_penalty_value

    # ---- 任务完成近似信号：多条件组合的软着陆 bonus ----
    # 条件：双脚接触 + 位置接近垫中心 + 低速 + 姿态平稳
    soft_landing_bonus = 0.0
    if (next_obs[6] == 1.0 and next_obs[7] == 1.0
            and abs(next_obs[0]) < 0.3
            and abs(next_obs[1]) < 0.1
            and abs(vx) < 0.3
            and abs(vy) < 0.3
            and abs(angle) < 0.2):
        soft_landing_bonus = 1.0  # 为完成提供明确的溢价信号

    # ---- 总奖励 ----
    total_reward = progress_reward + soft_landing_bonus + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,  # 负值
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-108.316585, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.883170, -78.636102]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -9.457480 | -78.9% | 78.9% | 100.0% |
| progress_reward | 2.280888 | 19.0% | 19.0% | 91.9% |
| soft_landing_bonus | 0.250000 | 2.1% | 2.1% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器着陆任务。飞行器从视口上部中心附近出发，受随机初始力影响。目标是**尽快平稳地降落在中央的着陆垫上**，同时**尽可能少地使用引擎推力**。智能体需要学习：向目标靠近、减速、保持机身姿态稳定、实现安全触垫。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x位置（相对着陆垫的水平坐标）
- obs[1]: y位置（相对垫高度的垂直坐标）
- obs[2]: x方向线速度
- obs[3]: y方向线速度
- obs[4]: 机身角度（俯仰角）
- obs[5]: 角速度
- obs[6]: 左支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: 右支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎（不施加推力）
- action 1: 左侧姿控引擎点火
- action 2: 主引擎点火（向下推力）
- action 3: 右侧姿控引擎点火

## 5. step 与终止条件分析
### 5.1 终止模式
Termination 由以下三种条件触发（任一为真即结束）：
1. `crash_or_body_contact`：坠毁或身体与地面/其他物体发生接触（大概率是失败）。
2. `horizontal_position_outside_viewport`：水平位置超出边界（失败）。
3. `body_not_awake_or_settled`：机体进入休眠/稳定状态。当飞行器完全静止、不再受物理唤醒时触发。若此时已处于着陆垫上且支撑腿接触，则通常表示成功着陆；若在坠毁后静止，则被前面的 crash 条件先行触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（step 返回的 info 为空，无显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`，不可使用任何字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。由于源码显式返回 `{}`，不能假设存在 `success`、`termination_reason` 等键。

## 7. 可用于奖励函数的信号
- **位置**：相对着陆垫的 x, y 坐标（obs[0], obs[1]），可直接用于评估靠近/悬停/触垫。
- **速度**：x, y 线速度（obs[2], obs[3]），可用于惩罚高速撞击或奖励稳定减速。
- **姿态**：机身角度（obs[4]），角速度（obs[5]），用于鼓励保持直立姿态。
- **接触**：左右支撑接触标志（obs[6], obs[7]），可用于检测是否着陆成功、是否平稳（双脚着垫）。
- **动作/引擎使用**：动作编号本身，可以用于惩罚推力使用（动作 1,2,3 为引擎点火）以鼓励省油。

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.32 | -108.32 | 0.00 | 68.50 | progress_reward=0.033 soft_landing_bonus=0.005 stability_penalty=-0.136 | new_best |
