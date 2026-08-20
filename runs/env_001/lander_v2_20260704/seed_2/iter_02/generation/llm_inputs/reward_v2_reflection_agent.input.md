# 上一轮奖励函数代码（该轮得分: -110.971651）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]           # relative horizontal distance to platform center
    y_pos = next_obs[1]           # relative vertical distance to platform support surface
    x_vel = next_obs[2]           # horizontal velocity
    y_vel = next_obs[3]           # vertical velocity
    body_angle = next_obs[4]      # tilt angle
    angular_vel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]    # left support leg contact (0 or 1)
    right_contact = next_obs[7]   # right support leg contact (0 or 1)

    # 1. Primary learning signal: continuous negative Euclidean distance to goal
    #    Guides the agent to move towards the landing platform center.
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Stability penalty: light constraint to suppress excessive speed, tilt, and spin
    #    Helps smooth approach and safe landing, but weights are low enough not to freeze exploration.
    w_vx = 0.15
    w_vy = 0.05   # vertical velocity penalty smaller to allow descent
    w_angle = 0.2
    w_angvel = 0.2
    stability_penalty = -(
        w_vx * abs(x_vel) +
        w_vy * abs(y_vel) +
        w_angle * abs(body_angle) +
        w_angvel * abs(angular_vel)
    )

    # 3. Soft landing proxy: multi‑condition bonus that signals near‑perfect landing posture
    #    Only triggers when the agent is close to the platform, has low speed, is nearly upright,
    #    and both legs are in contact. This gives a clear hint for the desired final state.
    prox_thresh = 0.5
    speed_thresh_x = 0.2
    speed_thresh_y = 0.3
    angle_thresh = 0.1
    soft_landing_bonus = 0.0
    if (distance_to_target < prox_thresh and
        abs(x_vel) < speed_thresh_x and
        abs(y_vel) < speed_thresh_y and
        abs(body_angle) < angle_thresh and
        left_contact == 1.0 and right_contact == 1.0):
        soft_landing_bonus = 0.5

    total_reward = distance_reward + stability_penalty + soft_landing_bonus

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-110.971651, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-127.611244, -95.909090]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -66.465033 | -88.7% | 88.7% | 100.0% |
| stability_penalty | -8.304364 | -11.1% | 11.1% | 100.0% |
| soft_landing_bonus | 0.175000 | 0.2% | 0.2% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。智能体从靠近视口顶部中心的位置出发，并承受一个随机初始作用力。目标是**尽快飞行并稳定降落在中央目标平台上，同时尽可能节省发动机推力**。智能体需要学习：平滑接近目标区域、控制下落速度、保持机身姿态稳定，并安全地让两侧支撑腿同时接触平台。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8,]  
- dtype: 通常为 float32（环境未显式声明，按连续控制环境惯例）  
- 各维度含义（index 从 0 开始）：
  - `obs[0]`: **x_position** —— 机体相对于目标平台中心的水平距离（相对坐标）  
  - `obs[1]`: **y_position** —— 机体相对于目标平台支撑面高度的垂直距离（相对坐标）  
  - `obs[2]`: **x_velocity** —— 机体水平线速度  
  - `obs[3]`: **y_velocity** —— 机体垂直线速度  
  - `obs[4]`: **body_angle** —— 机身倾斜角度（朝向）  
  - `obs[5]`: **angular_velocity** —— 机身绕质心的角速度  
  - `obs[6]`: **left_support_contact** —— 左侧支撑腿是否接触（0.0 或 1.0）  
  - `obs[7]`: **right_support_contact** —— 右侧支撑腿是否接触（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)  
- 离散动作含义：
  - `action 0`: **no_engine** —— 不开启任何发动机，被动受重力与惯性影响  
  - `action 1`: **left_orientation_engine** —— 点燃左侧姿态控制发动机（产生旋转力矩并伴随微小推力）  
  - `action 2`: **main_engine** —— 点燃主发动机（产生向上的主推力，同时可能伴随偏航力矩）  
  - `action 3`: **right_orientation_engine** —— 点燃右侧姿态控制发动机（与 action 1 方向相反的姿态控制）

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：`body_not_awake_or_settled` 当机身静止/稳定时触发。若此时机身位于目标平台区域、姿态接近水平且两腿均接触，则该终止可能对应成功着陆。但由于步骤函数未提供成功标记，此终止本质上仍具有歧义性。
- **failure-like termination**：  
  - `crash_or_body_contact` —— 机身本体（非支撑腿）撞击地面或其它障碍物，几乎一定意味着失败。  
  - `horizontal_position_outside_viewport` —— 水平位置超出允许边界，通常为飞出任务区域，属于失败。  
- **ambiguous termination**：  
  - `body_not_awake_or_settled` 若在平台外触发（如悬空稳定、在边界外停住或翻倒后静止），则属于失败；但因缺乏显式的成功/失败旗标，无法在终止时直接分辨。  
- **truncation**：代码片段未展示步数限制，但许多类似环境有 `max_steps` 截断；当前信息不足，暂不列入可用信号。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
- `explicit_failure_flag_available`: **false**  
- `allowed_info_fields`: `[]`（`info` 始终返回空字典 `{}`，未提供任何额外字段，亦无 `success`/`failure` 等字段）  
- `forbidden_or_uncertain_info_fields`: 所有 `info` 字段均不可用（因信息为空，强行使用会导致 KeyError 或不可预料行为）。特别说明：`info["success"]`、`info["failure"]`、`info["termination_reason"]` 等均不存在，严禁假设。

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`（水平相对距离）、`obs[1]`（垂直相对距离）及 `next_obs` 对应值，可构建距离惩罚或接近奖励  
- **速度**：`obs[2]`、`obs[3]`，用于惩罚过大的线速度（软着陆要求）  
- **朝向/角速度**：`obs[4]`、`obs[5]`，用于惩罚倾斜姿态或旋转  
- **接触**：`obs[6]`、`obs[7]`，两腿接触状态可用于奖励安全着陆（两腿同时接触）或惩罚单腿/无接触  
- **动作/发动机使用**：`action` 本身可用于计算节省燃料的代价（如动作=2 主发动机时惩罚，动作=0 时奖励）  
- 上述信号均可在 `compute_reward` 中通过 `obs`、`action`、`next_obs` 安全获取。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_bonus + stability_penalty | -110.97 | -110.97 | 0.00 | 68.40 | distance_reward=-0.974 soft_landing_bonus=0.002 stability_penalty=-0.127 | new_best |
