# Search objective
- target_score: 200.000000
- current_score: 222.454991
- gap_to_target: -22.454991
- target_achievement_ratio: 111.227%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 222.454991）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Compute Euclidean distance to target (landing pad center)
    d_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Main learning signal: progress towards the target
    progress = d_curr - d_next
    progress_clipped = max(-0.5, min(0.5, progress))
    progress_reward = 10.0 * progress_clipped

    # Stability penalty: discourage high speed, large angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.05 * abs(vx) - 0.02 * abs(vy) - 0.05 * abs(angle) - 0.01 * abs(angular_vel)

    # Landing quality reward: continuous bounded proxy replacing sparse binary bonus
    # Active only in landing zone (d_next < 0.3), providing gradient toward touchdown
    landing_zone = 0.3
    if d_next < landing_zone:
        # Proximity: 1 at d=0, 0 at d=landing_zone
        prox = 1.0 - d_next / landing_zone

        # Speed quality: 1 when still, 0 when speed >= 0.5
        speed = (next_obs[2]**2 + next_obs[3]**2)**0.5
        vel_quality = max(0.0, 1.0 - speed / 0.5)

        # Angle quality: 1 when upright, 0 when |angle| >= 0.3
        angle_quality = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)

        # Contact quality: 0 with no contact, 1 with both legs
        contact_quality = 0.5 * (next_obs[6] + next_obs[7])

        # prox gates all qualities; sum allows partial credit during learning
        landing_quality_reward = 2.0 * prox * (vel_quality + angle_quality + 0.5 * contact_quality)
    else:
        landing_quality_reward = 0.0

    total_reward = progress_reward + stability_penalty + landing_quality_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_quality_reward': landing_quality_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=222.454991, len=442.700000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[88.813365, 278.966143]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality_reward | 403.485551 | 95.4% | 95.4% | 65.3% |
| progress_reward | 12.381417 | 2.9% | 3.2% | 97.7% |
| stability_penalty | -5.812219 | -1.4% | 1.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从视口顶部中心区域出发（初始带有一定随机速度），尽快抵达中央的着陆垫并稳定着陆。  
要求：以尽可能快的速度接近目标，减少移动速度，保持稳定姿态，最后安全地接触到着陆垫。  
此外，在整个过程中尽量少使用主引擎推力，实现节能。

## 3. 观察空间 observation_space
- type: Box（连续值）
- shape: (8,)
- dtype: float32
- 每一维含义：
  - obs[0]: x_position — 飞行器相对于着陆垫中心的水平坐标（负左正右）
  - obs[1]: y_position — 飞行器相对于着陆垫表面高度的垂直坐标（正值在上方）
  - obs[2]: x_velocity — 水平线速度
  - obs[3]: y_velocity — 竖直线速度
  - obs[4]: body_angle — 机体倾斜角度
  - obs[5]: angular_velocity — 机体角速度
  - obs[6]: left_support_contact — 左支撑/接触标志（1.0 接触，0.0 不接触）
  - obs[7]: right_support_contact — 右支撑/接触标志（1.0 接触，0.0 不接触）

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- 动作含义：
  - action 0: no_engine — 不启动任何引擎，只受重力/惯性影响
  - action 1: left_orientation_engine — 启动一个姿态引擎，使机体逆时针（或某一方向）旋转
  - action 2: main_engine — 启动主引擎，产生与机体朝向相关的推力（通常向上）
  - action 3: right_orientation_engine — 启动另一个姿态引擎，使机体相反方向旋转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  没有显式的成功标记。但在任务描述中，“到达着陆垫并稳定接触”是目标。  
  当机体在垫上方、两侧接触标志均为1、且速度/角速度极小、位置稳定（即 `body_not_awake_or_settled`）时，环境会因 `body_not_awake_or_settled` 而终止。这种情况很可能对应成功着陆（需结合实际行为判断）。
- failure-like termination:
  - `crash_or_body_contact`：可能指机体与除着陆垫以外的地面或墙壁发生碰撞（如侧翻、头部碰撞等），这应属于失败。
  - `horizontal_position_outside_viewport`：飞行器水平位置超出视野边界，显然失败。
- ambiguous termination:  
  - `body_not_awake_or_settled` 这个终止条件本身不区分是成功着陆还是单纯因机体“睡着”（如动作太弱、陷入死区）。如果出现在没有着陆接触、远离目标点时，可能不代表成功；若出现在垫上且接触时，代表成功。因此单独看终止信号不能直接作为奖励依据。
- truncation: 未提供（本次环境没有设定最大步数截断，或已融入终止条件中）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `{}` （step 返回的 info 为空字典，可用字段无）
- forbidden_or_uncertain_info_fields: 任何自定义字段（如 `success`、`landed`、`crash` 等）均不可用，因为环境未提供

## 7. 可用于奖励函数的信号
- **位置（相对目标）**：`next_obs[0]` (x_position) 和 `next_obs[1]` (y_position) 可用于评判距着陆垫中心的距离
- **速度**：`next_obs[2]` (x_velocity) 和 `next_obs[3]` (y_velocity) 可用于评判机体在着陆时的平稳性
- **姿态**：`next_obs[4]` (body_angle) 和 `next_obs[5]` (angular_velocity) 可用于评判姿态是否保持水平/稳定
- **接触**：`next_obs[6]` (left_support_contact) 和 `next_obs[7]` (right_support_contact) 可用于检测是否着落到垫上（双接触通常说明成功着陆）
- **动作/引擎**：`action` 可用于衡量引擎使用情况，特别是 action 2（主引擎）的使用频率

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.