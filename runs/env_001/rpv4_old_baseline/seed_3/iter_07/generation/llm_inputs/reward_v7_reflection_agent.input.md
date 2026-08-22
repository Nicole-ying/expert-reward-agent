# 1. Search objective
- target_score: 200.000000
- current_score: -87.870704
- gap_to_target: 287.870704

# 2. Current reward program (score: -87.870704)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    x, y = next_obs[0], next_obs[1]
    xv, yv = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact

    # 距离与速度
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (x ** 2 + y ** 2) ** 0.5
    speed_norm = (xv ** 2 + yv ** 2) ** 0.5

    # 1. 进度奖励：鼓励向目标移动
    progress = dist_prev - dist_next
    w_progress = 15.0

    # 2. 全局速度惩罚：鼓励整体减速
    speed_penalty = speed_norm
    w_speed = 0.1

    # 3. 姿态正则化：保持机体平稳
    angle_penalty = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.01

    # 4. 微弱接近引导：只用于初期的方向指示，避免 hover
    proximity = 0.5 / (1.0 + 20.0 * dist_next ** 2)

    # 5. 软着陆奖励：仅在双足接触、靠近目标且低速时给予
    soft_landing = 10.0 * both_legs / (1.0 + 50.0 * dist_next ** 2) / (1.0 + 10.0 * speed_norm ** 2)

    # 6. 接触鼓励：在接近目标时双足接触且速度不太高时给予阶段性奖励
    near_factor = 1.0 / (1.0 + 100.0 * dist_next ** 2)
    contact_enc = 0.0
    if both_legs and speed_norm < 2.0:
        contact_enc = near_factor * 10.0 * (1.0 - speed_norm / 2.0)

    # 7. 引擎惩罚：鼓励节约燃料（次要目标）
    engine_penalty = 0.05 if action != 0 else 0.0

    total_reward = (
        w_progress * progress
        - w_speed * speed_penalty
        - w_ori * angle_penalty
        + proximity
        + soft_landing
        + contact_enc
        - engine_penalty
    )

    components = {
        'progress': w_progress * progress,
        'speed_penalty': -w_speed * speed_penalty,
        'orientation': -w_ori * angle_penalty,
        'proximity': proximity,
        'soft_landing': soft_landing,
        'contact_encouragement': contact_enc,
        'engine_penalty': -engine_penalty,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-87.870704, len=773.650000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-152.048351, -30.434026]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| engine_penalty | -38.157500 | -36.3% | 36.3% | 98.6% |
| proximity | 33.776547 | 32.2% | 32.2% | 100.0% |
| progress | 10.946222 | 10.4% | 21.0% | 100.0% |
| speed_penalty | -10.901868 | -10.4% | 10.4% | 100.0% |
| orientation | -0.173134 | -0.2% | 0.2% | 100.0% |
| contact_encouragement | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
本环境是一个二维平面内的轨迹优化问题。智能体控制的类飞行器从视口顶部中心附近以随机初速度出发，核心目标是**快速且平稳地到达并停靠在中央目标垫上**。  
为了实现这一主目标，智能体需要学会：  
- 向中央目标区域靠近  
- 逐步降低运动速度  
- 维持稳定的朝向  
- 在目标垫上实现安全、低速的接触与稳定  

**次要目标**（应服从主目标）是在满足成功到达的前提下**尽可能少消耗引擎推力**。**不应混淆的目标**包括单纯追求极低能耗而放弃速度控制，或为了快速到达而引发猛烈坠地。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32 (假设，原始字段未声明，但连续值通常如此)  
- 各维度含义（index 从 0 开始）：
  - obs[0]: x_position，相对于目标垫的水平坐标，usable for reward: true  
  - obs[1]: y_position，相对于目标垫高度的垂直坐标，usable for reward: true  
  - obs[2]: x_velocity，水平线速度，usable for reward: true  
  - obs[3]: y_velocity，垂直线速度，usable for reward: true  
  - obs[4]: body_angle，机体朝向角，usable for reward: true  
  - obs[5]: angular_velocity，角速度，usable for reward: true  
  - obs[6]: left_support_contact，左侧支撑接触标志（0/1），usable for reward: true  
  - obs[7]: right_support_contact，右侧支撑接触标志（0/1），usable for reward: true  

注：所有字段均为环境直接提供，reward 函数中可以全部使用。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义：
  - action 0: `no_engine` – 不激活任何引擎，依靠惯性滑行  
  - action 1: `left_orientation_engine` – 点燃左侧姿态引擎，产生转向力矩（推测可使机体逆时针旋转）  
  - action 2: `main_engine` – 点燃主引擎，产生指向机体正向的推力（用于减速或抬升）  
  - action 3: `right_orientation_engine` – 点燃右侧姿态引擎，产生反向转向力矩（推测使机体顺时针旋转）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` 在**同时满足**位置接近目标、速度极低、双足接触目标垫的情况下，很可能代表成功着陆并稳定。  
- failure-like termination:  
  - `crash_or_body_contact`：机体或任何部位与地面/障碍发生非预期接触，视为坠毁或硬着陆。  
  - `horizontal_position_outside_viewport`：机体横向飞出允许范围，视为失控。  
- ambiguous termination:  
  `body_not_awake_or_settled` 出现在未到达目标区域或接触状态异常时，可能是中途卡死或坠落失败；需要结合下一状态观察区分成功/失败。  
- truncation:  
  源 step 代码中返回 `truncated=False`，无截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无 success 字段）  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info={}）  
- forbidden_or_uncertain_info_fields: 任何未在 step 源码中出现的字段均禁止使用（如 `success`, `done_reason`, `reward_components` 等）

因此奖励函数不能依赖 info 来获知成功或失败，必须基于 next_obs 的信号自行判断。

## 7. 可用于奖励函数的信号
- position: next_obs[0] (相对目标垫的水平距离)、next_obs[1] (相对垫高度)  
- velocity: next_obs[2] (水平速度)、next_obs[3] (垂直速度)  
- orientation: next_obs[4] (朝向角)、next_obs[5] (角速度)  
- contact: next_obs[6] (左腿接触)、next_obs[7] (右腿接触)  
- action/engine: `action` 本身可用于判断是否开启主引擎或姿态引擎  
- other: 可从上述信号推导出“已安全着陆”的复合条件（如位置接近零、速度接近零、双足均接地）