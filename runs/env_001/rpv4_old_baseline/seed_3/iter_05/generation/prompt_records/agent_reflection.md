# Prompt Record

## System Prompt

```text
你是奖励函数修订 Agent。根据当前轮次的训练反馈，修改奖励函数以改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated判断。

# 工作方式

阅读训练反馈和当前奖励代码，找出最可能导致低分或失败行为的一个组件，修改它。你可以调整系数、替换数学形式、删除组件或添加新组件。修改后输出完整的 `compute_reward` 函数。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件。

# 输出

直接输出完整Python代码。第一个Python code block必须只包含完整且可执行的`compute_reward`函数。
```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: -158.525996
- gap_to_target: 358.525996

# 2. Current reward program (score: -158.525996)
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

    # 1. 进度奖励：每一步缩小的距离
    progress = dist_prev - dist_next
    w_progress = 2.0

    # 2. 全局速度惩罚，鼓励全程保持低速，便于控制
    speed_penalty_global = speed_norm ** 2
    w_speed_global = 0.1

    # 3. 近端速度惩罚，在接近目标时强烈要求减速
    near_gate = 1.0 / (1.0 + dist_next)
    speed_penalty_near = speed_norm ** 2
    w_speed_near = 2.0

    # 4. 垂直速度惩罚，接近地面时防止重着陆
    ground_prox = 1.0 / (1.0 + abs(y) + 1e-5)
    vert_speed_penalty = yv ** 2
    w_vert_speed = 0.5

    # 5. 着陆引导：接触奖励 + 接近/低速额外奖励
    dist_gate = 1.0 / (1.0 + 10.0 * dist_next)
    speed_gate = 1.0 / (1.0 + 10.0 * speed_norm)
    contact_base = 5.0 * both_legs * (1.0 / (1.0 + dist_next))  # 越近接触越有价值
    landing_bonus = 200.0 * dist_gate * speed_gate * both_legs
    landing_reward = contact_base + landing_bonus

    # 6. 姿态正则化（保持平稳，避免旋转）
    angle_penalty = angle ** 2 + 0.1 * ang_vel ** 2
    w_ori = 0.001

    total_reward = (
        w_progress * progress
        - w_speed_global * speed_penalty_global
        - w_speed_near * near_gate * speed_penalty_near
        - w_vert_speed * ground_prox * vert_speed_penalty
        + landing_reward
        - w_ori * angle_penalty
    )

    components = {
        'progress': w_progress * progress,
        'speed_penalty_global': -w_speed_global * speed_penalty_global,
        'speed_penalty_near': -w_speed_near * near_gate * speed_penalty_near,
        'vert_speed_penalty': -w_vert_speed * ground_prox * vert_speed_penalty,
        'landing': landing_reward,
        'orientation': -w_ori * angle_penalty,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-158.525996, len=533.000000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-231.888213, -62.208703]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty_near | -36.471318 | -84.1% | 84.1% | 100.0% |
| speed_penalty_global | -3.778586 | -8.7% | 8.7% | 100.0% |
| progress | 0.464312 | 1.1% | 4.3% | 100.0% |
| vert_speed_penalty | -1.241775 | -2.9% | 2.9% | 100.0% |
| orientation | -0.002666 | -0.0% | 0.0% | 100.0% |
| landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
```
