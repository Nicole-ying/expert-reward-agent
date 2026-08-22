# Prompt Record

## System Prompt

```text
你是奖励函数修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- 你将看到：最终评估的组件表（episode_sum_mean, magnitude_share, active_rate），以及训练过程中的组件轨迹（每个 checkpoint 的标量值，含 max/mean/min）。轨迹能告诉你组件是在学习还是停滞了。
- 组件统计是观察证据，不是因果贡献。必须结合 score、episode_length、terminated/truncated 判断。

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。一次只改一个组件，不要顺带调整其他。

# 代码约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的 obs、next_obs、action 和 info 字段，不得发明字段、切片维度或新输入。
- 第一个 Python code block 只能包含一个完整的 `compute_reward` 函数；不要写 import、class、try/except 或额外函数，不要使用 self。
- 禁止 eval/exec/open，禁止使用 original_reward 或原始环境 reward。
- 需要平方根时使用 `** 0.5`，禁止 import numpy。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`；components 只放总公式中直接出现的奖励组件。

# 输出

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: -23.830843
- gap_to_target: 223.830843

# 2. Current reward program (score: -23.830843)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Core dense signal: exponential state goodness ----
    # Encourage all state components to approach zero (landed, upright, still)
    # Higher squared penalties for y, angle, and velocities to drive descent and stability
    squared_error = (0.2 * x**2 + 1.0 * y**2 + 0.5 * vx**2 + 0.5 * vy**2 +
                     5.0 * body_angle**2 + 2.0 * angvel**2)
    state_goodness = 10.0 * (2.71828 ** (-squared_error))  # use e^(-error)
    # Maximum ~10 when fully landed, decays gracefully with any deviation

    # ---- Contact reward: encourage touching the platform ----
    contact_reward = (left_contact + right_contact) * 0.5

    # ---- Descent bonus: reward downwards progress ----
    # y decreases when moving down, so obs[1]-next_obs[1] is positive on descent
    descent_bonus = 1.0 * max(obs[1] - next_obs[1], 0.0)

    # ---- Small per‑step penalty to discourage lingering ----
    time_penalty = -0.02

    total = state_goodness + contact_reward + descent_bonus + time_penalty

    components = {
        "state_goodness": state_goodness,
        "contact_reward": contact_reward,
        "descent_bonus": descent_bonus,
        "time_penalty": time_penalty
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-23.830843, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-52.136274, 15.534130]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| state_goodness | 8039.784593 | 99.7% | 99.7% | 100.0% |
| time_penalty | -20.000000 | -0.2% | 0.2% | 100.0% |
| descent_bonus | 1.272571 | 0.0% | 0.0% | 77.5% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Component checkpoint trajectories
# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_reward | 0.362038 | 0.362038 | 0.420979 | 0.859989 | 0.859989 | 0.000000 | 1.000000 | 1003520 |
| component.descent_bonus | 0.003866 | 0.003866 | 0.694558 | 0.005566 | 0.005566 | 0.000000 | 0.045305 | 1003520 |
| component.state_goodness | 6.466202 | 6.466202 | 0.999568 | 6.468999 | 6.468999 | 0.000000 | 9.999992 | 1003520 |
| component.time_penalty | -0.020000 | 0.020000 | 1.000000 | -0.020000 | 0.020000 | -0.020000 | -0.020000 | 1003520 |
| component.total_reward | 6.812105 | 6.812129 | 1.000000 | 6.812105 | 6.812129 | -0.020000 | 10.979992 | 1003520 |
| generated_reward | 6.812105 | 6.812129 | 1.000000 | 6.812105 | 6.812129 | -0.020000 | 10.979992 | 1003520 |
| original_env_reward | -0.136804 | 2.870004 | 1.000000 | -0.136804 | 2.870004 | -100.000000 | 126.658273 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_reward | 139.151457 | 139.151457 | 0.000000 | 755.000000 | 2608 |
| descent_bonus | 1.486367 | 1.486367 | 0.057584 | 2.069271 | 2608 |
| state_goodness | 2485.871524 | 2485.871524 | 16.394150 | 9324.536423 | 2608 |
| time_penalty | -7.688834 | 7.688834 | -20.000000 | -1.160000 | 2608 |
| total_reward | 2618.820514 | 2618.820514 | 15.535918 | 10017.527353 | 2608 |


# 4. Environment facts
## 1. 任务目标
主目标：控制一个从画面顶部中央附近出发的飞行器，安全、稳定地降落在画面中央的目标平台上。要求着陆时速度接近于零、姿态接近竖直，且所有支脚平稳接触平台。

次要目标：在确保主目标达成的前提下，尽量缩短飞行时间，并尽量减少主引擎和姿态引擎的使用（即节省燃料）。

不可混淆的目标：不应将“快速到达”或“节省燃料”凌驾于“安全着陆”之上；也不能将“悬停”或“保持在目标上方”当作成功条件。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- obs[0]: x_position (相对目标平台的水平坐标), 可直接用于距离/接近奖励，reward_usable: true
- obs[1]: y_position (相对目标平台高度的垂直坐标), 同上，reward_usable: true
- obs[2]: x_velocity (水平线速度), 可用于着陆软度控制，reward_usable: true
- obs[3]: y_velocity (垂直线速度), 同上，reward_usable: true
- obs[4]: body_angle (机体朝向角), 可用于姿态奖励，reward_usable: true
- obs[5]: angular_velocity (角速度), 可用于姿态稳定性惩罚，reward_usable: true
- obs[6]: left_support_contact (左侧支脚接触标志，1.0 表示接触), 可用于着陆状态判断，reward_usable: true
- obs[7]: right_support_contact (右侧支脚接触标志，1.0 表示接触), 同上，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力，仅受重力/物理影响)
- action 1: left_orientation_engine (点燃左侧姿态推进器，产生旋转力矩)
- action 2: main_engine (点燃主推进器，提供向上推力并可能产生力矩)
- action 3: right_orientation_engine (点燃右侧姿态推进器，产生反方向旋转力矩)

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: body_not_awake_or_settled 如果发生在飞行器已接触地面且速度/角速度极低时，极可能意味着成功着陆；但如果发生在半空中或刚碰撞后，则可能是早期终止。
- failure‑like termination: crash_or_body_contact（与地面或障碍的异常碰撞）、horizontal_position_outside_viewport（水平飞出边界）明确为失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分成功/失败，需要结合观察判断。
- truncation: 代码中未出现 episode length 截断，但实际部署时可能通过外部 wrapper 实现，当前源中未见。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {} 或未返回任何键）
- forbidden_or_uncertain_info_fields: 任何未在以上列出的字段均不可用

## 7. 可用于奖励函数的信号
- position: x_position, y_position（可直接计算到目标的距离、高度差）
- velocity: x_velocity, y_velocity（可衡量接近速度、着陆软度）
- orientation: body_angle（偏离竖直的角度），angular_velocity（旋转速度）
- contact: left_support_contact, right_support_contact（着陆脚是否触地，可判断着陆状态）
- action/engine: action 索引可映射到是否使用主引擎、姿态引擎，用于推力/燃料惩罚
- other: 可通过 (obs, next_obs) 的组合构造微分信号，如速度变化、角速度变化等
```
