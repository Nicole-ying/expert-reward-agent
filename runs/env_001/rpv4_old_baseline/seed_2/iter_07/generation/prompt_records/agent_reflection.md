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
- current_score: 15.372568
- gap_to_target: 184.627432

# 2. Current reward program (score: 15.372568)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Shaped descent reward: encourage vy to match a target speed that decreases with height ----
    # Target downward speed should reduce as the lander gets closer to the platform (y approaching 0)
    # Using target_vy = -0.8 * max(y, 0.02) to avoid division‑by‑zero and keep target bounded
    height_for_target = max(y, 0.02)
    target_vy = -0.8 * height_for_target
    vy_error = abs(vy - target_vy)
    descent_shaping = -0.5 * vy_error

    # ---- Horizontal position penalty (keep x near 0) ----
    horiz_penalty = -0.2 * abs(x)

    # ---- Horizontal speed penalty (discourage lateral drift) ----
    vx_penalty = -0.1 * abs(vx)

    # ---- Orientation and angular velocity penalties ----
    orient_penalty = -0.1 * abs(body_angle)
    angvel_penalty = -0.05 * abs(angvel)

    # ---- Contact reward for touching feet ----
    contact_reward = (left_contact + right_contact) * 2.0

    # ---- Soft‑landing bonus (relaxed thresholds to make it more attainable) ----
    contact_ok = float(left_contact > 0.5 and right_contact > 0.5)
    speed_ok = float(abs(vx) < 0.5 and abs(vy) < 0.5 and abs(angvel) < 0.2)
    angle_ok = float(abs(body_angle) < 0.2)
    landing_bonus = 200.0 * contact_ok * speed_ok * angle_ok

    # ---- Small per‑step penalty to discourage lingering ----
    time_penalty = -0.01

    total = (descent_shaping + horiz_penalty + vx_penalty +
             orient_penalty + angvel_penalty + contact_reward +
             landing_bonus + time_penalty)

    components = {
        "descent_shaping": descent_shaping,
        "horiz_penalty": horiz_penalty,
        "vx_penalty": vx_penalty,
        "orient_penalty": orient_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "time_penalty": time_penalty
    }
    return total, components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=15.372568, len=254.950000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-121.335318, 234.496414]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 1880.000000 | 89.6% | 89.6% | 3.7% |
| contact_reward | 169.000000 | 8.1% | 8.1% | 25.8% |
| descent_shaping | -24.876707 | -1.2% | 1.2% | 100.0% |
| horiz_penalty | -15.039741 | -0.7% | 0.7% | 100.0% |
| vx_penalty | -4.304837 | -0.2% | 0.2% | 99.9% |
| time_penalty | -2.549500 | -0.1% | 0.1% | 100.0% |
| orient_penalty | -1.307606 | -0.1% | 0.1% | 100.0% |
| angvel_penalty | -0.368502 | -0.0% | 0.0% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Component checkpoint trajectories
# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.005014 | 0.005014 | 0.999839 | -0.005014 | 0.005014 | -0.414878 | -0.000000 | 1003520 |
| component.contact_reward | 0.449434 | 0.449434 | 0.135634 | 3.313590 | 3.313590 | 0.000000 | 4.000000 | 1003520 |
| component.descent_shaping | -0.297225 | 0.297225 | 1.000000 | -0.297225 | 0.297225 | -1.096806 | -0.000000 | 1003520 |
| component.horiz_penalty | -0.042572 | 0.042572 | 0.999999 | -0.042572 | 0.042572 | -0.203948 | -0.000000 | 1003520 |
| component.landing_bonus | 11.403460 | 11.403460 | 0.057017 | 200.000000 | 200.000000 | 0.000000 | 200.000000 | 1003520 |
| component.orient_penalty | -0.008258 | 0.008258 | 1.000000 | -0.008258 | 0.008258 | -0.375664 | -0.000000 | 1003520 |
| component.time_penalty | -0.010000 | 0.010000 | 1.000000 | -0.010000 | 0.010000 | -0.010000 | -0.010000 | 1003520 |
| component.total_reward | 0.972233 | 1.696638 | 1.000000 | 0.972233 | 1.696638 | -1.809165 | 20.000000 | 1003520 |
| component.vx_penalty | -0.033203 | 0.033203 | 0.999845 | -0.033209 | 0.033209 | -0.236849 | -0.000000 | 1003520 |
| generated_reward | 0.972233 | 1.696638 | 1.000000 | 0.972233 | 1.696638 | -1.809165 | 20.000000 | 1003520 |
| original_env_reward | -0.835950 | 2.747690 | 1.000000 | -0.835950 | 2.747690 | -100.000000 | 128.539549 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -0.449960 | 0.449960 | -9.356988 | -0.044301 | 11179 |
| contact_reward | 40.105555 | 40.105555 | 0.000000 | 3438.000000 | 11179 |
| descent_shaping | -26.674112 | 26.674112 | -82.628918 | -5.515745 | 11179 |
| horiz_penalty | -3.811126 | 3.811126 | -157.543727 | -0.003485 | 11179 |
| landing_bonus | 1012.595044 | 1012.595044 | 0.000000 | 163000.000000 | 11179 |
| orient_penalty | -0.741048 | 0.741048 | -38.870290 | -0.022034 | 11179 |
| time_penalty | -0.896764 | 0.896764 | -10.000000 | -0.490000 | 11179 |
| total_reward | 86.162000 | 126.349712 | -131.867449 | 16353.277980 | 11179 |
| vx_penalty | -2.979546 | 2.979546 | -24.729806 | -0.037618 | 11179 |


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
