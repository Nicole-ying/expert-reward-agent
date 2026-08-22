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
- current_score: -90.786259
- gap_to_target: 290.786259

# 2. Current reward program (score: -90.786259)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Parameters
    w_radial = 8.0             # reward for speed toward the target
    w_slow = 1.5               # reward for low speed (globally)
    w_angle = 1.0              # reward for upright attitude
    w_near = 3.0               # shaping reward when close to the pad
    w_orient = -0.05           # penalty for orientation engine
    w_main = -0.1              # penalty for main engine
    w_time = -0.01             # tiny time penalty
    w_landing = 200.0          # successful landing reward
    w_crash = -50.0            # penalty for non‑pad contact

    # Positions and distances
    x = next_obs[0]
    y = next_obs[1]
    dist = (x * x + y * y) ** 0.5 + 1e-8

    # Velocities
    vx = next_obs[2]
    vy = next_obs[3]

    # Attitude
    angle = next_obs[4]

    # Contacts
    left = next_obs[6]
    right = next_obs[7]

    # 1. Radial velocity reward: encourage moving toward the target
    dir_x = -x / dist
    dir_y = -y / dist
    radial_vel = vx * dir_x + vy * dir_y   # positive -> closing distance
    if radial_vel > 0:
        radial_reward = w_radial * radial_vel
    else:
        radial_reward = 0.0

    # 2. Low speed reward (global, decays when speed > 2.0)
    speed = (vx * vx + vy * vy) ** 0.5
    slow_reward = w_slow * max(0.0, 1.0 - speed / 2.0)

    # 3. Upright reward (angle in radians, reward falls linearly for |angle| > 0.5)
    angle_reward = w_angle * max(0.0, 1.0 - abs(angle) / 0.5)

    # 4. Proximity shaping: extra reward when inside a 2.0‑radius neighbourhood
    if dist < 2.0:
        near_reward = w_near * (2.0 - dist)
    else:
        near_reward = 0.0

    # 5. Engine usage penalty
    if action == 0:
        act_pen = 0.0
    elif action in (1, 3):
        act_pen = w_orient
    elif action == 2:
        act_pen = w_main
    else:
        act_pen = 0.0

    # 6. Constant time penalty
    time_pen = w_time

    # 7. Successful landing: both legs contact the pad near the target with low speed and upright
    full_contact = (left > 0.5 and right > 0.5)
    near_target = (abs(x) < 0.3 and abs(y) < 0.3)
    low_speed = (abs(vx) < 0.5 and abs(vy) < 0.5)
    upright = abs(angle) < 0.5
    if full_contact and near_target and low_speed and upright:
        landing_reward = w_landing
    else:
        landing_reward = 0.0

    # 8. Non‑pad contact penalty
    any_contact = (left > 0.5 or right > 0.5)
    if any_contact and not near_target:
        crash_pen = w_crash
    else:
        crash_pen = 0.0

    total_reward = (radial_reward + slow_reward + angle_reward +
                    near_reward + act_pen + time_pen +
                    landing_reward + crash_pen)

    components = {
        'radial_reward': radial_reward,
        'slow_reward': slow_reward,
        'angle_reward': angle_reward,
        'near_reward': near_reward,
        'act_pen': act_pen,
        'time_pen': time_pen,
        'landing_reward': landing_reward,
        'crash_pen': crash_pen,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-90.786259, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-126.160256, -55.371043]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| near_reward | 3219.904346 | 54.6% | 54.6% | 100.0% |
| slow_reward | 1439.041499 | 24.4% | 24.4% | 100.0% |
| angle_reward | 921.422495 | 15.6% | 15.6% | 99.8% |
| radial_reward | 225.722571 | 3.8% | 3.8% | 54.2% |
| act_pen | -78.560000 | -1.3% | 1.3% | 99.5% |
| time_pen | -10.000000 | -0.2% | 0.2% | 100.0% |
| crash_pen | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
这是一个 2D 飞行器/着陆器任务。主体从画面顶部中央附近出发，受到随机初始力作用。**主要目标**是尽快飞到并稳定在中央的目标着陆垫上，同时尽可能少用引擎推力。智能体必须学会：平滑接近目标、减小速度、保持水平姿态，并以安全接触方式着陆。**次要目标**是节约燃料（即少用引擎）。不应将目标分解为纯导航或纯生存，到达并稳定着陆是本环境的唯一核心目标，燃料效率是附加在相同轨迹上的性能优化。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: `x_position_relative_to_target`，水平坐标（相对目标垫），reward_usable: true
- obs[1]: `y_position_relative_to_pad_height`，垂直坐标（相对垫高度），reward_usable: true
- obs[2]: `x_velocity`，水平线速度，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，reward_usable: true
- obs[4]: `body_angle`，机体方向角（弧度？），reward_usable: true
- obs[5]: `angular_velocity`，角速度，reward_usable: true
- obs[6]: `left_support_contact`，左侧支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: `right_support_contact`，右侧支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作 0: `no_engine` —— 所有引擎关闭，不施加推力。
- 动作 1: `left_orientation_engine` —— 点燃左姿态引擎，产生改变姿态的力。
- 动作 2: `main_engine` —— 点燃主引擎（可能产生向上的主推力）。
- 动作 3: `right_orientation_engine` —— 点燃右姿态引擎，产生反向改变姿态的力。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确的成功终止类型，但 `body_not_awake_or_settled` 可能对应于成功着陆并稳定在目标垫上，亦可能在到达垫之前就休眠造成失败，需结合位置判断。
- **failure-like termination**: `crash_or_body_contact`（硬碰撞或非目标垫的接触）和 `horizontal_position_outside_viewport`（水平超出画布）显然是失败终止。
- **ambiguous termination**: `body_not_awake_or_settled` 本身不能直接区分成功或失败。
- **truncation**: 未提供任何截断条件。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空）
- explicit_failure_flag_available: false （info 为空）
- allowed_info_fields: 无（info 是空字典 `{}`）
- forbidden_or_uncertain_info_fields: 无（所有 info 字段均不可用）

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`（x）, `obs[1]`（y），以及对应的 next_obs 值，可构造距离、接近速度等。
- **速度**：`obs[2]`（vx）, `obs[3]`（vy），可用于惩罚过快着陆或水平漂移。
- **姿态与角速度**：`obs[4]`（角度）, `obs[5]`（角速度），可用于鼓励水平姿态和稳定性。
- **接触标志**：`obs[6]`, `obs[7]`，可判断是否与垫接触（但无法区分是目标垫还是其他表面，只能结合位置估计成功着陆）。
- **动作/引擎使用**：`action` 本身，0 为无推力，1/2/3 表示使用了引擎，可用于惩罚或奖励节油。
- **其他**：无。
```
