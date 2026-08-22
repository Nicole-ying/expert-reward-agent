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
- current_score: -125.711857
- gap_to_target: 325.711857

# 2. Current reward program (score: -125.711857)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包状态
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 势能塑造：鼓励接近目标，同时惩罚高速 ----
    w_dist = 6.0                 # 适当加大距离权重
    w_speed = 3.0                # 略微降低速度惩罚，避免阻碍机动
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping = potential_next - potential_cur

    # ---- 存活奖励：每步微小幅励，延长存活以增加着陆机会 ----
    survival_bonus = 0.05

    # ---- 角度惩罚：只在偏离竖直较大时生效，容忍小幅调整 ----
    angle_penalty = -2.0 * max(0.0, abs(nangle) - 0.25)

    # ---- 角速度惩罚：只在旋转过快时惩罚 ----
    angvel_penalty = -0.15 * max(0.0, abs(nang_vel) - 0.8)

    # ---- 燃料效率惩罚 ----
    if action == 2:           # 主引擎
        fuel_penalty = -0.2
    elif action in (1, 3):    # 姿态引擎
        fuel_penalty = -0.05
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 地面接近危险速度惩罚（防止硬着陆/撞毁） ----
    # 当飞行器已接近地面（ny 很小）且快速下落时强力惩罚
    ground_danger_penalty = 0.0
    if ny < 0.4 and nvy < -0.35:
        ground_danger_penalty = -5.0 * (0.4 - ny) * (abs(nvy) ** 0.5)  # 越接近越危险

    # ---- 双腿接触持续奖励 ----
    contact_continuous = 0.0
    if nleft and nright:
        speed_factor = max(0.0, 1.0 - next_speed)          # 慢速着陆
        angle_factor = max(0.0, 1.0 - abs(nangle) / 0.3)
        contact_continuous = 2.0 * speed_factor * angle_factor   # 扩大份量

    # ---- 成功软着陆大奖励（提高至200分） ----
    success_bonus = 0.0
    if nleft and nright and (next_speed < 0.5) and (abs(nangle) < 0.2):
        success_bonus = 200.0

    # ---- 猛烈着陆惩罚 ----
    crash_penalty = 0.0
    if nleft and nright and ((next_speed > 2.0) or (abs(nangle) > 0.6)):
        crash_penalty = -30.0

    # 汇总
    total_reward = (shaping +
                    survival_bonus +
                    angle_penalty +
                    angvel_penalty +
                    fuel_penalty +
                    ground_danger_penalty +
                    contact_continuous +
                    success_bonus +
                    crash_penalty)

    components = {
        "shaping": shaping,
        "survival_bonus": survival_bonus,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "fuel_penalty": fuel_penalty,
        "ground_danger_penalty": ground_danger_penalty,
        "contact_continuous": contact_continuous,
        "success_bonus": success_bonus,
        "crash_penalty": crash_penalty
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-125.711857, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -105.020073]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 30.000000 | 52.2% | 52.2% | 0.2% |
| ground_danger_penalty | -15.165073 | -26.4% | 26.4% | 16.4% |
| shaping | 6.220238 | 10.8% | 11.6% | 100.0% |
| survival_bonus | 3.420000 | 6.0% | 6.0% | 100.0% |
| angle_penalty | -1.176939 | -2.0% | 2.0% | 23.1% |
| angvel_penalty | -0.497319 | -0.9% | 0.9% | 1.1% |
| contact_continuous | 0.417198 | 0.7% | 0.7% | 0.8% |
| fuel_penalty | -0.122500 | -0.2% | 0.2% | 3.6% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
主体为一个2D飞行器（vehicle-like），初始位置在视口顶部中央附近，带有随机初始作用力。  
核心目标是 **尽快到达并稳定停靠在中央目标着陆平台上**，同时尽可能少使用引擎推力。  
智能体需要学习：趋近目标、降低线速度、保持姿态竖直、实现两腿同时安全接触（软着陆）。  
不应混淆的目标：单纯求快但忽略安全着陆，或一味省燃料而无法到达目标。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: [8]  
- **dtype**: 默认为 float64 或 float32（取决于环境实现，但通常为 float64）  
- **obs[0]**: `x_position`，水平方向相对于目标着陆平台中心的偏移量，可用于奖励趋近目标，reward_usable: true  
- **obs[1]**: `y_position`，垂直方向相对于平台高度（接触面）的偏移量，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，reward_usable: true  
- **obs[4]**: `body_angle`，机体倾角（如弧度），reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，reward_usable: true  
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **动作/索引 0**: `no_engine` (不做任何事)，语义：无推力，用于滑行或停靠后保持  
- **动作/索引 1**: `left_orientation_engine` (左姿态引擎)，语义：产生逆时针或顺时针旋转力矩（具体方向取决于环境）  
- **动作/索引 2**: `main_engine` (主引擎)，语义：产生纵向（向上）推力  
- **动作/索引 3**: `right_orientation_engine` (右姿态引擎)，语义：产生与左引擎相反的旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 可能表示机体静止稳定，通常意味着已着陆并静止，可能视为成功；  
- **failure-like termination**: `crash_or_body_contact` （如机身碰撞地面或平台以外部分）、`horizontal_position_outside_viewport` （漂出水平边界）很可能表示失败；  
- **ambiguous termination**: `crash_or_body_contact` 若接触平台但判定为 crash 则为失败，但描述未区分成功接触与失败接触；我们需要从“到达并稳定停靠”推断理想行为是两腿接触且低速，但不能直接从终止信号中获知成功。  
- **truncation**: 源代码未显示截断（max_steps），但多数环境有步数限制，此处未给出，视为不存在或不可直接用于奖励。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空字典 `{}`，无可用字段 。  
- **forbidden_or_uncertain_info_fields**: 任何 info 字段均不可用（因为提供的信息为空）。终止原因也不能从 info 获取。

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（可直接组成距离度量）  
- **velocity**: `x_velocity`, `y_velocity`（线速度）  
- **orientation**: `body_angle`, `angular_velocity`（角速度）  
- **contact**: `left_support_contact`, `right_support_contact`（双腿触地标志），可构造着陆条件  
- **action/engine**: 可以基于动作选择构建惩罚（如鼓励 no_engine，惩罚主引擎和姿态引擎）  
- **other**: 无
```
