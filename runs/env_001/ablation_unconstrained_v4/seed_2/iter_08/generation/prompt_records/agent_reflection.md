# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。用训练证据分析失败原因，提出修改方案并实施。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现"跳跃、着陆、抓取"等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写"未知"，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 分析要求

拿到训练反馈后，回答：
1. 这个agent发生了什么？用行为证据说明。
2. 哪个组件最值得干预？结合数学形态和统计数据判断。
3. 我之前改了什么？从Agent Memory检查上一轮动作与实际效果。

随后根据分析结果修改奖励函数。你可以调整系数、改变数学形态、增加或删除组件、或重构整体结构——修改范围由证据决定。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

# 工具

以下工具可在诊断不确定时调用：

- `search_reward_design_knowledge(query)`：检索相似失败模式和经验修复。
- `get_skeleton_detail(skeleton_name)`：查看数学形态、原理和陷阱。
- `get_reward_transformation(query)`：查看结构变换的原理、风险和验证目标。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先简要说明分析和修改理由，然后立即输出完整Python代码。

```

## User Prompt

```markdown
# Duplicate reward retry
The previous generation duplicated iter 7 (runs\env_001\ablation_unconstrained_v4\seed_2\iter_07\generation\reward_v7.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 读取状态
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 全局前进引导（向原点缩短距离）
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5
    progress = 3.0 * (dist_old - dist_new)

    # 高度近地奖励（平方形态，宽范围）
    height_threshold = 1.5
    height_near_raw = max(0.0, height_threshold - y_new)
    w_height = 5.0
    height_near = w_height * (height_near_raw ** 2)

    # 水平对准（依赖高度的 gate）
    height_gate = max(0.0, 1.0 - y_new / height_threshold)
    alignment_raw = 1.0 / (1.0 + x_new ** 2)
    w_align = 0.5
    alignment = w_align * alignment_raw * height_gate

    # 接触奖励
    any_contact = max(left_contact, right_contact)
    w_contact = 2.0
    contact_reward = w_contact * any_contact

    # 水平速度惩罚（hinge）
    vx_limit = 0.2
    lat_penalty = -2.0 * (max(0.0, abs(vx_new) - vx_limit) ** 2)

    # 下坠速度惩罚（hinge，防止过快下降）
    vy_down_limit = 0.5
    down_penalty = -2.0 * (max(0.0, -vy_new - vy_down_limit) ** 2)

    # 姿态与角速度惩罚（hinge）
    angle_limit = 0.1
    att_penalty = -2.0 * (max(0.0, abs(angle_new) - angle_limit) ** 2)
    angvel_limit = 0.1
    angvel_penalty = -2.0 * (max(0.0, abs(angvel_new) - angvel_limit) ** 2)

    # 总奖励
    total_reward = (
        progress +
        height_near +
        alignment +
        contact_reward +
        lat_penalty +
        down_penalty +
        att_penalty +
        angvel_penalty
    )

    components = {
        'progress': progress,
        'height_near': height_near,
        'alignment': alignment,
        'contact_reward': contact_reward,
        'lat_penalty': lat_penalty,
        'down_penalty': down_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components
```

# Search objective
- target_score: 200.000000
- current_score: -119.598908
- gap_to_target: 319.598908
- target_achievement_ratio: -59.799%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -119.598908）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 读取状态 ----------
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 全局前进引导（向原点缩短距离）----------
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    progress = 3.0 * (dist_old - dist_new)

    # ---------- 高度近地奖励（平方形态，宽范围）----------
    height_threshold = 1.5
    height_near_raw = max(0.0, height_threshold - y_new)
    w_height = 5.0
    height_near = w_height * (height_near_raw ** 2)

    # ---------- 水平对准（依赖高度的 gate）----------
    height_gate = max(0.0, 1.0 - y_new / height_threshold)
    alignment_raw = 1.0 / (1.0 + x_new**2)
    w_align = 0.5
    alignment = w_align * alignment_raw * height_gate

    # ---------- 接触奖励 ----------
    any_contact = max(left_contact, right_contact)
    w_contact = 2.0
    contact_reward = w_contact * any_contact

    # ---------- 水平速度惩罚（hinge）----------
    vx_limit = 0.2
    lat_penalty = -2.0 * (max(0.0, abs(vx_new) - vx_limit) ** 2)

    # ---------- 下坠速度惩罚（hinge，防止过快下降）----------
    vy_down_limit = 0.5
    down_penalty = -2.0 * (max(0.0, -vy_new - vy_down_limit) ** 2)

    # ---------- 姿态与角速度惩罚（hinge）----------
    angle_limit = 0.1
    att_penalty = -2.0 * (max(0.0, abs(angle_new) - angle_limit) ** 2)
    angvel_limit = 0.1
    angvel_penalty = -2.0 * (max(0.0, abs(angvel_new) - angvel_limit) ** 2)

    # ---------- 总奖励 ----------
    total_reward = (progress + height_near + alignment + contact_reward +
                    lat_penalty + down_penalty + att_penalty + angvel_penalty)

    components = {
        'progress': progress,
        'height_near': height_near,
        'alignment': alignment,
        'contact_reward': contact_reward,
        'lat_penalty': lat_penalty,
        'down_penalty': down_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-119.598908, len=685.750000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-211.465199, 115.345158]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_near | 4738.597271 | 94.9% | 94.9% | 100.0% |
| alignment | 219.240640 | 4.4% | 4.4% | 100.0% |
| contact_reward | 16.900000 | 0.3% | 0.3% | 1.2% |
| lat_penalty | -7.154757 | -0.1% | 0.1% | 25.4% |
| progress | 1.469594 | 0.0% | 0.1% | 100.0% |
| att_penalty | -2.408608 | -0.0% | 0.0% | 11.7% |
| angvel_penalty | -1.970244 | -0.0% | 0.0% | 7.7% |
| down_penalty | -0.007635 | -0.0% | 0.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主任务：在二维平面中，控制一个带有两侧支撑腿的着陆器从视野上方中心附近受随机初速出发，**安全、稳定地降落到视野中央的目标平台上**，要求着陆器最终静止、姿态竖直且两侧支撑腿均与平台接触。  
次任务：在保证主目标的前提下，**尽可能快地完成着陆**，并且**尽量降低发动机推力消耗**（即少用燃料）。  
不应混淆的目标：燃料优化与快速着陆是辅助目标，它们不能凌驾于安全稳定着陆之上；本任务的最终评判是“是否成功停靠在平台上”这一离散事件，并非将能耗作为等价的主要目标。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: 根据上下文推断为 float32 (或混合了 bool 转为 float 的标志位，实际为连续量)
- **各维度含义**：
  - `obs[0]`: x_position (目标平台的相对水平坐标) —— reward_usable: true，用于控制水平对准。
  - `obs[1]`: y_position (相对于目标平台高度的垂直坐标) —— reward_usable: true，用于测量剩余高度/下降进度。
  - `obs[2]`: x_velocity (水平线速度) —— reward_usable: true，用于减速控制。
  - `obs[3]`: y_velocity (垂直线速度) —— reward_usable: true，用于控制坠落/着陆速度。
  - `obs[4]`: body_angle (机体偏转角) —— reward_usable: true，用于保持竖直姿态。
  - `obs[5]`: angular_velocity (角速度) —— reward_usable: true，姿态稳定惩罚。
  - `obs[6]`: left_support_contact (左支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。
  - `obs[7]`: right_support_contact (右支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `action 0` (no_engine): 不启动任何引擎（零推力）。
  - `action 1` (left_orientation_engine): 启动左侧姿态引擎，产生旋转力矩。
  - `action 2` (main_engine): 启动主引擎，产生向上的升力。
  - `action 3` (right_orientation_engine): 启动右侧姿态引擎，产生相反的旋转力矩。
- **值域**：`[0, 1, 2, 3]`，每个动作只消耗一个离散选择的“燃料单位”。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 机体静止且（可能）已与平台产生接触。根据任务描述“settle at a central target pad... make safe contact”，此条件很可能标志着成功着陆。但是否伴随 `left_support_contact` 和 `right_support_contact` 同时为真，需要观察，但终止代码中只用此单一条件判定终止，可能表示“已稳定着陆”。
- **failure-like termination**:  
  `crash_or_body_contact` —— 机体与外环境发生碰撞（非支撑腿安全着陆），通常是坠毁。  
  `horizontal_position_outside_viewport` —— 水平飘出视野边界，失败。
- **ambiguous termination**:  
  在没有显式成功标志的情况下，终止由这三种条件之一触发，其中 `body_not_awake_or_settled` 语义为“非清醒或已稳定”，我们暂时将其视为**成功候选**，但无法 100% 确认。需要后续根据经验收集证据。
- **truncation**:  
  源代码未展示任何步数限制或超时截断（`return ..., False, {}` 中没有 truncated）。如果存在，可能在上层包装器中，但当前分析不可见。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false （info 字典为空）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（info={} 且没有声明任何可用字段）
- **forbidden_or_uncertain_info_fields**: 所有 info 字段都不允许使用，因为没有可靠信息源。`original_reward` 被屏蔽，禁止访问。

## 7. 可用于奖励函数的信号
- **position**:
  - `next_obs[0]` (x_position)：到目标平台的水平距离。
  - `next_obs[1]` (y_position)：到目标平台高度的垂直距离。
- **velocity**:
  - `next_obs[2]` (x_velocity)
  - `next_obs[3]` (y_velocity)
- **orientation**:
  - `next_obs[4]` (body_angle)
  - `next_obs[5]` (angular_velocity)
- **contact**:
  - `next_obs[6]` (left_support_contact)
  - `next_obs[7]` (right_support_contact)
- **action/engine**:
  - `action`：离散动作编号，可用于惩罚引擎使用（燃料消耗），但不能直接读出推力大小。
- **other**:  
  无。

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_establishment + goal_proximity + soft_landing_stabilization + upright_attitude | -111.84 | -111.84 | 0.00 | 68.50 | contact_establishment=0.050 goal_proximity=-1.075 soft_landing_stabilization=-0.133 upright_attitude=-0.041 | new_best |
| 2 | att_penalty + contact_bonus + horizontal_penalty + progress + vel_penalty | -21.25 | -21.25 | 0.00 | 1000.00 | att_penalty=-0.008 contact_bonus=0.864 horizontal_penalty=-0.019 progress=0.003 vel_penalty=-0.009 | new_best |
| 3 | alignment + att_penalty + contact_reward + height_near_reward + horizontal_penalty + progress | 71.06 | 71.06 | 0.00 | 977.60 | alignment=0.469 att_penalty=-0.003 contact_reward=0.868 height_near_reward=0.293 horizontal_penalty=-0.004 | new_best |
| 4 | alignment + any_contact_reward + att_penalty + height_near_reward + progress + stable_contact_reward | -8.16 | 71.06 | -79.22 | 1000.00 | alignment=0.465 any_contact_reward=0.462 att_penalty=-0.004 height_near_reward=0.239 progress=0.010 | no_meaningful_improvement |
| 5 | alignment + any_contact_reward + att_penalty + height_near_reward + horizontal_penalty + progress | -69.73 | 71.06 | -140.79 | 937.95 | alignment=0.424 any_contact_reward=1.973 att_penalty=-0.006 height_near_reward=1.958 horizontal_penalty=-0.041 | no_meaningful_improvement |
| 6 | alignment + any_contact_reward + att_penalty + height_near_reward + horizontal_penalty + progress | -139.01 | 71.06 | -210.07 | 536.45 | alignment=0.077 any_contact_reward=0.936 att_penalty=-0.004 height_near_reward=0.278 horizontal_penalty=-0.014 | unsolved_high_achievement_continue_from_best |
| 7 | alignment + angvel_penalty + att_penalty + contact_reward + down_penalty + height_near | -119.60 | 71.06 | -190.66 | 685.75 | alignment=0.333 angvel_penalty=-0.126 att_penalty=-0.101 contact_reward=0.632 down_penalty=-0.039 | no_meaningful_improvement |

```
