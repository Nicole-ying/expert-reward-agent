# 1. Search objective
- target_score: 200.000000
- current_score: 165.623437
- gap_to_target: 34.376563
- target_achievement_ratio: 82.812%

# 2. 上一轮奖励函数代码（该轮得分: 165.623437）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 进度奖励：距离减小的量（鼓励靠近目标点）
    progress = current_dist - next_dist
    progress_reward = 2.0 * progress

    # 速度惩罚：抑制冲击速度（二次惩罚）
    velocity_penalty = 0.05 * (next_obs[2] ** 2 + next_obs[3] ** 2)

    # 姿态惩罚：抑制大幅倾斜（二次惩罚）
    angle_penalty = 0.1 * (next_obs[4] ** 2)

    # 软着陆近似奖励：同时满足双腿接触、靠近中心、低速、小角度时给予正向信号
    contact = next_obs[6] * next_obs[7]  # 1.0 仅当双腿都接触
    pos_factor = max(0.0, 1.0 - next_dist / 0.5)
    vel_sum = abs(next_obs[2]) + abs(next_obs[3])
    vel_factor = max(0.0, 1.0 - vel_sum / 0.5)
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.2)
    soft_landing = 0.5 * contact * pos_factor * vel_factor * angle_factor

    total_reward = progress_reward - velocity_penalty - angle_penalty + soft_landing
    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "soft_landing_proxy": soft_landing
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=165.623437, len=979.000000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[126.588596, 241.536025]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 325.849044 | 98.5% | 98.5% | 70.8% |
| progress_reward | 2.791908 | 0.8% | 0.9% | 99.8% |
| velocity_penalty | 1.642290 | 0.5% | 0.5% | 99.8% |
| angle_penalty | 0.428463 | 0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个二维飞行器/着陆器轨迹优化任务。agent 从视窗上方中央附近以随机初始力开始，需要尽快、省油地**到达视窗中央的目标着陆平台，并以安全姿态稳定接触**（即实现软着陆）。  
核心是导航到目标并实现 safe and stable contact，附属优化是节省发动机推力（能量效率）和缩短耗时，但不改变核心目标。  
**不可混淆**：任务不是持续前行（没有前进方向），也不是纯粹的存活（没有存活计时器），而是**定点到达 + 停稳**。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（推测）  

维度说明（索引从 0 开始，均为可用信号，reward_usable 均为 true）：

- **obs[0]**: `x_position` — 飞行器质心相对目标着陆平台的水平坐标（单位未知，相对值）。reward_usable: true  
- **obs[1]**: `y_position` — 飞行器质心相对平台高度的垂直坐标（下正？待确认方向；通常上正，但可通过初始位置和下降过程推断方向）。rewars_usable: true  
- **obs[2]**: `x_velocity` — 水平线速度。reward_usable: true  
- **obs[3]**: `y_velocity` — 垂直线速度。reward_usable: true  
- **obs[4]**: `body_angle` — 机体方向角（弧度）。reward_usable: true  
- **obs[5]**: `angular_velocity` — 角速度。reward_usable: true  
- **obs[6]**: `left_support_contact` — 左支撑腿是否接触平台（布尔化 float: 1.0/0.0）。reward_usable: true  
- **obs[7]**: `right_support_contact` — 右支撑腿是否接触平台。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  

动作含义：
- **action 0**: `no_engine` — 不启动任何发动机（滑行）。
- **action 1**: `left_orientation_engine` — 点燃左定向发动机，产生侧向/旋转力矩，可改变机体角度并小幅移动。
- **action 2**: `main_engine` — 点燃主发动机，产生主要推力（推测在机体坐标系向上或向下，结合角度影响水平和垂直速度）。
- **action 3**: `right_orientation_engine` — 点燃右定向发动机，与左对称，改变旋转和侧向移动。

## 5. step 与终止条件分析
### 5.1 终止模式
环境给出三个终止条件，经抽象后为：
- `crash_or_body_contact` — 飞行器坠毁或身体其他部分（非支撑腿）接触地面/平台，属于 likely failure。
- `horizontal_position_outside_viewport` — 水平坐标超出视窗范围，显然为 failure。
- `body_not_awake_or_settled` — 飞行器“休眠”或已经稳定停靠，这**很可能对应成功软着陆**（双腿接触且速度、角度足够小后触发）。由于任务目标是到达并 settle，该条件可作为 success-like termination。

当前 info 字典为空，无任何 explicit success/failure flag。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: 无（info 为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段（不可用）

**成功推断路径**（derived_possible）：  
当 episode 终止（terminated=True）且最后一次观测满足：  
 `left_support_contact == 1.0 && right_support_contact == 1.0`  
 并将 `abs(body_angle)`、`|x_velocity|`、`|y_velocity|` 控制在很小阈值内，且 `x_position` 和 `y_position` 接近零，则可认为发生了成功软着陆。  
**失败推断路径**：  
若终止时 `abs(x_position)` 很大（出界），或存在坠毁迹象（极端 body_angle 突变、两腿未接触），可判断为失败。由于缺少身体接触传感器，无法直接获得碰撞信号，角度过陡、速度冲击可作为间接证据。

## 7. 可用于奖励函数的信号
由于 info 不可用，reward 只能依赖 `obs`、`action` 和 `next_obs`。

- **位置**：`obs[0], obs[1]` 和 `next_obs[0], next_obs[1]`  
- **速度**：`obs[2], obs[3]` 和 `next_obs[2], next_obs[3]`  
- **姿态**：`obs[4]` (body_angle) 和 `next_obs[4]`  
- **角速度**：`obs[5]` 和 `next_obs[5]`  
- **接触**：`obs[6], obs[7]` 和 `next_obs[6], next_obs[7]` (双腿接触标志)  
- **动作**：`action` 值（离散 0-3），可用于动作效率惩罚/奖励

**可从观测间接推断的衍生信号**（derived_possible）：  
- 成功率线索：两腿接触 + 小速度 + 小倾角 + 接近零位置 → 可推断成功着陆  
- 坠毁线索：倾角突然超过安全阈值（如 abs(angle)>某一临界值）或速度骤变 → 可推断碰撞

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + progress_reward + soft_landing_proxy + velocity_penalty | 165.62 | 165.62 | 0.00 | 979.00 | angle_penalty=0.001 progress_reward=0.005 soft_landing_proxy=0.252 velocity_penalty=0.006 | new_best |
