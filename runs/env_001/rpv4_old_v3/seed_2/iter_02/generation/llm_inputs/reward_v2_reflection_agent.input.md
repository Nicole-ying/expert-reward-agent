# 1. Search objective
- target_score: 200.000000
- current_score: -155.975043
- gap_to_target: 355.975043
- target_achievement_ratio: -77.988%

# 2. 你的修改自传
（这是你第一次反思，没有历史。专注于当前反馈。）

# 3. 上一轮奖励函数代码（该轮得分: -155.975043）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Lunar Lander 风格任务奖励函数 v1。

    设计理念:
    - 主学习信号: 基于距离势能的 shaping 奖励，持续引导 agent 靠近目标平台。
    - 软着陆约束: 在下降末段对垂直速度施加 hinge 惩罚，防止高速撞击。
    - 姿态稳定: 轻量二次型角度惩罚，鼓励竖直姿态。
    - 能量效率: 轻量动作代价，抑制不必要的引擎使用。
    未使用稀疏成功/失败奖励，因为没有显式标志且稀疏信号难以在 v1 有效学习。
    """
    # ------------------------------------------------------------
    # 0. 解析观测
    # ------------------------------------------------------------
    # 从 obs 提取状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 从 next_obs 提取状态（用于势能塑造）
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ------------------------------------------------------------
    # 1. 主学习信号: 距离势能塑造 (goal_progress_guidance)
    # ------------------------------------------------------------
    # 用于持续引导智能体靠近目标 (x=0, y=0)
    # 势能函数: -sqrt(x^2 + y^2)，距离越近势能越高
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    potential_curr = -dist
    potential_next = -next_dist
    
    # 势能塑造: 向目标靠近一步的奖励增量
    progress_reward = 5.0 * (potential_next - potential_curr)
    
    # ------------------------------------------------------------
    # 2. 稳定/安全约束: 软着陆速度惩罚 (soft_landing_safety, 针对垂直速度)
    # ------------------------------------------------------------
    # 在高度较低时（y_pos < 0.5），严格惩罚过大的向下速度
    # 使用 hinge 惩罚：只惩罚超过安全阈值的 y_vel (向下为负，越大越负越危险)
    # 安全阈值：着陆时 y_vel 应 > -0.2 (即下降速度不超过 0.2)
    safe_y_vel_threshold = -0.2
    proximity_gate = max(0.0, 1.0 - y_pos / 0.5)  # y<0.5 时逐渐激活 (0->1)
    proximity_gate = min(1.0, proximity_gate)
    
    # 危险速度部分: 如果 y_vel < -0.2 (下降过快)，则产生惩罚
    dangerous_y_vel = max(0.0, -y_vel + safe_y_vel_threshold)  # 例如 y_vel=-0.8 -> max(0, 0.8-0.2)=0.6
    landing_velocity_penalty = -2.0 * proximity_gate * dangerous_y_vel
    
    # ------------------------------------------------------------
    # 3. 稳定/安全约束: 姿态惩罚 (soft_landing_safety, 针对角度)
    # ------------------------------------------------------------
    # 始终鼓励竖直姿态，但权重适中，不阻碍必要的机动
    # 使用二次型：轻微偏离时惩罚很小，大幅偏离时惩罚变大
    angle_penalty = -0.5 * (body_angle ** 2)
    
    # ------------------------------------------------------------
    # 4. 能量效率: 引擎动作代价 (energy_efficiency)
    # ------------------------------------------------------------
    # 对任何引擎点火（action != 0）施加小惩罚，鼓励利用惯性飞行
    # 注意：权重必须很小，不能压制主任务
    action_cost = -0.02 if action != 0 else 0.0
    
    # ------------------------------------------------------------
    # 5. 总奖励组装
    # ------------------------------------------------------------
    total_reward = (
        progress_reward +
        landing_velocity_penalty +
        angle_penalty +
        action_cost
    )
    
    # 记录各个组件（只记录被加进 total 的）
    components = {
        "progress_shaping": progress_reward,
        "landing_velocity_penalty": landing_velocity_penalty,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }
    
    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 训练数据报告

## 1. 行为事实
- 得分：score=-155.975043，range=[-180.244147, -114.227696]
- 终止方式：terminated 20/20（环境判定终止），truncated 0/20（无超时）——注：early_terminal 10/20（步数<150且score<-50），说明一半episode提前结束
- 回合长度：len=154.400000
- 奖励计算错误：reward_errors=0
- Checkpoint 轨迹：
  - cp_40%: score_mean=0.00, score_std=0.00
  - cp_80%: score_mean=0.00, score_std=0.00
  - 轨迹形态：持平（0.00→0.00），cp1~cp3、cp5~cp7、cp9~cp10 缺失

## 2. 骨架演进表
| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | action_cost_ep_sum | action_cost_active | angle_penalty_ep_sum | angle_penalty_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -155.98 | 154.40 | T=20/20, TR=0/20 | 1.582203 | 100.0% | -1.804000 | 58.4% | -1.140523 | 100.0% | -0.066830 | 5.3% |

## 3. 数字标记

### 主导
- **无**：最大 magnitude_share 为 progress_shaping=59.8%，未超过 80% 阈值

### 死组件
- **无**：最低 active_rate 为 landing_velocity_penalty=5.3%，高于 2% 阈值

### 淹没
- **无**：无组件同时满足 active_rate > 90% 且 magnitude_share < 3%

### 全覆盖惩罚
- **无**：landing_velocity_penalty 的 signed_share 为负（-0.9%），但 active_rate=5.3%，未超过 95%

### 量级对比（每步均值 = ep_sum ÷ len）
| 组件 | 每步均值 |
|---|---|
| progress_shaping | +0.010248 |
| action_cost | -0.011685 |
| angle_penalty | -0.007387 |
| landing_velocity_penalty | -0.000433 |

- 量级最大两个：progress_shaping（+0.010248）、action_cost（-0.011685）
- 量级最小两个：angle_penalty（-0.007387）、landing_velocity_penalty（-0.000433）

### 修改变化（Δ）
- 无历史轮次，无法计算 Δ

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | progress_shaping | -155.98 | N/A（仅一轮） |
| 着陆/接触信号 | landing_velocity_penalty | -155.98 | N/A（仅一轮） |
| 稳定性/约束信号 | angle_penalty, action_cost | -155.98 | N/A（仅一轮） |

**说明**：
- 接近/进度信号（progress_shaping）：基于位置势能的 shaping 奖励，连续奖励形态，无历史切换。
- 着陆/接触信号（landing_velocity_penalty）：基于垂直速度的 hinge 惩罚，proximity_gate 门控形态（y<0.5 激活），无历史切换。
- 稳定性/约束信号（angle_penalty + action_cost）：angle_penalty 为二次型姿态惩罚（始终激活），action_cost 为离散动作代价（按动作触发），均无历史切换。

**方向切换次数汇总**：0 次（仅一轮迭代，无可对比历史）

# 5. 组件逐项对比
（第一轮反思，无历史对比）

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-155.975043, len=154.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-180.244147, -114.227696]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.582203 | 21.1% | 59.8% | 100.0% |
| action_cost | -1.804000 | -24.1% | 24.1% | 58.4% |
| angle_penalty | -1.140523 | -15.2% | 15.2% | 100.0% |
| landing_velocity_penalty | -0.066830 | -0.9% | 0.9% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 的飞行器/着陆器轨迹优化任务。智能体从一个靠近顶部中央的位置出发，带有随机初始作用力。  
**主目标**：尽快到达画面中央的目标平台，并安全停靠（soft landing），即要求位置收敛到平台上、速度很低、姿态竖直、支撑脚安全触地。  
**次目标**：在完成任务的前提下，尽量少使用引擎推力（节省燃料），同时保持姿态稳定。  
**不应混淆的目标**：该任务不是纯生存或平衡任务——到达目标平台是刚性的成功条件，能量节省只是辅助约束，不是多目标的平等折衷。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: [8]
- **dtype**: float32 (推断)
- **obs[0]**: `x_position`，水平坐标（相对于目标平台中心），用于衡量横向偏移，奖励可用 ✅
- **obs[1]**: `y_position`，垂直坐标（相对于平台高度），用于衡量高度/下降进程，奖励可用 ✅
- **obs[2]**: `x_velocity`，水平线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[3]**: `y_velocity`，垂直线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[4]**: `body_angle`，机体角度（可能以弧度为单位，0 表示竖直），奖励可用 ✅
- **obs[5]**: `angular_velocity`，角速度，用于稳定姿态，奖励可用 ✅
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `0`: `no_engine` – 不激活任何引擎，仅靠惯性飞行。
  - `1`: `left_orientation_engine` – 点火左定向引擎，产生一个使机体逆时针（或某一方向）旋转的力矩。
  - `2`: `main_engine` – 点火主引擎，产生向上的推力（对抗重力并控制下降速度）。
  - `3`: `right_orientation_engine` – 点火右定向引擎，产生与动作 1 相反的旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `MASKED_STEP_SOURCE` 的 `terminated` 计算逻辑：
- **crash_or_body_contact**：躯体发生碰撞或身体支撑点接触地面。此情景可能包括：
  - 成功的软着陆（接触时速度极低、姿态正）。
  - 失败的高速撞击/翻倒等。
- **horizontal_position_outside_viewport**：x 坐标超出视区边界，明确失败。
- **body_not_awake_or_settled**：躯体进入睡眠状态或已稳定（settled）。这可能是：
  - 成功着陆后自然睡眠。
  - 坠毁后物理引擎判定身体安定，属于失败。

**总结终止模式**：
- **success-like termination**：`body_not_awake_or_settled` 在速度接近于 0、位置接近目标平台、接触标志为 1 且角度接近 0 时发生。
- **failure-like termination**：`horizontal_position_outside_viewport`；或因高速、大角度导致的 `crash_or_body_contact`/`body_not_awake_or_settled`。
- **ambiguous termination**：`crash_or_body_contact`/`body_not_awake_or_settled` 需结合 next_obs 判定成败。
- **truncation**：未提及，但有最大 episode 长度（通常存在），应在 reward 中注意时间步惩罚。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无任何显式标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因为没有提供）

> 因此，任何成功/失败的判断必须从 `next_obs` 与 `terminated` 的组合中推断，而不能依赖 `info`。

## 7. 可用于奖励函数的信号
- **position**：`x_position`, `y_position`（相对目标），可计算到目标的欧氏距离或分别控制。
- **velocity**：`x_velocity`, `y_velocity`，可设计近地减速、稳定约束。
- **orientation**：`body_angle`，鼓励竖直。
- **angular**：`angular_velocity`，鼓励旋转抑制。
- **contact**：`left_support_contact`, `right_support_contact`，标志着陆接触已发生，可配合速度用于判定软着陆。
- **action/engine**：`action` 选择，可用于惩罚引擎使用（燃料节省）。
- **other**：`terminated` 标志（作为环境返回，但 `compute_reward` 接口未直接提供，可通过在调用 `compute_reward` 时传入 `done` 标志获取；我们可以在训练循环中以额外参数传递，但接口中只有 `next_obs` 和 `info`；`terminated` 通常可在调用时通过另一个变量获得，若必须在 `compute_reward` 中获取，需要从外部传入——在此假设可以在调用时提供 `done` 标志，否则可通过 `next_obs` 的特征推断终止状态。由于题干未明确禁止，但谨慎起见，直接使用 `done` 需要额外参数，在标准接口中不鼓励，所以这里只强调可从 `next_obs` 和 `info` 推断。但 info 空，所以成功稀疏奖励只能基于 `next_obs` 的状态是否符合成功条件给出，而不依赖外部 `done`。所以我们仍可以将“是否处于终止且状态符合成功条件”作为奖励计算的一部分，但必须在训练循环的每个 step 结束时调用，需要能够判断 `done`，这部分由训练框架负责。我们此处声明信号可用，但不具体到实现细节。）

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_cost + angle_penalty + landing_velocity_penalty + progress_shaping | -155.98 | -155.98 | 0.00 | 154.40 | action_cost=-0.013 angle_penalty=-0.015 landing_velocity_penalty=-0.118 progress_shaping=0.035 | new_best |
