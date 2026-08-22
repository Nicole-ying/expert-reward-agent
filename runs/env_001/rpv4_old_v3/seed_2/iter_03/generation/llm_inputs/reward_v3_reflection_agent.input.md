# 1. Search objective
- target_score: 200.000000
- current_score: -119.506414
- gap_to_target: 319.506414
- target_achievement_ratio: -59.753%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `landing_velocity_penalty` → 分数 -156.0 → -119.5（+36.5）✅ 大幅改善
- **你做了什么**：将`landing_velocity_penalty`组件整体替换为`success_landing_bonus`，基于next_obs的接触、位置、速度、角度给出稀疏大正奖励。
- **你为什么这么做**：添加成功奖励后，策略在成功沾盘瞬间获得+100，远大于所有单步惩罚，应能强化接近并软着陆的行为，使最终得分由负转正、terminated中成功终止占比增加。
- **回合长度**：len: 154→68


**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: -119.506414）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Lunar Lander 风格任务奖励函数 v2。

    修改:
    - 移除无效的 landing_velocity_penalty，新增 success_landing_bonus:
      当 next_obs 满足软着陆条件时给予 +100 稀疏大奖励，解决 v1 无成功信号的问题。
    - 其余组件保持不变。
    """
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 主学习信号: 距离势能塑造 (goal_progress_guidance)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    # 判定条件: 距离目标极近、速度很小、姿态竖直、至少一个支撑脚触地
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    # 5. 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 训练数据报告

## 1. 行为事实

- **得分**：-119.506414，范围 [-142.824994, -96.557485]
- **终止方式**：terminated=20/20（全部因早停条件触发），truncated=0/20
- **回合长度**：均值 68.3 步
- **Checkpoint 轨迹**：
  - 40% 训练步数：官评得分 0.00（std=0.00）
  - 80% 训练步数：官评得分 0.00（std=0.00）
- **奖励组件统计**：
  - `progress_shaping`：ep_sum=5.583, active_rate=100.0%
  - `success_landing_bonus`：ep_sum=5.000, active_rate=0.1%
  - `angle_penalty`：ep_sum=-0.205, active_rate=100.0%
  - `action_cost`：ep_sum=-0.071, active_rate=5.2%

---

## 2. 骨架演进表

| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | angle_penalty_ep_sum | angle_penalty_active | action_cost_ep_sum | action_cost_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -155.975 | 154.4 | terminated=20/20, truncated=0/20 | 1.582 | 100.0% | -1.141 | 100.0% | -1.804 | 58.4% | -0.067 | 5.3% | — | — |
| 2 | -119.506 | 68.3 | terminated=20/20, truncated=0/20 | 5.583 | 100.0% | -0.205 | 100.0% | -0.071 | 5.2% | — | — | 5.000 | 0.1% |

---

## 3. 数字标记

### 修改变化（iter1 → iter2）

| 组件 | Δ ep_sum | Δ active_rate |
|---|---|---|
| progress_shaping | +4.001 | 0 pp |
| angle_penalty | +0.936 | 0 pp |
| action_cost | +1.733 | −53.2 pp |
| landing_velocity_penalty → success_landing_bonus | +5.067 | −5.2 pp |

### 死组件
- **`success_landing_bonus`**：active_rate=0.1%（< 2%）——该组件在绝大多数步中值为 0

### 淹没
- **`angle_penalty`**：active_rate=100.0%（> 90%）但 magnitude_share=1.9%（< 3%）

### 每步均值（ep_sum ÷ len）量级对比
- **最大两个**：`progress_shaping`=0.082, `success_landing_bonus`=0.073
- **最小两个**：`angle_penalty`=−0.003, `action_cost`=−0.001

### 无主导、无全覆盖惩罚
- 无组件 magnitude_share > 80%
- 无组件满足 active_rate > 95% 且 signed_share 为负

---

## 4. 语义演化

### 接近/进度信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 距离势能塑造 | v1: `progress_shaping` → v2: `progress_shaping` | v1: −155.975 → v2: −119.506 | 无 |
- v1→v2 未改变设计，为连续型 distance-to-goal shaping

### 着陆/接触信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 成功着陆反馈 | v1: `landing_velocity_penalty`(死) → v2: `success_landing_bonus`(死) | v1: −155.975 → v2: −119.506 | 1 次（连续惩罚 → 稀疏 bonus） |
- v1 使用 hinge 速度惩罚（高度门控），active_rate=5.3%（接近死组件）
- v2 替换为稀疏大额正面奖励（+100），active_rate=0.1%（死组件）
- 方向切换次数：1

### 稳定性/约束信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 姿态约束 | v1: `angle_penalty` → v2: `angle_penalty` | v1: −155.975 → v2: −119.506 | 无 |
- 两轮均使用二次角惩罚，权重 0.5，无设计变化
- v2 中该组件被淹没（active_rate=100%, magnitude_share=1.9%）

### 能效信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 动作代价 | v1: `action_cost` → v2: `action_cost` | v1: −155.975 → v2: −119.506 | 无 |
- 两轮均为 −0.02 per 非零动作，active_rate 从 58.4% 降至 5.2%

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| action_cost | -0.013 | -0.006 | +0.008 | 65.9% | 28.0% | -37.9pp |
| angle_penalty | -0.015 | -0.004 | +0.011 | 100.0% | 100.0% | -0.0pp |
| landing_velocity_penalty | -0.118 | — (新增) | 已删除 | 13.4% | — | — |
| progress_shaping | 0.035 | 0.080 | +0.045 | 99.8% | 100.0% | +0.2pp |
| success_landing_bonus | — (新增) | 0.208 | 新增 | — | 0.2% | — |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -155.98 | -119.51 | +36.47 |
| 平均回合长度 | 154.4 | 68.3 | -86.1 |
| 终止回合数 | 20/20 | 20/20 | +0 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-119.506414, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.824994, -96.557485]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 5.583480 | 50.5% | 52.3% | 100.0% |
| success_landing_bonus | 5.000000 | 45.2% | 45.2% | 0.1% |
| angle_penalty | -0.204852 | -1.9% | 1.9% | 100.0% |
| action_cost | -0.071000 | -0.6% | 0.6% | 5.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| 2 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -119.51 | -119.51 | 0.00 | 68.30 | action_cost=-0.006 angle_penalty=-0.004 progress_shaping=0.080 success_landing_bonus=0.208 | new_best |
