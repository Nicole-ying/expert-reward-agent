# 1. Search objective
- target_score: 200.000000
- current_score: -46.512046
- gap_to_target: 246.512046
- target_achievement_ratio: -23.256%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `landing_velocity_penalty` → 分数 -156.0 → -119.5（+36.5）✅ 大幅改善
- **你做了什么**：将`landing_velocity_penalty`组件整体替换为`success_landing_bonus`，基于next_obs的接触、位置、速度、角度给出稀疏大正奖励。
- **你为什么这么做**：添加成功奖励后，策略在成功沾盘瞬间获得+100，远大于所有单步惩罚，应能强化接近并软着陆的行为，使最终得分由负转正、terminated中成功终止占比增加。
- **回合长度**：len: 154→68

### v3：修改 `crash_penalty` → 分数 -119.5 → -203.4（-83.9）❌ 明显退步
- **你做了什么**：新增 `crash_penalty` 组件，在接触地面但未满足软着陆条件、或 x 位置超出视区边界时给予 -100，与 success_landing_bonus 互斥，其余组件不变。
- **你为什么这么做**：添加与成功对称的 crash 惩罚后，agent 将被迫避免高速接触和越界，从而逐渐探索减速并在靠近目标时精准控制，最终触发 success_landing_bonus，score 应上升，终止方式可能出现成功终止或 len 增加。
- **回合长度**：len: 68→154

### v4：修改 `crash_penalty` → 分数 -203.4 → -114.4（+89.1）✅ 大幅改善
- **你做了什么**：删除 crash_penalty，将 success_landing_bonus 的判定距离阈值放宽到 0.2，速度阈值放宽到 0.5，角度阈值放宽到 0.2。
- **你为什么这么做**：放宽条件后 success_landing_bonus 激活率将显著上升，为 agent 提供可达的正向终极奖励，驱动减速和软着陆，从而使 score 大幅提升。
- **回合长度**：len: 154→68

### v5：修改 `progress_shaping` → 分数 -114.4 → -46.5（+67.9）✅ 大幅改善
- **你做了什么**：将success_landing_bonus的判定条件中的距离阈值由0.2提至0.4，水平/垂直速度阈值由0.5提至1.0，角度阈值由0.2提至0.5，保持接触条件不变。
- **你为什么这么做**：放宽后触发率应升至5%~15%，每episode平均bonus从30增至50以上，总score有望升至0附近或正数，terminated中成功着陆占比明显增加。
- **回合长度**：len: 68→740

## 你的修改轨迹（按组件分组）

- **`crash_penalty`**：修改了 **2 次**，分数变化：v3(-84) → v4(+89)
- **`landing_velocity_penalty`**：修改了 **1 次**，分数变化：v2(+36)
- **`progress_shaping`**：修改了 **1 次**，分数变化：v5(+68)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: -46.512046）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_reward = 5.0 * ((-next_dist) - (-dist))

    # 2. 成功着陆奖励 —— 进一步放宽阈值
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.4 and
        abs(next_x_vel) < 1.0 and
        abs(next_y_vel) < 1.0 and
        abs(next_body_angle) < 0.5 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

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

# 数据报告

## 1. 行为事实
- **得分**：score=-46.51，range=[-274.01, 201.88]  
- **终止方式**：terminated 7/20（成功或接触终止），truncated 13/20（达到时间上限 1000 步后被截断），实际回合长度 len=740.15  
- **Checkpoint 轨迹**：  
  - 40% 步数时官方得分 0.00  
  - 80% 步数时官方得分 0.00  
  得分在评估过程中始终为 0，直至最终阶段出现稀疏成功信号。

## 2. 骨架演进表

| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | action_cost_ep_sum | action_cost_active | angle_penalty_ep_sum | angle_penalty_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active | crash_penalty_ep_sum | crash_penalty_active |
|------|-------|-----|------|------------------------|----------------------|------------------|-----------------|--------------------|-------------------|-----------------------------|------------------------------|-------------------------|-------------------------|-------------------|-------------------|
| 1 | -155.98 | 154.4 | term=20 | 1.582 | 100.0% | -1.804 | 58.4% | -1.141 | 100.0% | -0.067 | 5.3% | — | — | — | — |
| 2 | -119.51 | 68.3 | term=20 | 5.583 | 100.0% | -0.071 | 5.2% | -0.205 | 100.0% | — | — | 5.0 | 0.1% | — | — |
| 3 | -203.45 | 154.2 | term=20 | 1.488 | 100.0% | -1.472 | 47.7% | -2.460 | 100.0% | — | — | 0.0 | 0.0% | 0.0 | 0.0% |
| 4 | -114.38 | 68.5 | term=20 | 5.597 | 100.0% | -0.097 | 7.1% | -0.087 | 100.0% | — | — | 30.0 | 0.4% | — | — |
| 5 | -46.51 | 740.2 | term=7, trunc=13 | 5.105 | 99.9% | -13.469 | 91.0% | -6.828 | 100.0% | — | — | 1195.0 | 1.6% | — | — |

*说明：空白项表示该轮未使用该组件；ep_sum 为 episode_sum_mean，active_rate 为 active_rate。*

## 3. 数字标记（基于当前轮 iter=5）
- **主导**：`success_landing_bonus` magnitude_share=97.7% >80%  
- **死组件**：`success_landing_bonus` active_rate=1.6% <2%  
- **淹没**：  
  - `action_cost` active_rate=91.0%，magnitude_share=1.1% <3%  
  - `progress_shaping` active_rate=99.9%，magnitude_share=0.7% <3%  
  - `angle_penalty` active_rate=100.0%，magnitude_share=0.6% <3%  
- **全覆盖惩罚**：`angle_penalty` active_rate=100% >95%，signed_share=-0.6% （为负）  
- **量级对比**（每步均值，ep_sum ÷ len）：  
  - 最大两个：`success_landing_bonus` (1.614)，`action_cost` (0.0182)  
  - 最小两个：`angle_penalty` (0.00922)，`progress_shaping` (0.00690)  
- **修改变化**（iter=5 相对于 iter=4 的 Δ）：
  - `success_landing_bonus`：ep_sum Δ = +1165.0，active_rate Δ = +1.2%
  - `action_cost`：ep_sum Δ = −13.372，active_rate Δ = +83.9%
  - `progress_shaping`：ep_sum Δ = −0.492，active_rate Δ = −0.1%
  - `angle_penalty`：ep_sum Δ = −6.741，active_rate Δ = 0%

## 4. 语义演化（跨轮组件功能追踪）

**分组依据**  
- **接近/进度信号**：基于位置/距离的引导（delta 形式）  
- **着陆/接触信号**：基于着陆事件/接触条件的奖励/惩罚  
- **稳定性/约束信号**：基于速度/角度/动作的惩罚或约束  
- **完成/成功代理**：大额条件性正信号（本任务无独立代理，由着陆信号承担）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|--------|---------|
| 接近/进度信号 | progress_shaping (v1-v5) | -156, -120, -203, -114, -46.5 | 无切换（持续势能塑造） |
| 着陆/接触信号 | v1: —<br>v2: success_landing_bonus<br>v3: success_landing_bonus, crash_penalty<br>v4,5: success_landing_bonus | -156, -120, -203, -114, -46.5 | 稀疏bonus → 稀疏bonus+稀疏惩罚 → 稀疏bonus；切换次数：2 |
| 稳定性/约束信号 | v1: landing_velocity_penalty, angle_penalty, action_cost<br>v2-v5: angle_penalty, action_cost | -156, -120, -203, -114, -46.5 | v1→v2 移除门控速度惩罚，其余持续连续二次型/线性惩罚；对立设计切换次数：1 |

**着陆/接触信号方向切换详情**：  
- v2 引入稀疏大正奖励 (`success_landing_bonus`)，为单一边界。  
- v3 增加稀疏大惩罚 (`crash_penalty`) 并与成功奖励互斥，形成正/负对称稀疏信号。  
- v4 删除惩罚，回到仅稀疏奖励，奖励形状再次变为单侧正激励。  
由此在“稀疏bonus ↔ 稀疏bonus+稀疏惩罚”之间经历了两次改动。

**稳定性/约束信号切换详情**：  
- 仅一次从 v1 的门控高度‑速度惩罚 (`landing_velocity_penalty`) 彻底移除，后续轮次未再恢复，未出现往复振荡。

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| action_cost | -0.008 | -0.014 | -0.006 | 40.4% | 71.7% | +31.3pp |
| angle_penalty | -0.004 | -0.022 | -0.017 | 100.0% | 100.0% | -0.0pp |
| progress_shaping | 0.079 | 0.040 | -0.039 | 100.0% | 100.0% | -0.0pp |
| success_landing_bonus | 0.449 | 28.088 | +27.639 | 0.4% | 28.1% | +27.6pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -114.38 | -46.51 | +67.86 |
| 平均回合长度 | 68.5 | 740.1 | +671.7 |
| 终止回合数 | 20/20 | 7/20 | -13 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-46.512046, len=740.150000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-274.011265, 201.880974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_landing_bonus | 1195.000000 | 97.7% | 97.7% | 1.6% |
| action_cost | -13.469000 | -1.1% | 1.1% | 91.0% |
| progress_shaping | 5.104587 | 0.4% | 0.7% | 99.9% |
| angle_penalty | -6.827954 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
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
| 3 | action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus | -203.45 | -119.51 | -83.94 | 154.15 | action_cost=-0.015 angle_penalty=-0.034 crash_penalty=-0.425 progress_shaping=-0.007 success_landing_bonus=0.002 | no_meaningful_improvement |
| 4 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -114.38 | -114.38 | 0.00 | 68.45 | action_cost=-0.008 angle_penalty=-0.004 progress_shaping=0.079 success_landing_bonus=0.449 | new_best |
| 5 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -46.51 | -46.51 | 0.00 | 740.15 | action_cost=-0.014 angle_penalty=-0.022 progress_shaping=0.040 success_landing_bonus=28.088 | new_best |
