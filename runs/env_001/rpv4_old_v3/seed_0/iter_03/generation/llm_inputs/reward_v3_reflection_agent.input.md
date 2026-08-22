# 1. Search objective
- target_score: 200.000000
- current_score: 217.212007
- gap_to_target: -17.212007
- target_achievement_ratio: 108.606%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `soft_landing_penalty` → 分数 -40.0 → 217.2（+257.2）✅ 大幅改善
- **你做了什么**：唯一目标组件为soft_landing_penalty，将其替换为landing_success_reward，条件为双腿接触、近中心、低速、小姿态角，持续给予正奖励。
- **你为什么这么做**：添加着陆成功信号后，agent会在dist接近0时获得持续激励，从而学会减速并维持姿态，减少失败硬着陆。
- **回合长度**：len: 154→484


**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: 217.212007）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    # 腿接触信息（0/1）
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励（替代原 soft_landing_penalty）
    # 当双腿接触、位置接近原点、速度小、姿态水平时给予持续正奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15   # 用于水平/垂直线速度绝对值
    landing_ang_thresh = 0.1    # 角度绝对值
    w_landing = 1.0

    contacts = left_contact + right_contact           # 0, 1, 2
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    # 只有两条腿都与地面接触且满足其他条件才算成功着陆
    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 碰撞预防（保留原有逻辑，尝试微调使之可能触发）
    x_bound = 0.9
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 4. 时间惩罚
    living_penalty = -0.002

    total_reward = (reward_prox + reward_landing +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 训练数据报告 — 基于 iter_02（本轮）与 iter_01（历史）

## 1. 行为事实
- **得分**  
  - iter_02：score=217.21，range=[61.62, 265.55]  
  - iter_01：score=-40.04，range=[-133.28, 15.32]  
- **终止方式**  
  - iter_02：terminated=18/20（成功/失败），truncated=2/20（时间上限）  
  - iter_01：terminated=20/20，truncated=0/20  
- **回合长度**  
  - iter_02：len=484.10  
  - iter_01：len=154.05  
- **Checkpoint 轨迹**（iter_02 官方环境得分）  
  - pct 40%：0.00  
  - pct 80%：0.00  
  - 变化趋势：持平（全程 0.00）

## 2. 骨架演进表
| iter | score | len | term | crash_prevention_x_ep_sum | crash_prevention_x_active | crash_prevention_y_ep_sum | crash_prevention_y_active | goal_proximity_ep_sum | goal_proximity_active | living_penalty_ep_sum | living_penalty_active | soft_landing_penalty_ep_sum | soft_landing_penalty_active | landing_success_reward_ep_sum | landing_success_reward_active |
|------|-------|-----|------|----------------------------|---------------------------|----------------------------|---------------------------|------------------------|-----------------------|------------------------|-----------------------|-----------------------------|-----------------------------|-------------------------------|-------------------------------|
| 1    | -40.04 | 154.05 | terminated=20/20, truncated=0/20 | 0.000 | 0.0% | 0.000 | 0.0% | 2.139 | 100.0% | -0.308 | 100.0% | -0.053 | 1.5% | — | — |
| 2    | 217.21 | 484.10 | terminated=18/20, truncated=2/20 | 0.000 | 0.0% | 0.000 | 0.0% | 2.568 | 97.6% | -0.968 | 100.0% | — | — | 36.450 | 7.5% |

> 注：iter_01 的 `soft_landing_penalty` 在 iter_02 中被替换为 `landing_success_reward`，故二者在同一行不可共存，分别列示。

## 3. 数字标记
- **主导**  
  - iter_02：`landing_success_reward` magnitude_share = 90.6% > 80%，主导。
- **死组件**  
  - iter_01 与 iter_02：`crash_prevention_x`、`crash_prevention_y` active_rate 均为 0.0% < 2%。  
  - iter_01：`soft_landing_penalty` active_rate = 1.5% < 2%。
- **淹没**  
  - 无。（iter_02：`goal_proximity` active_rate = 97.6% > 90%，但 magnitude_share = 7.0% > 3%，不满足淹没条件。）
- **全覆盖惩罚**  
  - iter_02：`living_penalty` active_rate = 100.0% > 95% 且 signed_share = -2.4% < 0，全覆盖惩罚。
- **量级对比**（iter_02 每步均值 = episode_sum_mean ÷ len）  
  - 最大两个  
    - `landing_success_reward`：36.450 ÷ 484.10 ≈ 0.0753  
    - `goal_proximity`：2.568 ÷ 484.10 ≈ 0.0053  
  - 最小两个  
    - `crash_prevention_x`：0.000  
    - `crash_prevention_y`：0.000  
    - （`living_penalty`：-0.968 ÷ 484.10 ≈ -0.002）
- **修改变化**（iter_01 → iter_02 的 Δ）
  - `goal_proximity`：ep_sum +0.429，active_rate -2.4pp  
  - `living_penalty`：ep_sum -0.660，active_rate 0.0pp  
  - `crash_prevention_x`：ep_sum 0.000，active_rate 0.0pp  
  - `crash_prevention_y`：ep_sum 0.000，active_rate 0.0pp  
  - `landing_success_reward`（新增）：ep_sum +36.450，active_rate +7.5pp  
  - `soft_landing_penalty`（移除）：ep_sum +0.053（由 -0.053 变为不适用），active_rate -1.5pp

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| crash_prevention_x | -0.000 | -0.000 | +0.000 | 0.3% | 0.0% | -0.2pp |
| crash_prevention_y | -0.007 | -0.001 | +0.006 | 0.3% | 0.0% | -0.3pp |
| goal_proximity | 0.021 | 0.013 | -0.008 | 100.0% | 99.8% | -0.2pp |
| landing_success_reward | — (新增) | 0.312 | 新增 | — | 31.2% | — |
| living_penalty | -0.002 | -0.002 | 0.000 | 100.0% | 100.0% | 0.0pp |
| soft_landing_penalty | -0.012 | — (新增) | 已删除 | 4.8% | — | — |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -40.04 | 217.21 | +257.25 |
| 平均回合长度 | 154.1 | 484.1 | +330.1 |
| 终止回合数 | 20/20 | 18/20 | -2 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=217.212007, len=484.100000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[61.623931, 265.546415]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 36.450000 | 90.6% | 90.6% | 7.5% |
| goal_proximity | 2.567816 | 6.4% | 7.0% | 97.6% |
| living_penalty | -0.968200 | -2.4% | 2.4% | 100.0% |
| crash_prevention_x | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_prevention_y | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：从画面顶部中央附近的随机初态出发，控制二维飞行器尽快到达中心目标垫并稳定着陆（速度接近零、姿态接近水平、支撑腿接触）。  
次目标：在完成主目标的前提下，尽可能少用引擎推力（节能）。  
混淆澄清：该任务不是纯粹的速度竞赛，也不是单纯的节能悬停，核心是实现“成功、快速、节能”的精确着陆。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: `float32`（推断）
- obs[0]: `x_position`，水平坐标（相对目标垫），可用于奖励接近目标，reward_usable: true
- obs[1]: `y_position`，垂直坐标（相对垫高），可用于奖励接近目标和防越界，reward_usable: true
- obs[2]: `x_velocity`，水平线速度，可用于奖励稳定着陆和禁止硬着陆，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，同上，reward_usable: true
- obs[4]: `body_angle`，机体俯仰角，可用于奖励姿态稳定，reward_usable: true
- obs[5]: `angular_velocity`，角速度，可用于奖励姿态平滑，reward_usable: true
- obs[6]: `left_support_contact`，左支撑腿接触标志（0/1），用于判断着陆成功，reward_usable: true
- obs[7]: `right_support_contact`，右支撑腿接触标志，同上，reward_usable: true

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- action 0: `no_engine`，不做任何推力/旋转，完全靠惯性
- action 1: `left_orientation_engine`，激发侧向方向引擎（改变机体转角，可能逆时针）
- action 2: `main_engine`，激发主引擎（沿机体朝向产生推力）
- action 3: `right_orientation_engine`，激发另一侧方向引擎（改变机体转角，可能顺时针）

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（当飞行器停稳、身体不再移动时触发，可能在正确着陆后发生）
- failure-like termination:  
  - `crash_or_body_contact`（主体非支撑部分碰地或坠毁）  
  - `horizontal_position_outside_viewport`（水平方向飞出视野）
- ambiguous termination: `body_not_awake_or_settled`在未到达目标垫时也可能触发（如坠落后停在边界外），但这时的终止不应视为成功；需要通过观察是否在目标垫上和接触确认。
- truncation: 代码中 `truncated` 恒为 `False`，无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `{}`（空字典，无任何 info 字段）
- forbidden_or_uncertain_info_fields: 不存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]`，任何推断成功/失败的逻辑必须从 `obs` / `next_obs` 的状态条件进行判定。

## 7. 可用于奖励函数的信号
- **位置**：`x_position`(obs[0]), `y_position`(obs[1]) — 水平与垂直相对目标垫，可直接用于奖励接近和判断边界。
- **速度**：`x_velocity`(obs[2]), `y_velocity`(obs[3]) — 线速度，用于奖励减速、软着陆。
- **姿态**：`body_angle`(obs[4]) — 角度（可能是弧度或归一化角），用于奖励水平姿态；`angular_velocity`(obs[5]) — 用于奖励姿态平稳。
- **接触**：`left_support_contact`(obs[6]), `right_support_contact`(obs[7]) — 布尔值，标志支撑腿是否接触地面，可奖励着陆成功。
- **动作/引擎**：`action` — 离散索引，可对主引擎和方向引擎使用进行惩罚，实现节能。

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_prevention_x + crash_prevention_y + goal_proximity + living_penalty + soft_landing_penalty | -40.04 | -40.04 | 0.00 | 154.05 | crash_prevention_x=-0.000 crash_prevention_y=-0.007 goal_proximity=0.021 living_penalty=-0.002 soft_landing_penalty=-0.012 | new_best |
| 2 | crash_prevention_x + crash_prevention_y + goal_proximity + landing_success_reward + living_penalty | 217.21 | 217.21 | 0.00 | 484.10 | crash_prevention_x=-0.000 crash_prevention_y=-0.001 goal_proximity=0.013 landing_success_reward=0.312 living_penalty=-0.002 | target_solved_new_best |
