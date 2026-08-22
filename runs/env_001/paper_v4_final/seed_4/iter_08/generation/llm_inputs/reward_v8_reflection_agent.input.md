# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。 (record: runs\env_001\paper_v4_final\seed_4\iter_08\generation\validations\reward_v8.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
**信号覆盖审计**：
- **终止模式**：14/20 terminated，结合 len≈579，大部分 episode 以成功着陆后自然终止结束。landing_success_bonus 的 active_rate=3.4%、episode_sum_mean=291.75，说明 agent 已学会反复触发成功条件来刷分（波动进出阈值），属于 exploit。
- **观测使用**：8 维观测全部使用，无显著信号缺口。
- **僵尸组件**：无（active_rate 均 >0.4%）。
- **诊断结论**：当前高分主要来自 exploit，而非真正的任务效率。首要任务是修复 exploit，同时保住真实成功率并提升速度。

**修改目标**：`landing_success_bonus` 组件。
- **为什么改它**：当前宽松的成功阈值（dist<0.3, speed<0.5, angle<0.4）使得 agent 着陆后很容易因微小物理波动退出再进入，导致一次性奖励反复触发。收紧阈值（dist<0.15, speed<0.2, angle<0.1）让成功判定更稳固，一旦进入就很难脱出，从而自然实现“每 episode 仅一次”的奖励。同时将单次奖励从 15 提升至 150，以弥补 exploit 消除后的奖励损失并强化真正的完工信号。
- **数学形式**：保持 `(cur_success and not prev_success)` 的结构，但 `cur_success` 的条件改为：
  `dist_next < 0.15, cur_speed < 0.2, cur_angle < 0.1, contact`
  以及 `prev_success` 用上一帧观测相同计算。一次性奖金 = 150.0。
- **系数校准**：原成功奖金累计平均 291.75（全部来自 exploit），现在若成功率为 70%，平均 105，加上 progress/survival 等可望向 200 逼近。主信号 per-step ≈ 0.04，一次性 150 远大于 2x 主信号，但属于终端事件奖励，合理。

**为什么不改其他组件**：survival、progress、fuel、stability 在当前 exploit 修复后可能自动调整贡献，本轮仅专注修复 exploit 并加强完工信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0
    SURVIVAL_PENALTY = -0.08
    FAIL_PENALTY = -30.0               # 出界 / 坠毁 / 远距离静止 一次性惩罚

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功着陆检测参数（收紧以杜绝 exploit）
    SUCCESS_DIST_THRESH = 0.15
    SUCCESS_SPEED_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1
    LANDING_SUCCESS_BONUS = 150.0

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号 ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 一次性成功着陆奖励（收紧阈值杜绝反复触发） ----
    # 上一帧状态
    prev_dist = dist_cur
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    prev_angle = abs(obs[4])
    prev_contact = max(obs[6], obs[7]) > 0.5

    # 当前帧状态
    cur_speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    cur_angle = abs(body_angle_next)
    cur_contact = max(left_contact, right_contact) > 0.5

    # 判断成功条件（收紧）
    prev_success = (prev_dist < SUCCESS_DIST_THRESH and prev_speed < SUCCESS_SPEED_THRESH and
                    prev_angle < SUCCESS_ANGLE_THRESH and prev_contact)
    cur_success = (dist_next < SUCCESS_DIST_THRESH and cur_speed < SUCCESS_SPEED_THRESH and
                   cur_angle < SUCCESS_ANGLE_THRESH and cur_contact)

    # 仅在首次满足时授予
    landing_success_bonus = LANDING_SUCCESS_BONUS if (cur_success and not prev_success) else 0.0

    # ---- 6.

# 1. Search objective
- target_score: 200.000000
- current_score: 164.338982
- gap_to_target: 35.661018
- target_achievement_ratio: 82.169%

# 2. 上一轮奖励函数代码（该轮得分: 164.338982）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0
    SURVIVAL_PENALTY = -0.08
    FAIL_PENALTY = -30.0               # 出界 / 坠毁 / 远距离静止 一次性惩罚

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功着陆检测参数
    SUCCESS_DIST_THRESH = 0.3
    SUCCESS_SPEED_THRESH = 0.5
    SUCCESS_ANGLE_THRESH = 0.4
    LANDING_SUCCESS_BONUS = 15.0

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号 ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 一次性成功着陆奖励（替代连续 landing_prox） ----
    # 上一帧状态
    prev_dist = dist_cur
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    prev_angle = abs(obs[4])
    prev_contact = max(obs[6], obs[7]) > 0.5

    # 当前帧状态
    cur_speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    cur_angle = abs(body_angle_next)
    cur_contact = max(left_contact, right_contact) > 0.5

    # 判断成功条件
    prev_success = (prev_dist < SUCCESS_DIST_THRESH and prev_speed < SUCCESS_SPEED_THRESH and
                    prev_angle < SUCCESS_ANGLE_THRESH and prev_contact)
    cur_success = (dist_next < SUCCESS_DIST_THRESH and cur_speed < SUCCESS_SPEED_THRESH and
                   cur_angle < SUCCESS_ANGLE_THRESH and cur_contact)

    # 仅在首次满足时授予
    landing_success_bonus = LANDING_SUCCESS_BONUS if (cur_success and not prev_success) else 0.0

    # ---- 6. 终止事件推断（保留失败检测） ----
    out_of_bounds = abs(x_next) > X_BOUNDARY

    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 远距离静止失败
    distant_stall = (
        (y_next < GROUND_Y_CLOSE) and
        (dist_next > SUCCESS_DIST_THRESH) and
        cur_speed < SUCCESS_SPEED_THRESH and
        cur_angle < SUCCESS_ANGLE_THRESH and
        (not (left_contact > 0.5 or right_contact > 0.5))
    )

    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash or distant_stall) else 0.0

    # 合并奖励
    total_reward = (progress + survival + stability + fuel +
                    landing_success_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_success_bonus': landing_success_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=164.338982, len=579.050000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[26.235764, 253.199283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_bonus | 291.750000 | 66.2% | 66.2% | 3.4% |
| failure_penalty | -67.500000 | -15.3% | 15.3% | 0.4% |
| survival | -46.324000 | -10.5% | 10.5% | 100.0% |
| progress | 23.485588 | 5.3% | 6.4% | 99.1% |
| fuel | -4.191500 | -1.0% | 1.0% | 72.4% |
| stability | -2.801043 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 轨迹优化与精准着陆任务。一架拥有主引擎和双定向引擎的飞行器从视口顶部中心附近随机初始力释放，需要尽快飞到中心目标垫（着陆平台）上，实现安全、稳定的接触着陆。主要目标是**快速到达并悬停/降落在目标垫上**，次要目标是**节约燃料（减少引擎使用）**，同时在接近过程中保持姿态稳定，避免猛烈碰撞或飞出视口。任务不应被误解为持续前进或单纯生存平衡：它是以到达指定位置并“软接触”为核心的导航目标到达任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据典型实现推断）
- **obs[0]**: `x_position` – 飞行器相对于目标垫的水平坐标（偏移），reward_usable: true （可用作距离度量）
- **obs[1]**: `y_position` – 飞行器相对于垫高度的垂直坐标（偏移），reward_usable: true （距离度量）
- **obs[2]**: `x_velocity` – 水平线速度，reward_usable: true （可惩罚/奖励减速）
- **obs[3]**: `y_velocity` – 垂直线速度，reward_usable: true （同上）
- **obs[4]**: `body_angle` – 机体倾角，reward_usable: true （可鼓励保持水平）
- **obs[5]**: `angular_velocity` – 角速度，reward_usable: true （可惩罚过快的旋转）
- **obs[6]**: `left_support_contact` – 左侧支撑脚接触标志（0 或 1），reward_usable: true （可用于判断着陆状态）
- **obs[7]**: `right_support_contact` – 右侧支撑脚接触标志，reward_usable: true （同上）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- **action 0**: `no_engine` – 不点火，飞行器仅受重力、风等影响（节省燃料的动作）
- **action 1**: `left_orientation_engine` – 启动左侧定向引擎，产生力矩调整姿态
- **action 2**: `main_engine` – 启动主引擎，产生向上的推力（可能也有水平分量，取决于姿态）
- **action 3**: `right_orientation_engine` – 启动右侧定向引擎，反向力矩调整姿态

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 飞行器在目标垫附近稳定停靠（触发 `body_not_awake_or_settled`，且位置靠近原点，速度极小，倾角接近 0）。环境未显式提供成功 flag，需从观测组合推断。
- **failure-like termination**: 
  - `horizontal_position_outside_viewport`：飞行器飞出视口水平边界→失败。
  - `crash_or_body_contact`（但与目标垫安全接触不同的碰撞）：若飞行器身体（非支撑脚）触地或触垫，或姿态严重倾覆后接触→失败。
- **ambiguous termination**: 在远离目标处触发 `body_not_awake_or_settled`（如早期静止在顶部或其他位置）→应视为失败或无效终止。
- **truncation**: 未在 masked step 中出现，可能无时间截断，但实际环境通常有最大步数（未说明，此处只考虑给定的终止条件）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 没有任何明确可用的字段（step 返回 `{}`）
- **forbidden_or_uncertain_info_fields**: 不允许依赖 info 中的任何字段；如环境中存在的“success”、“failure”等均为未知，不能直接使用。

**推断路径**：通过观测信号间接区分成功/失败。成功着陆特征：`terminated=True` 时 `|x_position| < ε_x`、`|y_position| < ε_y`（接近垫中心）、`|x_velocity|, |y_velocity|` 接近 0、`|body_angle|` 接近 0、`left_support_contact` 与 `right_support_contact` 至少一个为 1（或两者为 1）。失败则表现为出界、远离目标时的静止、或高速/大角度接触。这些均属于 derived_possible 信号。

## 7. 可用于奖励函数的信号
从观测与终止推断的角度：
- **position**:
  - `x_position`（obs[0]）、`y_position`（obs[1]）— 反映到目标垫的欧氏距离或分量距离。
  - 可从 `next_obs` 获取下一步位置。
- **velocity**:
  - `x_velocity`（obs[2]）、`y_velocity`（obs[3]）— 反映接近速度，可用于减速奖励。
- **orientation**:
  - `body_angle`（obs[4]）— 应保持接近 0（水平）。
  - `angular_velocity`（obs[5]）— 抑制过快自旋。
- **contact**:
  - `left_support_contact`（obs[6]）、`right_support_contact`（obs[7]）— 脚部与垫接触，可用于终端着陆检测或稳定性奖励。
- **action/engine**:
  - `action` 本身可用于惩罚引擎使用（燃料惩罚），三个有推力动作为 1,2,3，可赋予不同权重。
- **other**:
  - 终端事件推断：根据终止时的 `next_obs` 状态组合判断成功着陆或失败（出界、坠毁等），提供 derived_possible 奖励/惩罚。