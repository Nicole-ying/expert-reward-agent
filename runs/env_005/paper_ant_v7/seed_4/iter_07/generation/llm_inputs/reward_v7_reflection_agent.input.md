# ⚠️ 上一版代码验证失败
错误信息：Reward v7 failed validation: 代码无法解析 AST: invalid character '：' (U+FF1A) (<unknown>, line 3) (record: runs/env_005/paper_ant_v7/seed_4/iter_07/generation/validations/reward_v7.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由

**审计结论**：当前奖励漏掉了对“灾难性摔倒瞬间爆发性角速度”的钳制——`angular_velocity_penalty` 使用无界二次惩罚，在机器人失衡、角速度飙升的少数几步会产生极大的负奖励（单步可达数十甚至上百），这是导致部分 episode 总奖励坍塌至 -1256 并拉低平均分的核心原因。而持续性的小幅角速度与侧向速度惩罚（active_rate ≈ 90%）虽然幅度不大，但联合作用已在历史 iter 4 中证实会诱导 agent 选择“少动苟活”策略——这表明全局无界惩罚正在与前进目标发生结构性冲突。

**行为诊断**：agent 学会了中等速度前进（gated_forward per‑step ≈ 2.0，len ≈ 577），但稳定性不足，约 60% 的 episode 因高度越界而提前终止。在终止前的极短窗口内，角速度爆发导致 `angular_velocity_penalty` 产生巨大负值，这些“惩罚尖峰”是平均 score 仍为负（–184）且 score_range 极宽（–1256 ~ +20）的直接原因。

**干预目标**：消除角速度惩罚在极端值时的爆炸性，同时保留其对正常步态下过度旋转的抑制梯度，引导 agent 在不被“惩罚尖峰”炸碎梯度的情况下逐步收敛到更稳定的步态。

**层级与变换**：**Level 2 结构变换**——`无界 → 有界`。当前 `angular_velocity_penalty = -0.02 · sum_sq` 在 `sum_sq` 较大时线性增长，现改为有界有理函数形态，使其在 `sum_sq` 超出正常范围后饱和于一个温和的上限（大约 –2.0/步），而在正常范围内保持与原惩罚相近的绝对量和梯度。这样在保持正常步态引导的同时，彻底切断“一步炸穿 episode”的路径。

**系数校准**：
- 新公式：`ang_penalty = -0.03 · sum_sq / (1 + sum_sq / 100)`
- 饱和上限：当 `sum_sq → ∞` 时，`ang_penalty → –3.0`（可控，不超过主信号 per‑step 的 1.5 倍，且只在极端少数步触发）。
- 原始 per‑step 约 –0.15，新惩罚在典型 `sum_sq ≈ 40‑60` 范围内约 –0.10 ~ –0.14，基本持平，不会过度削弱正常探索所需的转动自由度。
- 总惩罚负担预估：lateral 约 –0.18 + ang 约 –0.13 + height_hinge 约 –0.04 ≈ –0.35，对主信号 2.0 的比例 17.5%，远低于 50% 上限。

**为什么这个修改应该打破僵局**：历史上 iter 4 证明全局强惩罚会诱导 agent 为了生存而抑制前进；iter 5 证明单纯放大前进奖励会让 agent 跑得更快但不稳；当前迭代处在“跑得快但不稳→偶发巨罚”的脆性平衡中。将角速度惩罚有界化，相当于在保持前进引诱的同时，移除了炸碎梯度的地雷，agent 可以安全地在更大策略空间中搜索稳定步态，而不必因少数几次摔倒就被负反馈彻底淹没。这应能提高最差 episode 的得分，把平均分拉正，并为后续进一步精调留出空间。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract state from next_obs ----------
    body_z = next_obs[0]                     # body height
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]                         # quaternion (not used directly)
    body_x_vel = next_obs[13]                # forward velocity
    body_y_vel = next_obs[14]                # lateral velocity
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]                   # angular velocities

    # ---------- 1. soft health gate: upright factor ----------
    body_up_z = 1.0 - 2.0 * (qx*qx + qy*qy)
    tilt_error = 1.0 - body_up_z
    lambda_up = 3.0
    upright_gate = 2.718281828 ** (-lambda_up * tilt_error)   # ~1 when upright, ~0 when tilted

    # main progress signal: forward velocity, gated by upright
    gated_forward = body_x_vel * upright_gate

    # ---------- 2. height hinge penalty ----------
    lower_safe = 0.30
    upper_safe = 0.90
    height_penalty = max(0.0, lower_safe - body_z) + max(0.0, body_z - upper_safe)
    height_penalty_weighted = -10.0 * height_penalty

    # ---------- 3. lateral velocity penalty ----------
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    # ---------- 4. angular velocity penalty (bounded) ----------
    # Original: -0.02 * (roll^2 + pitch^2 + yaw^2) — unbounded quadratic expl

# 1. Search objective
- target_score: 2000.000000
- current_score: -184.033639
- gap_to_target: 2184.033639
- target_achievement_ratio: -9.202%

# 2. 上一轮奖励函数代码（该轮得分: -184.033639）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract state from next_obs ----------
    body_z = next_obs[0]                     # body height
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]                         # quaternion (not used directly)
    body_x_vel = next_obs[13]                # forward velocity
    body_y_vel = next_obs[14]                # lateral velocity
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]                   # angular velocities

    # ---------- 1. soft health gate: upright factor ----------
    # body_up_z = 1 - 2*(qx^2 + qy^2), ranges from -1 (inverted) to +1 (perfect upright)
    body_up_z = 1.0 - 2.0 * (qx*qx + qy*qy)
    # tilt_error (0 when perfectly upright, up to 2 when inverted)
    tilt_error = 1.0 - body_up_z
    # gate: decays exponentially when tilted, using e^{-lambda * tilt_error}
    lambda_up = 3.0
    upright_gate = 2.718281828 ** (-lambda_up * tilt_error)   # ~1 when upright, ~0 when tilted

    # main progress signal: forward velocity, gated by upright
    gated_forward = body_x_vel * upright_gate

    # ---------- 2. height hinge penalty ----------
    lower_safe = 0.30
    upper_safe = 0.90
    height_penalty = max(0.0, lower_safe - body_z) + max(0.0, body_z - upper_safe)
    # scale to keep penalty comparable but lighter than progress reward
    height_penalty_weighted = -10.0 * height_penalty

    # ---------- 3. lateral velocity penalty ----------
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    # ---------- 4. angular velocity penalty ----------
    ang_penalty = -0.02 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    # ---------- combine ----------
    w_forward = 2.0
    total_reward = (w_forward * gated_forward
                    + height_penalty_weighted
                    + lateral_penalty
                    + ang_penalty)

    components = {
        "gated_forward": w_forward * gated_forward,
        "height_hinge_penalty": height_penalty_weighted,
        "lateral_penalty": lateral_penalty,
        "angular_velocity_penalty": ang_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-184.033639, len=577.300000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-1256.425335, 20.101762]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 1154.737013 | 83.3% | 84.7% | 93.0% |
| lateral_penalty | -102.223362 | -7.4% | 7.4% | 90.0% |
| angular_velocity_penalty | -85.318223 | -6.2% | 6.2% | 89.8% |
| height_hinge_penalty | -24.267300 | -1.8% | 1.8% | 10.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个连续控制运动任务，控制一台具有四条腿、八个扭矩驱动关节的 3D 四足机器人。核心任务是让它持续、稳定地向前行走或奔跑，同时保持身体竖直。  
次要目标包括：抑制侧向漂移、降低身体角速度、减小关节动作能耗。机器人身体高度必须维持在健康范围内（0.2～1.0），否则立刻终止；同时任何状态量若变为 `NaN` 或 `inf` 也会终止。  
不应被混淆的目标：这里没有预定义的目标位置，不能把任务简单地当成“站立平衡”任务，因为必须产生稳定的前向位移才能持续获得奖励。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: 连续浮点（推测 float32）
- 各维度含义（索引 0～26）：
  - obs[0]: `body_z`，身体质心在世界坐标系中的高度，可用于检查健康范围，reward_usable: true
  - obs[1]: `quat_w`，身体姿态四元数的实部，reward_usable: true（用于计算直立程度）
  - obs[2]: `quat_x`，四元数第一虚部，reward_usable: true
  - obs[3]: `quat_y`，四元数第二虚部，reward_usable: true
  - obs[4]: `quat_z`，四元数第三虚部，reward_usable: true
  - obs[5]: `joint_1_angle`（第一个髋关节角度），reward_usable: true（可选用于姿态约束）
  - obs[6]: `joint_2_angle`（第一个踝关节角度），reward_usable: true
  - obs[7]: `joint_3_angle`（第二个髋关节角度），reward_usable: true
  - obs[8]: `joint_4_angle`（第二个踝关节角度），reward_usable: true
  - obs[9]: `joint_5_angle`（第三个髋关节角度），reward_usable: true
  - obs[10]: `joint_6_angle`（第三个踝关节角度），reward_usable: true
  - obs[11]: `joint_7_angle`（第四个髋关节角度），reward_usable: true
  - obs[12]: `joint_8_angle`（第四个踝关节角度），reward_usable: true
  - obs[13]: `body_x_velocity`，身体在世界 x 方向的线速度（前进速度），核心信号，reward_usable: true
  - obs[14]: `body_y_velocity`，身体在世界 y 方向（横向）的线速度，reward_usable: true（可惩罚侧向漂移）
  - obs[15]: `body_z_velocity`，身体在竖直方向的线速度，reward_usable: true（可选）
  - obs[16]: `body_roll_velocity`，滚转角速度，reward_usable: true
  - obs[17]: `body_pitch_velocity`，俯仰角速度，reward_usable: true
  - obs[18]: `body_yaw_velocity`，偏航角速度，reward_usable: true
  - obs[19]: `joint_1_velocity`，第一个髋关节角速度，reward_usable: true（可选用于平滑或能耗）
  - obs[20]: `joint_2_velocity`，第一个踝关节角速度，reward_usable: true
  - obs[21]: `joint_3_velocity`，第二个髋关节角速度，reward_usable: true
  - obs[22]: `joint_4_velocity`，第二个踝关节角速度，reward_usable: true
  - obs[23]: `joint_5_velocity`，第三个髋关节角速度，reward_usable: true
  - obs[24]: `joint_6_velocity`，第三个踝关节角速度，reward_usable: true
  - obs[25]: `joint_7_velocity`，第四个髋关节角速度，reward_usable: true
  - obs[26]: `joint_8_velocity`，第四个踝关节角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- 连续动作，每个维度范围 [-1.0, 1.0]（归一化扭矩指令）
- 各维度含义：
  - action[0]: `hip_1_torque`（第 1 条腿髋关节扭矩）
  - action[1]: `ankle_1_torque`（第 1 条腿踝关节扭矩）
  - action[2]: `hip_2_torque`
  - action[3]: `ankle_2_torque`
  - action[4]: `hip_3_torque`
  - action[5]: `ankle_3_torque`
  - action[6]: `hip_4_torque`
  - action[7]: `ankle_4_torque`

## 5. step 与终止条件分析
### 5.1 终止模式
- **失败型终止（failure-like termination）**：
  - 身体高度超出健康范围：`body_z ≤ 0.2` 或 `body_z ≥ 1.0`（collapse 或 jump/launch）。
  - 任意状态量变为 `NaN` 或 `inf`（在模拟中通常由极端碰撞或数值发散导致）。
- **截断型终止（truncation）**：
  - 达到时间步上限 `time_limit_reached`（超出最大步数），属于正常截断，不应直接视为成功或失败。

- **成功标志**：不存在显式的 `info["success"]` 字段。
- **模糊终止**：无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均被禁止，包括 `reward_forward`、`reward_ctrl`、`reward_contact`、`reward_survive`、`x_position`、`y_position`、`distance_from_origin` 等官方内部项。

注意：任务没有提供任何显式的成功/失败标志，奖励函数设计必须完全基于观察和动作信号，不能依赖 `info` 中的任何信息。

## 7. 可用于奖励函数的信号
- **位置与姿态**：
  - `body_z`（obs[0]）：用于高度存活惩罚。
  - 身体姿态四元数（obs[1:5]）：通过 `body_up_z = 1 - 2*(quat_x^2 + quat_y^2)` 计算身体 z 轴在世界坐标系中的投影，1.0 表示完全直立，0 表示水平。
  - 关节角度（obs[5:13]）：可用于姿态正则（但默认理想姿态未知，慎用）。

- **速度**：
  - 前向速度 `body_x_velocity`（obs[13]）：核心前进奖励信号。
  - 侧向速度 `body_y_velocity`（obs[14]）：横向漂移惩罚信号。
  - 垂直速度 `body_z_velocity`（obs[15]）：可选用于稳定性（例如惩罚快速起落）。
  - 角速度 `body_roll/pitch/yaw_velocity`（obs[16:19]）：旋转惩罚信号。
  - 关节角速度（obs[19:27]）：可用于动作平滑或能耗建模。

- **接触/力**：无。环境未提供接触力或足端接触布尔量。

- **动作/引擎**：
  - 动作向量 `action`：可直接惩罚扭矩大小（`torque_penalty = sum(action^2)`）或动作变化（若允许使用 `prev_action`，但本接口为单步无状态，无法直接获取上一步动作，因此只能使用当前动作幅度惩罚）。

- **其他**：`next_obs` 可对上述信号计算差分或在下个状态施加条件（例如若使用 next_obs 的身体高度，但可能导致奖励归属偏移，需谨慎）。