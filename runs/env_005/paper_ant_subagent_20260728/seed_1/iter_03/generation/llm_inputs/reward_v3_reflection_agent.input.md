# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 缺少准确函数签名; 函数体中使用了 original_reward; 代码无法解析 AST: unexpected EOF while parsing (<unknown>, line 1) (record: runs/env_005/paper_ant_subagent_20260728/seed_1/iter_03/generation/validations/reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由

当前 agent 的得分从 -11.32 暴跌至 -240.56，尽管存活长度显著提升至 244 步，这强烈暗示：
1. 乘积门控（`health_gate` × `forward_progress`）导致大量时间步 forward 信号为 0，agent 缺乏有效的引导；
2. 同时环境中很可能存在隐含的 per‑step 动作成本（动作维度多且无动作惩罚），存活越久累积的负代价越大，净收益崩盘。

本轮在**同一骨架内**实施一个 Level 2 结构变换：**将乘积门控改造成插值形式的“基础奖励 + 健康加成”**，使 forward 信号在所有状态下都保持非零梯度，从根本上解决乘积塌缩问题。具体变化：

- 移除 `forward_progress = max(0, body_x_velocity)` 的非负裁剪，改为直接使用原始前进速度 `body_x_velocity`（允许负值惩罚倒退）。
- 将 reward_forward 由 `forward_progress × health_gate` 改为 `forward_velocity × (0.3 + 0.7 × health_gate)`。  
  当健康状态完美时，增益为 1.0；当健康门控完全坍塌（gate=0）时，仍保留 0.3 的基础信号，保证每步都能感知“应该向前”的梯度。
- 系数保持 1.0，保证预期步均主信号量级与前轮大致可比，不引入额外尺度风险。

其他组件（upright_penalty, z_velocity_penalty）暂时不动，留待后续轮次根据反馈微调。

```python
def compute_reward(obs, action, next_obs, original_reward,

# 1. Search objective
- target_score: 2000.000000
- current_score: -240.556906
- gap_to_target: 2240.556906
- target_achievement_ratio: -12.028%

# 2. 上一轮奖励函数代码（该轮得分: -240.556906）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    body_x_velocity = obs[13]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: body_up_z = 1.0 - 2*(qx^2 + qy^2), range ~[-1, 1]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Health gate factors ----------
    # Height factor: healthy in (0.25, 0.95), smooth decay outside
    # Below 0.25: linear from 1.0 at z=0.25 down to 0.0 at z=0.15
    height_low_danger = max(0.0, (0.25 - body_z) / 0.10)  # 0.0 when z>=0.25, 1.0 when z<=0.15
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    # Above 0.95: linear from 1.0 at z=0.95 down to 0.0 at z=1.05
    height_high_danger = max(0.0, (body_z - 0.95) / 0.10)
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # Upright factor: healthy when body_up_z > 0.6, linear decay below
    upright_factor = min(1.0, max(0.0, (body_up_z - 0.2) / 0.4))
    # At body_up_z=0.6: (0.6-0.2)/0.4 = 1.0  → gate=1.0
    # At body_up_z=0.2: (0.2-0.2)/0.4 = 0.0  → gate=0.0
    # Smooth gradient throughout

    # Combined health gate: must satisfy both to earn full forward reward
    health_gate = min(height_factor, upright_factor)

    # ---------- Reward components ----------
    # 1) Forward progress gated by health
    forward_progress = max(0.0, float(body_x_velocity))
    reward_forward = 1.0 * forward_progress * health_gate

    # 2) Uprightness hinge penalty (only active when severely tilted)
    # body_up_z < 0.6: penalty grows linearly as tilt increases
    upright_error = max(0.0, 0.6 - body_up_z)
    reward_upright_penalty = -0.5 * upright_error

    # 3) Vertical velocity penalty (discourage bouncing/launching)
    # Soft hinge: penalize |vz| > 1.5
    vz_magnitude = abs(float(body_z_velocity))
    vz_excess = max(0.0, vz_magnitude - 1.5)
    reward_z_velocity_penalty = -0.2 * vz_excess

    # ---------- Total reward ----------
    total_reward = reward_forward + reward_upright_penalty + reward_z_velocity_penalty

    components = {
        "forward_gated": reward_forward,
        "health_gate": health_gate,
        "upright_penalty": reward_upright_penalty,
        "z_velocity_penalty": reward_z_velocity_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-240.556906, len=244.400000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-1133.041330, -2.945546]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 214.880584 | 65.8% | 65.8% | 88.1% |
| forward_gated | 88.621857 | 27.1% | 27.1% | 52.5% |
| upright_penalty | -22.195416 | -6.8% | 6.8% | 12.0% |
| z_velocity_penalty | -0.731142 | -0.2% | 0.2% | 3.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个连续控制四足机器人运动的环境。主要目标是使四足机器人持续、稳定地向前行走或奔跑，同时保持身体在健康高度范围内（0.2 ~ 1.0），并维持身体基本直立。机器人不应仅仅满足于不倒，而是需要产生有效的向前位移。次要目标包括抑制横向滑移、垂直弹跳、过大的关节力矩或速度，以提升运动的平稳性和能效。没有预设的目标位置，成功由存活至时间截断并持续向前移动来体现，不存在显式的“到达终点”成功标志。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- each index 的含义与 reward 可用性：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | body_z | 身体质心垂直位置（高度） | true |
| 1 | quat_w | 身体姿态四元数实部 | true（需组合） |
| 2 | quat_x | 身体姿态四元数 x | true |
| 3 | quat_y | 身体姿态四元数 y | true |
| 4 | quat_z | 身体姿态四元数 z | true |
| 5 | joint_1_angle | 第1髋关节角度 | true |
| 6 | joint_2_angle | 第1踝关节角度 | true |
| 7 | joint_3_angle | 第2髋关节角度 | true |
| 8 | joint_4_angle | 第2踝关节角度 | true |
| 9 | joint_5_angle | 第3髋关节角度 | true |
| 10 | joint_6_angle | 第3踝关节角度 | true |
| 11 | joint_7_angle | 第4髋关节角度 | true |
| 12 | joint_8_angle | 第4踝关节角度 | true |
| 13 | body_x_velocity | 身体在世界系 x 轴的线速度（前进速度） | true |
| 14 | body_y_velocity | 身体在世界系 y 轴的线速度（横向速度） | true |
| 15 | body_z_velocity | 身体在 z 轴的速度（垂直速度） | true |
| 16 | body_roll_velocity | 身体滚转角速度 | true |
| 17 | body_pitch_velocity | 身体俯仰角速度 | true |
| 18 | body_yaw_velocity | 身体偏航角速度 | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

注：所有观测均可用于奖励，没有禁止字段。四元数可转换为直立度量：`body_up_z = 1 - 2*(quat_x² + quat_y²)`，取值范围大约 `[-1,1]`，`1.0` 表示完美直立。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] 每维
- 动作维度含义：

| dim | name | meaning |
|-----|------|---------|
| 0 | hip_1_torque | 第1髋关节力矩 |
| 1 | ankle_1_torque | 第1踝关节力矩 |
| 2 | hip_2_torque | 第2髋关节力矩 |
| 3 | ankle_2_torque | 第2踝关节力矩 |
| 4 | hip_3_torque | 第3髋关节力矩 |
| 5 | ankle_3_torque | 第3踝关节力矩 |
| 6 | hip_4_torque | 第4髋关节力矩 |
| 7 | ankle_4_torque | 第4踝关节力矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无显式成功条件；时间截断（`truncated = time_limit_reached`）且未提前失败时可视为一种存活成功，但从终止信号角度不具有明确成功标志。
- **failure-like termination**: 身体高度超出 (0.2, 1.0) 范围（即跌落或弹起过高）；任意状态值变为 NaN 或 infinity。
- **ambiguous termination**: 无。
- **truncation**: 达到最大时间步，truncated 为 True，此时 episode 正常结束，不应视为失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`, `x_position`, `y_position`, `distance_from_origin` 等均被明确禁止。  
  说明：虽然原始环境可能提供这些字段，但当前实验环境下已屏蔽，奖励函数中不得使用。

## 7. 可用于奖励函数的信号
- position: 
  - 身体高度 `body_z` (obs[0])
  - 关节角度 (obs[5:13])
- velocity:
  - 前进速度 `body_x_velocity` (obs[13])
  - 横向速度 `body_y_velocity` (obs[14])
  - 垂直速度 `body_z_velocity` (obs[15])
  - 关节速度 (obs[19:27])
- orientation:
  - 四元数 (obs[1:5])，可计算出直立程度 `body_up_z`
  - 角速度 roll/pitch/yaw (obs[16:18])，可额外作用于稳定性
- action/engine:
  - 动作向量 `action` (8,)，可用于惩罚大扭矩、动作变化等
- other:
  - 无。