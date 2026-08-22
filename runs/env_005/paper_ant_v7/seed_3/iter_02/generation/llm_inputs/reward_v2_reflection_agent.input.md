# ⚠️ 上一版代码验证失败
错误信息：Reward v2 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。 (record: runs\env_005\paper_ant_v7\seed_3\iter_02\generation\validations\reward_v2.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由

**改了什么组件**：将原先的独立直立惩罚 `upright_reward` 删除，改为把身体直立度 `body_up_z` 作为**乘法门控**直接作用在前进奖励上，构成 `forward_reward = body_x_vel * max(0, body_up_z)`。保留原有高度 hinge 惩罚作为安全网，系数不变。

**为什么**：
- 上一轮中，19/20 回合因高度/姿态失控提前终止，说明主要失败模式是“为了追求速度而放弃稳定，最终摔倒”。  
- 原先对直立度的独立二次惩罚 active_rate = 100%，但其负奖励量级太小（约 0.055 per step），远小于前进奖励（约 1.43 per step），agent 仍会优先前进而忽略姿态。增大权重可能把整体奖励压入负值，破坏学习。  
- 乘法门控（soft health gate）是一种更强但更自然的耦合：**不倒才能有进展**。当 `body_up_z` 降低时，前进奖励自动衰减，迫使 agent 优先恢复直立再前进；当彻底倾倒（`body_up_z ≤ 0`）时前进奖励为零，此时只有高度惩罚和失败终止结束这一状态，避免无意义刷分。

**数学形式**：
```python
body_up_z  = 1 - 2*(quat_x**2 + quat_y**2)   # [-1,1], 1=完全直立
upright_gate = max(0.0, body_up_z)           # [0,1]，倾斜超过90°时归零
forward_reward = body_x_vel * upright_gate

height_margin_low  = 0.3
height_margin_high = 0.9
height_penalty = max(0.0, height_margin_low - body_z) + max(0.0, body_z - height_margin_high)
height_reward = -height_penalty  * 5.0

total = forward_reward + height_reward
```

**系数校准**：
- 不用额外缩放 `forward_reward`（系数 1.0），因为平均 `body_up_z` 约 0.8~0.9，乘以 gate 后主信号 per‑step 仍约 1.1~1.3，保持在有效学习范围。  
- `height_penalty` 系数保持 5.0，其平均 per‑step 约 0.01，远小于主信号的 0.5 倍，满足“总惩罚负担 ≤ 主信号 0.5 倍”的约束。  
- gate 在安全姿态下几乎不衰减前进奖励，在倾斜时平滑削弱，避免了二次惩罚那种“非零即负”的绝对惩罚。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键信号
    body_z      = obs[0]
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]

    # 身体直立度（向上分量），范围 [-1, 1]
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # 1. 门控前进奖励：只当身体基本直立时才给予前进奖励
    upright_gate = max(0.0, body_up_z)
    forward_reward = body_x_vel * upright_gate

    # 2. 高度安全 Hinge 惩罚（终止阈值 0.2 / 1.0 的提前预警）
    height_margin_low  = 0.3
    height_margin_high = 0.9
    below_margin = max(0.0, height_margin_low - body_z)
    above_margin = max(0.0, body_z - height_margin_high)
    height_penalty = below_margin + above_margin
    height_reward = -5.0 * height_penalty

    total_reward = forward_reward + height_reward

    components = {
        'forward_gated': forward_reward,
        'height_reward': height_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止主因是身体越界，但码中只有高度 hinge 前兆（active 8 %），缺少将直立姿态耦合进核心前进目标的信号。  
- **behavior**: 机器人竭尽全力向前冲，频繁因姿态失控摔倒（19/20 terminated）。  
- **signal**: 缺少姿态门控；原来的独立直立惩罚量级过小，无法与前进奖励竞争。  
- **level**: Level 2  
- **hypothesis**: 把直立度做成乘法门控后，前进奖励随倾斜衰减，迫使 agent 先恢复稳定再加速，应显著降低摔倒终止率。  
- **risk**: `upright_gate` 在完全倾倒时为零，可能使恢复阶段的梯度变弱；但高度 hinge 惩罚提供了额外梯度，且倾倒后 episode 很快终止，因此影响可控。

# 1. Search objective
- target_score: 2000.000000
- current_score: 67.712045
- gap_to_target: 1932.287955
- target_achievement_ratio: 3.386%

# 2. 上一轮奖励函数代码（该轮得分: 67.712045）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- 从 obs 中提取关键信号 -------------------
    body_z      = obs[0]      # 身体高度
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]     # 世界 x 方向前进速度
    body_y_vel  = obs[14]
    body_z_vel  = obs[15]
    roll_vel    = obs[16]
    pitch_vel   = obs[17]
    yaw_vel     = obs[18]

    # ------------------- 派生信号 -------------------
    # body_up_z: 身体在世界坐标系中的“上方向”分量，1 表示完全直立
    # 直接使用环境卡片提供的公式进行计算
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ------------------- 1. 主学习信号：前进速度 -------------------
    # 使用 linear dense_state_signal，每步都有梯度
    forward_reward = body_x_vel   # 正值鼓励前进，负值惩罚后退

    # ------------------- 2. 姿态约束：直立 -------------------
    # 二次惩罚 body_up_z 偏离 1 的程度（越小越好）
    # 在接近 1 时梯度很小，允许适当的身体摆动；倾斜越大惩罚越强
    upright_error = 1.0 - body_up_z
    upright_penalty = upright_error ** 2
    upright_reward = -upright_penalty

    # ------------------- 3. 高度安全约束：hinge penalty -------------------
    # 只在身体高度接近危险区（<0.3 或 >0.9）时施加惩罚
    # 安全区 (0.3~0.9) 内不惩罚，避免持续抑制正常的运动变化
    height_margin_low  = 0.3
    height_margin_high = 0.9
    below_margin = max(0.0, height_margin_low - body_z)
    above_margin = max(0.0, body_z - height_margin_high)
    height_penalty = below_margin + above_margin
    height_reward = -height_penalty

    # ------------------- 组合 -------------------
    w_forward = 1.0
    w_upright = 0.2
    w_height  = 5.0

    total_reward = (
        w_forward * forward_reward +
        w_upright * upright_reward +
        w_height  * height_reward
    )

    components = {
        'forward_reward': w_forward * forward_reward,
        'upright_reward': w_upright * upright_reward,
        'height_reward':  w_height  * height_reward
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=67.712045, len=201.550000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-29.956133, 213.348711]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 288.609528 | 89.4% | 95.7% | 100.0% |
| upright_reward | -11.210514 | -3.5% | 3.5% | 100.0% |
| height_reward | -2.733672 | -0.8% | 0.8% | 8.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 3D 四足机器人连续控制任务。机器人拥有四条腿、八个力矩控制关节，需要在保持身体直立且高度处于健康范围的前提下，尽可能稳定地向前行走或奔跑。主要目标是持续、快速的**前进运动**，而非仅仅保持平衡或存活。次要目标可包括维持身体姿态稳定、动作平滑、能量高效，但这些都服务于前进这一核心目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- 维度含义表（索引 0~26）

| obs index | name | meaning | reward_usable |
|---|---|---|---|
| 0 | body_z | 机器人主体的垂直高度 | true |
| 1 | quat_w | 身体方向四元数实部 | true |
| 2 | quat_x | 身体方向四元数虚部 x | true |
| 3 | quat_y | 身体方向四元数虚部 y | true |
| 4 | quat_z | 身体方向四元数虚部 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 世界 x 方向前进速度 | true |
| 14 | body_y_velocity | 世界 y 方向横向速度 | true |
| 15 | body_z_velocity | 垂直速度 | true |
| 16 | body_roll_velocity | 滚转角速度 | true |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

额外可用派生：  
- body_up_z = 1 - 2*(quat_x² + quat_y²)，范围 [-1,1]，1 表示完全直立，可用于姿态奖励。
- 所有关节角度、速度可用于动作平滑或关节姿态惩罚。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] per joint（对应标准化力矩）

| action dim | name | meaning |
|---|---|---|
| 0 | hip_1_torque | 第 1 髋关节力矩 |
| 1 | ankle_1_torque | 第 1 踝关节力矩 |
| 2 | hip_2_torque | 第 2 髋关节力矩 |
| 3 | ankle_2_torque | 第 2 踝关节力矩 |
| 4 | hip_3_torque | 第 3 髋关节力矩 |
| 5 | ankle_3_torque | 第 3 踝关节力矩 |
| 6 | hip_4_torque | 第 4 髋关节力矩 |
| 7 | ankle_4_torque | 第 4 踝关节力矩 |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 无显式成功终止标志。
- **failure-like termination**:  
  - body_height_outside_healthy_range: 主体垂直高度 ≤ 0.2 或 ≥ 1.0 时立即终止（可分别视为摔倒或腾空失控）。  
  - state_value_outside_finite_range: 任意状态值变为 NaN 或无穷大时终止（数值崩溃）。
- **ambiguous termination**:  
  - truncated = time_limit_reached，仅代表时间耗尽，不能直接诠释为成功或失败。
- **truncation**: 由时间限制触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false  
  （虽然终止原因可推断为高度越界或数值异常，但在 compute_reward 接口中无法获取 terminated 标志，只能通过 next_obs 的有限信息间接判断。）
- allowed_info_fields: []（本環境不允许在 reward 中使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等均不可用。

## 7. 可用于奖励函数的信号
- **position**:  
  - body_z (高度，可做高度保持奖励)  
  - 四元数 → 直立度 body_up_z  
  - 关节角度（可用于姿态正则化）
- **velocity**:  
  - body_x_velocity (世界 x 方向前进速度，核心前进信号)  
  - body_y_velocity, body_z_velocity（横向、垂直速度，可用于惩罚非前进方向运动）
  - 身体角速度 (roll, pitch, yaw) 及关节角速度（可用于平稳性惩罚）
- **orientation**:  
  - 通过四元数计算直立即时状态
- **contact**: 无（此环境无接触力信息）
- **action/engine**:  
  - 8 个关节力矩（可用于动作幅度惩罚、平滑性惩罚）
- **other**:  
  - next_obs 与 obs 的差分可用于瞬时变化量。