# 设计理由

## 第 0 步：信号覆盖审计

### 0.1 终止模式分析
- **数据**：20/20 个 episode 全部为 terminated（非 truncated），len=68.5，score 高度为负（-111.8）。
- **推断**：agent 在约 68 步时触发某种终止条件，且终止时距离目标很远（progress_reward 仅为 +1.12，而距离上累积总负值约 -113，说明存在大量未被奖励函数捕获的负面事件）。
- **根据 §5.5**：终止条件包括 crash_or_body_contact、horizontal_position_outside_viewport、body_not_awake_or_settled。在简短 episode 中，最可能的终止原因是 crash（姿态失控坠地）或出界。**当前奖励函数没有明确惩罚 crash 前兆信号（如高角速度、过大横向速度），也没有 gate 护盾保护 agent 不越界。**

### 0.2 观测使用扫描
| 索引 | 含义 | 已使用？ | 备注 |
|---|---|---|---|
| 0 | x_position | ✅ | progress 距离计算 |
| 1 | y_position | ✅ | progress 距离计算 |
| 2 | x_velocity | ✅ | vel_ok |
| 3 | y_velocity | ✅ | vel_ok |
| 4 | body_angle | ✅ | angle_ok + angle_penalty |
| 5 | angular_velocity | ❌ | **完全未使用** |
| 6 | left_contact | ✅ | contact_ok |
| 7 | right_contact | ✅ | contact_ok |

**发现**：obs[5]（角速度）没有被任何组件使用。结合终止模式（疑似姿态失控 crash），角速度是 crash 前兆的关键信号之一。

### 0.3 信号缺口判断
- **信号缺失**：angular_velocity 未被使用，且该维度可能解释 crash 终止。
- 同时，soft_landing active_rate 仅 0.7%——agent 几乎没有处于"近目标、正姿态、低速、双脚触地"的状态，**但 progress 在正向推进（episode_sum_mean=1.12）**。这意味着 agent 可能在"快速接近→但姿态崩溃→坠毁"。
- **判断**：信号缺失（缺少角速度约束）+ 结构性问题（soft_landing 门控因子坍塌）。

### 0.4 僵尸组件检查
- `soft_landing` active_rate=0.7% < 2% → 僵尸组件。
- 原因：乘积形式 `${proximity} * {angle_ok} * {vel_ok} * {contact_ok}$，在 episode 中间阶段至少有一个因子≈0。

## 1. 行为诊断
- **Agent 在做什么？**：快速下降接近目标，但姿态失控（角速度未被约束），导致 crash 终止。progress_reward 是唯一有效的正向驱动，但 agent 为了 maximize progress 而以不安全的方式加速。
- **干预目标**：姿态稳定性——通过对 angular_velocity 施加惩罚，降低姿态崩溃概率，让 agent 有机会活到 soft_landing 可触发的阶段。
- **方向值得继续吗？**：第一轮迭代，骨架为 `progress + soft_landing + angle_penalty`。当前证据是**信号缺失**而非方向错误，因此先增加缺失信号，不重建。

## 2. 干预层级：Level 2 — 结构变换（增加缺失信号组件）

**证据**：第 0 步发现信号缺口（angular_velocity 未使用），soft_landing 为僵尸组件（active_rate<2%）。

**变换选择**：**add 新组件** `angular_velocity_penalty`。
- 使用 hinge 惩罚形式 `max(0, abs(angular_velocity_n) - ANGULAR_VELOCITY_THRESHOLD)`，而非全时二次惩罚。
- 原因：允许 agent 在正常姿态调整中有一定角速度，只在超过安全阈值时才惩罚（防止 crash 前兆）。
- 阈值设在 crash 边界的 60-80%：假设角速度超过 1.0 rad/s 时姿态很快失控，阈值设为 0.5 rad/s。

**同时**：将 `soft_landing` 的乘积改为几何平均，解决坍塌问题（乘积=0 时几何平均仍可>0）。

## 3. 设计校准
- 主信号 per-step：progress_reward episode_sum_mean/len = 1.12/68.5 ≈ 0.0164。
- 新惩罚 per-step（预估）：`0.02 * max(0, ~0.3) ≈ 0.006`，≤ 0.3x 主信号 ✅。
- 几何平均 `(proximity * angle_ok * vel_ok * contact_ok) ** 0.25` 在"接近目标但还没接触"时仍能给出非零信号（如 proximity=0.8, angle_ok=0.7, vel_ok=0.6, contact_ok=0 → 乘积=0，但几何平均在缺失 contact 时为 0 仍不可避——contact 是强条件，需保留。其实几何平均对"一个因子为 0"的情况仍是 0，收益有限。**不如直接降低 soft_landing 对 contact 的依赖**：将 contact_ok 改为 gate，而非乘积因子。但本轮只改一个组件，先加角速度惩罚，soft_landing 留待下一轮）。

**最终修改**：仅新增 `angular_velocity_penalty` 组件。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    Drives the agent to reach the target pad and settle gently.
    v2: added angular velocity hinge penalty to prevent attitude crash.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.2
    ANGLE_PENALTY_WEIGHT = 0.01
    ANGULAR_VELOCITY_PENALTY_WEIGHT = 0.02

    PROXIMITY_THRESHOLD = 0.5    # distance to start shaping soft landing
    ANGLE_THRESHOLD = 0.5        # rad
    VELOCITY_THRESHOLD = 0.5     # sum of absolute linear velocities
    ANGULAR_VELOCITY_THRESHOLD = 0.5  # rad/s, safety boundary for attitude control

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)   # positive when getting closer

    # ---------- 2) soft landing incentive ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / ANGLE_THRESHOLD)
    vel_ok = max(0.0, 1.0 - (abs(x_v_n) + abs(y_v_n)) / VELOCITY_THRESHOLD)
    contact_ok = left_n * right_n   # both support feet on the ground

    soft_landing = LANDING_WEIGHT * proximity * angle_ok * vel_ok * contact_ok

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) angular velocity hinge penalty ----------
    # Penalize excessive angular velocity that leads to attitude crash.
    # Hinge form: only activates when |angvel| exceeds threshold.
    # This is a soft gate precursor — not a full gate, just a disincentive.
    angular_velocity_penalty = (
        -ANGULAR_VELOCITY_PENALTY_WEIGHT
        * max(0.0, abs(angvel_n) - ANGULAR_VELOCITY_THRESHOLD)
    )

    # ---------- aggregate ----------
    total_reward = progress_reward + soft_landing + angle_penalty + angular_velocity_penalty

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: obs[5]（角速度）完全未使用，同时 crash 终止率高，信号缺失 — 缺少角速度约束导致姿态失控。
- **behavior**: agent 快速下降但姿态崩溃，68 步 crash 终止，progress 正向但总分为负。
- **signal**: 缺少角速度约束（crash 前兆信号）；soft_landing 为僵尸组件（active_rate 0.7%，乘积坍塌）。
- **level**: Level 2 — 结构变换（add 新组件 angular_velocity_penalty）。
- **hypothesis**: 通过新增加速速度惩罚，agent 将学会在接近目标时控制姿态稳定性，减少 crash 终止，延长 episode 长度，从而有机会触发 soft_landing 奖励并积累更多 progress。
- **risk**: 角速度惩罚过强可能导致 agent 过度保守、不敢调整姿态，反而降低 progress。系数已校准至主信号的 0.3x 以下，若下一轮出现 len 上升但 progress 停滞，则需削弱此惩罚或转向 gate 形式。