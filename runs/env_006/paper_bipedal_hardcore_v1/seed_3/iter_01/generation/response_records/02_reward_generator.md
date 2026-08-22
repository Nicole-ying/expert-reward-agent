# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- 主学习信号：水平前进速度 -----
    horizontal_speed = obs[2]
    progress_reward = 0.3 * horizontal_speed  # 线性正向驱动

    # ----- 稳定/安全约束 -----
    # 1. 身体倾角惩罚 (hinge，只在倾角过大时生效)
    hull_angle = obs[0]
    angle_threshold = 0.8  # 倾角安全阈值
    posture_penalty = -0.12 * max(0.0, abs(hull_angle) - angle_threshold)

    # 2. 身体角速度惩罚 (quadratic，持续抑制旋转)
    hull_angular_vel = obs[1]
    angular_penalty = -0.06 * (hull_angular_vel ** 2)

    # 3. 垂直速度异常惩罚 (quadratic，抑制弹跳或坠落)
    vertical_speed = obs[3]
    vertical_penalty = -0.15 * (vertical_speed ** 2)

    # ----- 汇总奖励 -----
    total_reward = progress_reward + posture_penalty + angular_penalty + vertical_penalty

    components = {
        "progress_reward": progress_reward,
        "posture_penalty": posture_penalty,
        "angular_penalty": angular_penalty,
        "vertical_penalty": vertical_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与职责选择

**选定任务族**：`locomotion_continuous_control`（双足机器人复杂地形前进）  
**选定奖励职责**（基于 `reward_role_decomposition`）：

- **主学习信号（mandatory）**：`forward_progress` —— 水平前进速度作为核心驱动力
- **稳定/安全约束（conditional）**：`posture_stability` —— 防止身体倾角过大导致摔倒
- **姿态平滑约束（conditional）**：`angular_velocity_penalty` —— 抑制身体急剧旋转
- **垂直异常约束（conditional）**：`vertical_bounce_penalty` —— 惩罚异常弹跳或坠落

**排除的职责及原因**：

- `terminal_success_reward`：`explicit_success_flag_available=false`，info 为空，无法获取到达终点的显式标签，留到后续迭代通过观测推断补充。
- `terminal_failure_penalty`：`explicit_failure_flag_available=false`，同样因 info 缺失无法直接获取摔倒标志；当前用 hinge 倾角惩罚和垂直速度惩罚作为连续前兆信号替代硬覆盖。
- `energy_efficiency`（关节力矩惩罚）：遵循 v1 推荐，优先让 agent 学会稳健前进，能耗优化留到后续迭代；当前动作维度 4，暂不需要强制效率约束。
- `foot_contact_pattern`（步态模式奖励）：环境提供接触信号（obs[12:13]）但步态模式设计复杂，易被代理利用（如原地踏步刷分），留到后续有步态分析需求时再加入。
- `lidar_preview`（激光前瞻惩罚）：观测包含 lidar（obs[14:23]）但 v1 阶段先专注于基于当前状态的稳定前进，前瞻信号适合在出现"突然撞墙"失败模式时再引入。

## 2. 信号映射与公式算子选择

基于 `role_to_signal_mapping` 和 `expert_reward_context.md` 的 Formula Operator Library：

| 职责 | 信号 | 算子 | 数学形式 | 选择理由 |
|---|---|---|---|---|
| `forward_progress` | `obs[2]` (horizontal_speed) | `dense_state_signal` (线性正向) | `w * signal` | 直接奖励前进速度，每步提供连续梯度；速度本身已是变化率，无需 delta |
| `posture_stability` | `obs[0]` (hull_angle) | `dense_state_signal` (hinge penalty) | `-w * max(0, abs(signal) - threshold)` | 倾角超过安全阈值才惩罚，避免在正常姿态范围内压制探索；threshold=0.8 设在可能的摔倒边界（约1.0-1.2 rad）的 60-80% 处 |
| `angular_velocity_penalty` | `obs[1]` (hull_angular_velocity) | `quadratic_penalty` | `-w * signal**2` | 角速度无硬边界，越小越好，用二次持续抑制大幅旋转，避免硬阈值造成不连续 |
| `vertical_bounce_penalty` | `obs[3]` (vertical_speed) | `quadratic_penalty` | `-w * signal**2` | 垂直速度异常（弹跳过强或坠落）需要连续抑制，无具体安全阈值，二次惩罚兼顾小幅抖动抑制和大幅异常惩罚 |

**权重设计原则**：  
- 主信号 `progress_reward` (0.3) 占主导，确保 agent 有持续前进动力  
- 约束惩罚权重较小且相互平衡：`vertical_penalty` (0.15) 略高因垂直弹跳易导致摔倒；`posture_penalty` (0.12) 在 hinge 区域生效；`angular_penalty` (0.06) 最轻，避免过分压制正常转向  
- 总 per-step 预期量级：progress 约 0.1-0.3（取决于速度），惩罚项合计约 -0.02~-0.1（正常步态下），主信号不会被约束淹没

## 3. 设计自检与预期行为

**自检清单**：
- ✅ 每个终止条件是否有前兆软信号？—— 摔倒（前兆：倾角过大→hinge 惩罚，垂直速度异常→quadratic 惩罚）；到达终点（无需前兆，因 episode 自然终止）
- ✅ 任务目标是否有直接进展信号？—— `horizontal_speed` 每步提供梯度
- ✅ 动作维度 4 时效率约束？—— v1 未加，但已评估必要性；后续可加入 `action_efficiency` (quadratic on action)
- ✅ 各组件尺度大致可比？—— progress 0.1-0.3 vs. 惩罚 -0.01~-0.1，正向驱动大于约束

**预期优势**：
- 连续信号避免稀疏问题，agent 从第一步起就能获得前进反馈
- Hinge 倾角惩罚只在危险姿态时介入，不抑制正常倾斜
- 垂直速度惩罚天然惩罚弹跳和坠落，符合复杂地形需求
- 无硬覆盖 terminal 事件，避免梯度断裂

**潜在 failure modes 及后续迭代方向**：
- **"先冲后倒"**：agent 在平整路段猛冲然后摔倒——后续可加入 `soft_health_gate`（当倾斜过大时乘在主信号上衰减）
- **原地弹跳刷分**：若 vertical_penalty 权重不足，agent 可能弹跳保持水平速度——后续可增大 vertical_penalty 权重或加入 `lidar_preview` 抑制
- **缺乏到达终点激励**：agent 在终点前减速——后续可基于 lidar 或速度持续特征构造 `joint_condition_proxy` 作为软完成奖励
- **能耗过高**：动作幅度异常大但完成任务——后续加入 `action_efficiency`（quadratic on action）
- **步态不自然**：接触模式混乱但未摔倒——后续可基于 obs[12:13] 设计步态模式奖励

## 4. 未使用的 obs 信号说明

- `obs[4:11]`（关节角度/角速度）：当前仅通过动作隐含约束；直接惩罚关节角度可能限制必要的大幅动作，留到后续步态优化阶段
- `obs[12:13]`（地面接触）：v1 阶段步态模式设计复杂且易被利用，暂时不用
- `obs[14:23]`（激光测距）：未来信息，适合在出现"撞墙"失败模式后用 `preview_conditioned_reward` 引入
- `action`（4 维力矩）：v1 不加效率惩罚，因当前目标是前进而非节能

所有设计遵循"信号可用性优先"原则，仅使用 `environment_card.md` 中声明的观测维度和索引。
