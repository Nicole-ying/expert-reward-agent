## 分析

### 1. 当前 agent 发生了什么？（行为证据）
- 所有 20 个评估 episode 都在 150 步内提前终止（`early_terminal`），平均长度 68.55 步，平均得分 -104.19，说明飞行器极快坠毁或飞出边界，完全无法保持存活。
- 奖励组件统计显示：`soft_landing` 每 episode 平均累加 -4.49（负向贡献 58.8%），`progress_reward` +1.68，`global_speed_penalty` -0.92。`soft_landing` 的最大惩罚项未能阻止危险行为，反而成为主要负反馈来源，且由于极度靠近地面时 `landing_factor` 放大惩罚，可能在碰撞瞬间产生巨大负奖励，进一步破坏训练稳定性。
- `progress_reward` 虽然是正向的，但相对于安全着陆的需求，其引导的靠近趋势完全被高速坠落行为覆盖，说明单纯的进度奖励未与安全条件耦合。

### 2. 哪个组件最值得干预？
- **`soft_landing` 组件** 最值得干预。证据：
  - 它占据 magnitude share 的 65.2%，活性 100%，但 signed share 为 -58.8%，是当前奖励中压倒性的负贡献源。
  - 数学形态上，`landing_factor * (contact - speed - angle)` 在未接触时退化为纯惩罚，且靠近目标时惩罚被放大，导致“接近即惩罚”的逆向信号，极可能将 agent 推离靠近目标的正确行为。
  - 且这种独立惩罚既没有阻止坠毁，还破坏了 progress 的正向梯度，形成冲突。
- 正确方向是：将速度/姿态条件从“独立惩罚”转变为“进度奖励的门控因子”，只在危险状态下压制进度奖励，而非另行施加惩罚。这与历史 best（第 1 轮 `gated_goal_reward`）的思路一致。

### 3. 我之前改了什么？
- 第 5 轮基于第 4 轮的 `descent_reward` 体系，改为 `progress_reward + soft_landing + global_speed_penalty`，试图通过乘积式 landing factor 融合着陆奖励与速度惩罚。
- 结果 score 从 best 的 -10.16 跌至 -104.19，且 episode 长度从 1000 步骤降至 68 步。说明这次修改非但未解决问题，反而引入了严重的梯度冲突，导致 agent 彻底放弃存活。
- 因此必须放弃当前结构，回退到 best (`gated_goal_reward + contact_reward`) 的主结构思想，但与 best 不同，必须增加：
  - **真正的着陆成功激励**（best 可能缺乏此信号，致使得分仅在 -10 附近盘旋）。
  - **基于安全状态的门控**，而不是独立惩罚。

### 修改方案
- 主奖励：`progress_reward` = `w_progress * (dist_now - dist_next)`，乘以安全门控因子。
- 安全门控因子由两部分乘积组成：
  1. **坠落抑制门控**：当垂直速度向下过大且离地面很近时，接近 0；否则接近 1。采用光滑形式避免梯度截断。
  2. **姿态门控**：基于 `body_angle` 绝对值，倾角越大，进度奖励被削弱越强。
- 着陆奖励：当双腿同时接触、水平/垂直速度及角度均小于安全阈值时，给予一次性大额正奖励 `landing_bonus`，驱动 agent 完成最终着陆。
- 增加：
  - 横向位置惩罚 `w_x * abs(next_obs[0])`，抑制飞出边界。
  - 引擎使用惩罚 `w_engine * (action != 0)`，鼓励节约燃料。
- 所有组件直接出现在总公式中，不包含未使用的中间变量。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Hyperparameters
    w_progress = 2.0          # 靠近目标的权重
    w_x = 0.2                 # 横向偏离惩罚权重
    w_engine = 0.05           # 引擎点火惩罚权重
    landing_bonus = 20.0      # 成功着陆一次性奖励

    # 安全门控参数
    safe_vy_threshold = 0.5   # 垂直速度上限（绝对值），超过此值视为危险
    safe_height_threshold = 1.0   # 安全高度，低于此高度时激活坠落门控
    k_fall_gate = 10.0        # 坠落门控陡峭度
    k_angle_gate = 5.0        # 姿态门控陡峭度

    # 着陆判断阈值
    contact_thresh = 0.5      # 接触判断（0/1）
    vx_landing_max = 0.5
    vy_landing_max = 0.5
    angle_landing_max = 0.1

    # 计算与目标的距离
    dist_now = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 基础进度奖励：一步间靠近目标
    progress_reward = w_progress * (dist_now - dist_next)

    # 安全门控因子
    # 1) 坠落抑制：垂直速度向下(vy < 0)且离地很近时削弱
    vy = next_obs[3]
    height = next_obs[1]
    fall_danger = max(0.0, -(vy + safe_vy_threshold)) * max(0.0, safe_height_threshold - height)
    fall_gate = 1.0 / (1.0 + k_fall_gate * fall_danger)

    # 2) 姿态门控：倾角越大削弱越强
    angle_gate = 1.0 / (1.0 + k_angle_gate * abs(next_obs[4]))

    safety_gate = fall_gate * angle_gate

    # 着陆奖励：双腿触地且速度/角度都在安全范围内
    left_contact = next_obs[6] > contact_thresh
    right_contact = next_obs[7] > contact_thresh
    low_speed_vx = abs(next_obs[2]) < vx_landing_max
    low_speed_vy = abs(next_obs[3]) < vy_landing_max
    small_angle = abs(next_obs[4]) < angle_landing_max

    if left_contact and right_contact and low_speed_vx and low_speed_vy and small_angle:
        landing_reward = landing_bonus
    else:
        landing_reward = 0.0

    # 横向位置惩罚
    lateral_penalty = -w_x * abs(next_obs[0])

    # 引擎使用惩罚
    engine_penalty = -w_engine * (1.0 if action != 0 else 0.0)

    # 总奖励
    total_reward = progress_reward * safety_gate + landing_reward + lateral_penalty + engine_penalty

    components = {
        "progress_reward": progress_reward * safety_gate,
        "landing_reward": landing_reward,
        "lateral_penalty": lateral_penalty,
        "engine_penalty": engine_penalty
    }
    return float(total_reward), components
```