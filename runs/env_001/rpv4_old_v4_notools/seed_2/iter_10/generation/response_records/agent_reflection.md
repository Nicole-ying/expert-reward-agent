# Response Record

**1. 诊断报告**

- **evidence**  
  - `score=128.16, len=841.6, terminated=10/20, truncated=10/20`：一半 episode 超时，策略学到某种“着陆后停留”行为，但效率不高。  
  - `stable_landing_bonus` ep_sum_mean=494、active_rate=5.9%，且规模远大于其他组件；`stable_contact_bonus` ep_sum_mean=182、active_rate=43.3%。两者合起来 magnitude_share 97.7%，支配奖励。  
  - Checkpoint 趋势：`stable_landing_bonus` 从 cp40% 103.5 快速攀升到 cp60% 645，然后 cp80% 后回落至 522；score 从 cp90% 152.3 跌至 cp100% 121.0，显示后期过拟合/exploit 现象。  
  - `progress` 分量仅 10 左右（约位移 1.3 单位），说明 agent 已经接近目标但不再移动，专注刷接触和着陆奖励。  
  - 观察空间无推力信息，但燃料效率约束（次目标）未有任何体现；缺少 pushing 惩罚或时间惩罚，agent 可以无限逗留刷分。

- **behavior_diagnosis**  
  当前学到的行为模式是：**着陆后不断轻微弹跳并重新触发“稳定着陆事件”以获得巨额奖励，同时长时间保持低速接触状态刷 `stable_contact_bonus`**。由于前者是一次性事件奖励极高（10.0 一次，但 ep_sum 达 494 意味着平均约 49 次/集），agent 学会了在着陆后故意破坏稳定状态（例如轻微弹起）再重新进入稳定状态来重复拿奖。后者则鼓励一旦腿部接触就保持低速，即使姿态歪斜也可以持续获得 0.5/步的奖励。这种 exploit 导致后期 score 下降，且 episode 分布中有 50% 超时（truncated），效率低下。行为模式不健康。

- **signal_completeness**  
  - **已具备**：接近目标（progress）、姿态惩罚、接触/着陆信号。  
  - **缺失或错误**：  
    1. 没有“一次性成功”奖励——当前事件奖励可被重复触发，缺乏防 exploit 机制。  
    2. 稳定接触奖励门槛太低（仅要求 `|vy|<0.2`），没有要求姿态直立，导致歪斜着陆也能持续拿分。  
    3. 完全缺少时间/步骤惩罚，无法推动 agent 快速完成任务，符合次目标（节约燃料→减少步数）。  
  - 必要职责：需要一个**不可重复的着陆成功信号**（或改为持续稳定保持奖励），外加一个**轻量的 per-step 惩罚**，使 agent 不能无限停留刷分。

- **selected_level**: **L2**  
  理由：当前 exploit 源于事件 + 低门槛持续奖励的组合，仅调权重（L1）无法消除重复触发 exploit，必须改变数学形态——将“事件奖励”废弃、转为**仅在完全稳定条件下才给的持续 reward**，并新增 per-step 惩罚。

- **selected_intervention**  
  - **删除** `stable_landing_bonus`（一次性事件奖励）。  
  - **重写** `stable_contact_bonus` 为 `stable_holding_bonus`：只有同时满足腿部接触、极低速度、直立姿态时才给 per-step 奖励（例如 1.0/步），否则为 0。  
  - **新增** `step_penalty`：每步 -0.03，鼓励尽快完成任务。  
  - 保留并略微增大 `progress` 系数（8.0 → 10.0），帮助加快早期下降；保留姿态惩罚和速度惩罚（系数微调）。

- **falsifiable_hypothesis**  
  将着陆奖励改为“完全稳定→持续得分”而非“进入稳定→一次大奖”，能消除弹跳刷分 exploit；加上 step_penalty 后，agent 会被迫快速到达并保持完全稳定，从而提升成功率和得分。预计 landing 相关奖励不再出现异常高峰，episode 长度缩短，score 提升。

- **expected_next_round**  
  策略会学习：快速飞到中心平台，以极低速度和直立姿态轻着陆，然后保持完全稳定尽可能长时间，直到 episode 结束或因 step_penalty 累积而终止。终止率上升，truncated 比例下降，平均 episode 长度缩短，score 上升至接近或超过 200。

**2. 完整 Python 代码**
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (progress shaping) ---
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 10.0 * (dist_prev - dist_next)

    # --- velocity penalty (encourage gentle landing) ---
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty ---
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- stable holding reward: only when perfectly stable on ground ---
    leg_contact = (next_obs[6] > 0.5) or (next_obs[7] > 0.5)
    speed = (vx ** 2 + vy ** 2) ** 0.5
    upright = abs(next_obs[4]) < 0.1
    stable_holding_bonus = 1.0 if (leg_contact and speed < 0.15 and upright) else 0.0

    # --- step penalty to discourage lingering and improve fuel efficiency ---
    step_penalty = -0.03

    # --- combine ---
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + stable_holding_bonus
        + step_penalty
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "stable_holding_bonus": stable_holding_bonus,
        "step_penalty": step_penalty,
    }

    return float(total_reward), components
```
