# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.3         # 软着陆代理奖励权重

    dist_thresh   = 0.3      # 距离阈值（认为足够接近目标垫）
    vel_thresh    = 0.3      # 速度阈值（认为几乎静止）
    angle_thresh  = 0.2      # 姿态角阈值（认为接近竖直）
    contact_thresh = 0.5     # 接触标志阈值（二值化 1.0/0.0）
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty ---
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    stability_penalty = -(w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel)

    # --- Soft landing proxy ---
    near_target = (dist_next < dist_thresh)
    low_speed   = (abs_v_sum < vel_thresh)
    upright     = (abs_angle < angle_thresh)
    both_contact = (next_obs[6] > contact_thresh and next_obs[7] > contact_thresh)

    landing_bonus = w_landing if (near_target and low_speed and upright and both_contact) else 0.0

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及其角色

1. **progress_delta_reward**（主学习信号）  
   - 基于每一步到目标垫距离的变化：`w_progress * (dist(obs) - dist(next_obs))`。  
   - 直接引导飞行器向目标垫靠近，是导航任务的核心驱动力。

2. **stability_penalty**（稳定/安全约束）  
   - 组合惩罚速度大小、身体倾斜角和角速度。  
   - 迫使飞行器在飞行过程中保持较低的扰动，避免剧烈翻滚或高速撞击，为最终稳定着陆创造基础条件。  
   - 权重设置较小（0.01 和 0.005），避免压制接近目标的探索。

3. **soft_landing_proxy**（任务完成近似信号）  
   - 当飞行器**同时满足**：距离目标垫足够近、水平/垂直速度足够小、姿态接近竖直、双脚均触碰地面时，给予一个小额正向奖励。  
   - 它不是一个“成功标志”，而是奖励一个更接近任务完成末期状态的配置。配合 progress_delta 和 stability_penalty，推动飞行器在目标区域内减速并平稳着地。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境提供的 `info` 为空字典，**不存在任何显式 success 或 failure 标志**（`explicit_success_flag_available = false`）。  
- 若强行引入终点成功/失败奖励，就必须“发明”一个不存在的 `info['success']`，这违反安全约定，也会导致奖励函数在真实环境中失效。  
- 因此 v1 完全依赖观测本身（位置、速度、姿态、接触）来构造学习信号，而不依赖终点标志。

## 哪些组件留到后续迭代

- **terminal_success_reward / terminal_failure_penalty**：当环境包装器明确暴露出 success/failure 标志后，可以作为高权重终点奖励加入。  
- **energy_penalty (动作/推力惩罚)**：任务要求省油，但过早惩罚引擎使用可能导致智能体不敢动弹（`agent_afraid_to_move`）。计划在飞行器能稳定接近目标后再加入小型燃料成本项。  
- **time_penalty**：若发现智能体盘旋过久但不完成任务，可以引入每步小额时间惩罚，推动尽快着陆。  
- **gated_reward / curriculum**：门控或课程式奖励可用来处理安全边界或分阶段训练，但目前 v1 保持简单，以免过度约束初学过程。  
- **distance_reward 作为 anchor**：当前 v1 以 progress_delta 为主信号，未同时使用大权重 distance_reward，避免了相互抵消的问题。若后期需要更强的全局引导，可将小权重 distance_reward 作为辅助锚点加入。

## 训练后应观察的典型 failure mode

- **high_reward_without_success**：智能体可能学会了在目标附近来回震荡，不断获得 progress_delta 和 stability 的高分，却永远不形成双脚稳定的触地状态。此时需检查是否因 stability_penalty 太强导致不敢降速，或 soft_landing_proxy 阈值设置过严。  
- **fast_crash_near_goal**：飞行器以较高速度冲向目标垫，虽然 distance 快速下降，但因速度/角度惩罚不足（或权重过小）最终撞地失败。应观察平均速度和姿态分布，必要时调大 stability_penalty 的权重。  
- **only contact, no progress**：如果 soft_landing_proxy 的接近条件太宽松（例如距离阈值太大），智能体可能在别处触地后立即获得小奖励，而不必飞到中央目标垫。此时需要收紧 `dist_thresh` 或增加对位置的明确要求。