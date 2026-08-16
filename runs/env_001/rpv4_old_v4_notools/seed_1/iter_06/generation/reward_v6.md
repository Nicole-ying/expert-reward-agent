**1. 诊断报告**

- **evidence**：
  - 最终得分 -108.9，所有 20 个评估 episode 均 early terminated（<150 步且分数 <-50），最高分 -60.6，最低 -145.9。说明绝大多数轨迹以坠毁/失败终止，只有极少数触发了 soft_landing_bonus（但后续仍累积负奖励，导致总分仍然为负）。
  - `proximity_progress` 贡献稳定（ep_sum_mean ≈33.5），活跃率 100%，表明 agent 学会了持续向原点靠近。
  - `attitude_penalty`（-2.6）与 `speed_penalty`（-1.8）量级很小，且对最终得分影响微弱，未能有效抑制高速和大幅度倾斜。
  - `soft_landing_bonus` 活跃率仅 0.2%，ep_sum_mean 30，说明成功着陆事件极度稀疏，条件过严或前置行为未得到引导。
  - checkpoint 趋势：proximity_progress 一直平稳，而 attitude_penalty、speed_penalty 未见有效抑制，soft_landing_bonus 触发率无提高趋势。整体得分在 -70~-120 之间振荡，未趋向改善。

- **behavior_diagnosis**：
  - 策略当前的行为模式是：受 `proximity_progress` 驱动，快速、不顾姿态地直冲原点，导致高速撞击地面或姿态失控，几乎全部坠毁。极少数 episode 偶然满足软着陆条件获得 `soft_landing_bonus`，但因未能保持稳定而仍以负分结束。
  - 这是一种不健康的行为：agent 被单个正信号主导，忽略了姿态和速度约束，缺乏“接近垫面时需转为精细控制”的引导。

- **signal_completeness**：
  - **缺少对腿接触的连续正反馈**：最终成功需要两条支撑腿安全接触垫面，但目前仅在全部条件同时满足时给予一次性奖励。没有渐进信号鼓励腿接触地面，导致该行为极难探索。
  - `attitude_penalty` 与 `speed_penalty` 虽然存在，但力度不足且缺乏接近目标时的阶段增强，无法阻止 agent 的高速冲撞。

- **selected_level**：L2 —— 需要引入一个新的连续奖励组件来填补信号空缺，使腿接触行为得到引导。

- **selected_intervention**：
  - **新增组件 `leg_contact_reward`**：基于双腿接触量 `next_obs[6] + next_obs[7]` 给出正奖励，系数 `2.0`。使得 agent 在接近地面时获得局部正向反馈，鼓励双腿着垫，从而大幅提高触发 `soft_landing_bonus` 的概率。
  - 其他组件保持不变，以保留现有的向心趋近趋势和姿态/速度约束。

- **falsifiable_hypothesis**：
  - 添加 `leg_contact_reward` 后，agent 在接近原点时会获得额外的正奖励来探索腿接触行为，进而更频繁地达到成功条件，`soft_landing_bonus` 的 active_rate 与 ep_sum_mean 应显著提升。同时，成功的稳定着陆行为增多将拉高整体得分，减少 early terminal 比例。

- **expected_next_round**：
  - 下一轮训练后，期望 `leg_contact_reward` 活跃率较高（腿传感器有值阶段），`soft_landing_bonus` 触发次数增加，最终得分显著提升（至少从 -108 改善到 -50 以上），episode 长度可能延长（成功着陆后稳定），terminated 比例下降，truncated 或正常完成比例出现。

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense signal – progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : penalty for linear velocity (encourages slow approach)
      leg_contact_reward  : dense reward for having legs in contact with the ground
      soft_landing_bonus  : event-based – one-time +200 when a stable landing is first achieved
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty ----------
    vx = next_obs[2]
    vy = next_obs[3]
    speed_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---------- leg contact reward (new) ----------
    leg_contact = next_obs[6] + next_obs[7]        # both legs touching ground
    leg_contact_reward = 2.0 * leg_contact

    # ---------- soft landing bonus (one-time success event) ----------
    def success(ob):
        leg_contact_s = (ob[6] + ob[7]) > 1.5           # both legs must touch
        pos_ok = abs(ob[0]) < 0.3                       # horizontal offset
        speed_ok = (ob[2]**2 + ob[3]**2)**0.5 < 0.3     # low linear speed
        angle_ok = abs(ob[4]) < 0.2                      # low tilt
        return leg_contact_s and pos_ok and speed_ok and angle_ok

    old_success = success(obs)
    new_success = success(next_obs)
    landing_event = (not old_success) and new_success

    soft_landing_bonus = 200.0 if landing_event else 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + leg_contact_reward
                    + soft_landing_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "leg_contact_reward": leg_contact_reward,
        "soft_landing_bonus": soft_landing_bonus,
    }
    return float(total_reward), components
```